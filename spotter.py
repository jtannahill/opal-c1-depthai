"""Opal Lab service: Hudson ship spotter (#1), window webcam (#4), desk scanner (#5).

One DepthAI pipeline owns the Opal C1. Every Camera output runs at 30 fps:
on RVC2 outputs with mismatched fps stall the whole pipeline, and so does the
UVC node on macOS (the device enumerates but never delivers frames).
  * 4K30 -> lossy ImageManip crops over the river bands -> YOLOv6-nano on-chip (~3 fps/tile)
  * 4K30 -> on-device Script forwarding 1 frame in 2 (~6 fps) -> host "latest still" for the
    live stream, crops and scans
  * 1080p30 -> host -> OBS Virtual Camera via pyvirtualcam, if installed (Zoom/Teams webcam)
  * optional YuNet face detector that meters auto-exposure on a face when one appears

Host side tracks boats, suppresses anything that doesn't move, names vessels via
AIS (aisstream.io) once calib.py has fitted the camera geometry, logs sightings
to out/spotter.db, saves crops to out/sightings/, and notifies via macOS.

HTTP on 127.0.0.1:8810 (dashboard at /):
  GET  /status | /sightings | /ais | /frame.jpg?zoom=&cx=&cy=
  POST /scan | /crops/toggle | /calibrate {"bands": [[x1,y1,x2,y2], ...]}
Run: uv run python spotter.py [--face]
"""
import argparse
import asyncio
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import depthai as dai

import ais
import calib
import groundplane
import cam_in_use
import config
import composer
import ocr
import roi
import session as session_mod
import tracker
import water

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
W, H = roi.FRAME_W, roi.FRAME_H
FPS = 30
# Picture profile for the window view, measured Sep 11 2026 (see docs/quality-sweep).
# Contrast +4 and sharpness 1 roughly double measured detail and raise contrast from
# 38 to 57 on the water, at 3.5% clipped highlights; auto-everything blows the window out.
IMAGE_DEFAULTS = {"exp": 1500, "iso": 100, "contrast": 4, "sharpness": 1,
                  "luma": 1, "chroma": 1, "saturation": 0, "brightness": 0}
CAM_JSON = os.path.join(OUT, "cam.json")  # lens position found by the last focus calibration
FIT_EVERY_S = 600
VCAM_NAME = "OBS Virtual Camera"
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def cam_json():
    try:
        with open(CAM_JSON) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def river_lens():
    try:
        return int(cam_json()["lens"])
    except (KeyError, TypeError, ValueError):
        return 120  # measured best for distance on this mount, Sep 11 2026


def river_image():
    saved = cam_json().get("image") or {}
    return {**IMAGE_DEFAULTS, **{k: v for k, v in saved.items() if k in IMAGE_DEFAULTS}}


def apply_image(c, img, lens):
    """Put a whole picture profile on a CameraControl."""
    c.setManualExposure(int(img["exp"]), int(img["iso"]))
    c.setManualFocus(int(lens))
    c.setContrast(int(img["contrast"]))
    c.setSharpness(int(img["sharpness"]))
    c.setLumaDenoise(int(img["luma"]))
    c.setChromaDenoise(int(img["chroma"]))
    c.setSaturation(int(img["saturation"]))
    c.setBrightness(int(img["brightness"]))
    return c


def compass(deg):
    return COMPASS[int((deg % 360) / 22.5 + 0.5) % 16]
# Under the tile load the camera actually delivers ~12.5 fps (not 30), so every 2nd
# frame gives ~6 fps of 4K to the host (~75 MB/s NV12, fine on USB 3). Measured Sep 11 2026.
STILL_EVERY = 2
STILL_SCRIPT = f"""
n = 0
while True:
    f = node.inputs['in'].get()
    n += 1
    if n % {STILL_EVERY} == 0:
        node.outputs['out'].send(f)
"""
HLS_DIR = os.path.join(OUT, "hls")       # live 4K H.264, segmented by ffmpeg
CLIPS_DIR = os.path.join(OUT, "clips")
HD_FLAG = os.path.join(OUT, "hd.on")     # presence = build the 4K encoder branch at startup
HLS_IDLE_S = 25                          # stop encoding to disk once nobody is watching
TILE_W, TILE_H = 768, 576   # larger tiles mean fewer networks reading every 4K frame
STREAM_MAX_W = 2560  # 1x view is downscaled for the stream; zoomed views stay native
PORT = config.PORT


def lossy(inp):
    """Drop stale frames instead of back-pressuring the camera: a blocking input on
    any branch throttles every Camera output to that branch's pace (measured: all
    streams fell to ~0.1 fps with blocking ImageManip inputs)."""
    inp.setBlocking(False)
    inp.setMaxSize(1)


def in_view_region(v):
    """The stretch of water this window can see (config.toml), used until the bearing fit exists."""
    return config.in_view_region(v.lat, v.lon)


def in_band(cx, cy, bands):
    return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in bands)


class Service:
    def __init__(self, face=False, session_secs=None):
        self.face = face
        # Water: the auto-detected mask when present, else the manually drawn boxes.
        # They never mix: drawing boxes deletes the mask, and auto-detect supersedes old
        # boxes (unioning them bridged both panes into one oversized tile strip).
        self.manual_bands = roi.load() if os.path.exists(roi.PATH) else []
        mask = water.load()
        if mask is not None:
            self.water = mask
            self.bands = water.bands(self.water)
        else:
            self.water = None
            self.bands = self.manual_bands
        self.tiles = roi.tiles(self.bands, tw=TILE_W, th=TILE_H)
        self.boats = tracker.Tracker()
        self.ais = ais.Tracker()
        self.geo = calib.load()  # None until calib.py has a confident fit
        self.plane = groundplane.load()  # camera height + horizon row: turns pixels into positions
        self.plane_obs = 0               # unambiguous sightings the last fit had to work with
        self.marker = None               # (x, y, label, ts): last looked-up point, drawn on the view
        self.stopping = False            # set the moment the link drops, so device polling stops
        self.session = session_mod.Session(session_secs) if session_secs else None
        self.lock = threading.Lock()
        self.still = None  # (ts, BGR 4000x3000)
        self.mode = "river"
        # overlay toggles (dashboard checkboxes; the detector is unaffected either way)
        self.overlays = {"ais": True, "bearing": True, "trails": True, "water": True, "boxes": True,
                         "reticle": True, "ranges": True}
        # Off by default: sightings are logged, tracked and boxed either way; this only
        # controls whether a photo of each one is written to disk.
        self.save_crops = False
        self.lens = river_lens()
        self.image = river_image()
        self.calib_msg = ""
        self.vid = None  # latest 1080p frame, used by focus calibration
        self.view_rect = None  # what the dashboard shows (x, y, w, h as 0-1); crops follow it
        self.hd = os.path.exists(HD_FLAG)   # 4K encoder branch built only in HD mode
        self.enc_q = None      # on-camera H.264 bitstream (4K30), HD mode only
        self.hls = None        # ffmpeg process segmenting that bitstream for the browser
        self.hls_last = 0.0    # last time a viewer asked for a segment
        self.rec = None        # {"file": ..., "until": ..., "path": ...} while recording
        self.face_box = None
        self.stats = {"started": time.time(), "sightings": 0, "det_msgs": 0, "chip_c": None}
        os.makedirs(os.path.join(OUT, "sightings"), exist_ok=True)
        os.makedirs(os.path.join(OUT, "scans"), exist_ok=True)
        self.db_path = os.path.join(OUT, "spotter.db")
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.executescript(calib.SCHEMA)
        # one connection is shared by the detection loop and the calibration thread, so
        # writes take this lock; HTTP handlers read through q() on their own connection
        self.db_lock = threading.Lock()

    def q(self, sql, args=()):
        """Read query on a fresh connection (safe from any HTTP thread)."""
        con = sqlite3.connect(self.db_path)
        try:
            return con.execute(sql, args).fetchall()
        finally:
            con.close()

    # ---------- pipeline ----------
    def build(self, p):
        cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        apply_image(cam.initialControl, self.image, self.lens)
        self.ctl = cam.inputControl.createInputQueue()

        src = cam.requestOutput((W, H), dai.ImgFrame.Type.NV12, fps=FPS)
        arch = dai.NNArchive(dai.getModelFromZoo(dai.NNModelDescription("yolov6-nano", platform="RVC2")))
        self.labels = arch.getConfig().model.heads[0].metadata.classes
        self.iw, self.ih = arch.getInputWidth(), arch.getInputHeight()
        self.det_q = []
        for x, y, w, h in self.tiles:
            m = p.create(dai.node.ImageManip)
            m.initialConfig.addCrop(x, y, w, h)
            m.initialConfig.setOutputSize(self.iw, self.ih, dai.ImageManipConfig.ResizeMode.LETTERBOX)
            m.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            m.setMaxOutputFrameSize(self.iw * self.ih * 3)
            lossy(m.inputImage)
            src.link(m.inputImage)
            nn = p.create(dai.node.DetectionNetwork).build(m.out, arch)
            nn.setConfidenceThreshold(0.2)
            lossy(nn.input)
            self.det_q.append(nn.out.createOutputQueue(maxSize=4, blocking=False))

        sc = p.create(dai.node.Script)
        sc.setScript(STILL_SCRIPT)
        lossy(sc.inputs["in"])
        src.link(sc.inputs["in"])
        self.still_q = sc.outputs["out"].createOutputQueue(maxSize=1, blocking=False)

        if self.hd:
            enc = p.create(dai.node.VideoEncoder)
            enc.setDefaultProfilePreset(FPS, dai.VideoEncoderProperties.Profile.H264_MAIN)
            lossy(enc.input)
            src.link(enc.input)
            self.enc_q = enc.bitstream.createOutputQueue(maxSize=30, blocking=False)

        vid = cam.requestOutput((1920, 1080), dai.ImgFrame.Type.NV12, fps=FPS)
        self.vid_q = vid.createOutputQueue(maxSize=1, blocking=False)

        self.face_q = None
        if self.face:
            from depthai_nodes.node import ParsingNeuralNetwork

            farch = dai.NNArchive(dai.getModelFromZoo(dai.NNModelDescription("yunet", platform="RVC2")))
            fm = p.create(dai.node.ImageManip)
            fm.initialConfig.setOutputSize(farch.getInputWidth(), farch.getInputHeight(), dai.ImageManipConfig.ResizeMode.STRETCH)
            fm.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            fm.setMaxOutputFrameSize(farch.getInputWidth() * farch.getInputHeight() * 3)
            lossy(fm.inputImage)
            vid.link(fm.inputImage)
            fnn = p.create(ParsingNeuralNetwork).build(fm.out, farch)
            self.face_q = fnn.out.createOutputQueue(maxSize=2, blocking=False)

    def tile_to_frame(self, tile, d):
        x, y, w, h = tile
        s = max(w / self.iw, h / self.ih)
        px, py = (self.iw * s - w) / 2, (self.ih * s - h) / 2  # undo letterbox padding
        return (x + d.xmin * self.iw * s - px, y + d.ymin * self.ih * s - py,
                x + d.xmax * self.iw * s - px, y + d.ymax * self.ih * s - py)

    # ---------- exposure policy ----------
    def set_mode(self, mode):
        if mode == self.mode:
            return
        c = dai.CameraControl()
        if mode == "river":
            apply_image(c, self.image, self.lens)
        else:  # webcam in use: auto everything, metered on a face if we have one
            c.setAutoExposureEnable()
            c.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
            if self.face_box:
                x1, y1, x2, y2 = self.face_box
                c.setAutoExposureRegion(int(x1 * W), int(y1 * H), max(1, int((x2 - x1) * W)), max(1, int((y2 - y1) * H)))
        self.ctl.send(c)
        self.mode = mode
        print("mode ->", mode)

    def run_session(self):
        """Warm up, check the calibration against live AIS, watch, then write the report."""
        warmup = min(120, max(30, self.session.seconds * 0.25))
        time.sleep(8)
        jpg = self.frame_jpeg(max_w=2560, quality=88)
        self.session.save_frame("overview.jpg", jpg)
        print(f"session: watching for {self.session.seconds / 60:.0f} min "
              f"(calibration check after {warmup / 60:.0f} min)")
        time.sleep(max(0, warmup - 8))
        self.session.verdict = session_mod.verify(self.geo, self.session.checks)
        v = self.session.verdict
        print(f"session: calibration {v['status']} - {v.get('reason', '')}")
        if v["status"] == "drifted":
            self.session.notes.append(
                f"Heading looks {abs(v['drift_deg']):.1f} deg out; {v['suggested_heading']} would fit better.")
        while self.session.remaining > 0:
            time.sleep(min(10, max(1, self.session.remaining)))
        # one more check now that the whole session's sightings are in
        final = session_mod.verify(self.geo, self.session.checks)
        if final["status"] != "unknown":
            self.session.verdict = final
        self.session.save_frame("final.jpg", self.frame_jpeg(max_w=2560, quality=88))
        path = self.session.write(geo=self.geo, plane=self.plane,
                                  extra={"chip temperature": f"{self.stats.get('chip_c')} C",
                                         "detections": self.stats.get("det_msgs"),
                                         "frames": self.stats.get("stills")})
        print("session: report written to", path)
        os._exit(0)

    def watch_composer(self):
        """Opal Composer is relaunched by its login-item helper, and then claims the
        camera: DepthAI loses every stream, the pipeline dies, we restart, and about
        50 s later it happens again. Keep it closed for as long as we hold the camera.
        """
        while True:
            try:
                if composer.running():
                    composer.quit_composer()
                    self.stats["composer_quits"] = self.stats.get("composer_quits", 0) + 1
                    print("composer: reappeared and was quit again")
            except Exception as e:
                print("composer watchdog:", str(e)[:80])
            time.sleep(5)

    def watch_temp(self, dev):
        """Chip temperature, for diagnostics only: never worth crashing over.

        Polling a device whose link has died takes a fatal signal inside
        libdepthai-core, which killed the process before it could restart itself.
        So this stops at the first error instead of probing a dead device.
        """
        while not self.stopping:
            try:
                self.stats["chip_c"] = round(dev.getChipTemperature().average, 1)
            except Exception:
                self.stats["chip_c"] = None
                return
            for _ in range(20):
                if self.stopping:
                    return
                time.sleep(1)

    def watch_webcam(self):
        while True:
            if self.mode == "calibrating":
                time.sleep(1)
                continue
            try:
                self.set_mode("webcam" if self.vcam and cam_in_use.in_use(VCAM_NAME) else "river")
            except OSError:
                pass  # device list changes while apps open/close cameras
            time.sleep(2)

    def at_pixel(self, x, y):
        """What is at this pixel: bearing always, position once the plane is fitted."""
        out = {"x": round(x), "y": round(y)}
        if not self.geo:
            out["error"] = "set the camera position first"
            return out
        rel = math.degrees(math.atan((x - W / 2) / self.geo["f"]))
        out["rel_deg"] = round(rel, 2)
        out["bearing"] = round((self.geo["heading"] + rel) % 360, 1)
        out["compass"] = compass(out["bearing"])
        if self.plane:
            w = groundplane.pixel_to_world(self.geo, self.plane, x, y)
            if w:
                out.update(lat=round(w["lat"], 6), lon=round(w["lon"], 6),
                           range_m=round(w["range_m"]), range_km=round(w["range_m"] / 1000, 2),
                           estimated=bool(self.plane.get("estimated")))
            else:
                out["note"] = "above the horizon: no position, bearing only"
        else:
            out["note"] = "no ground plane yet: bearing only"
        # nearest AIS ship to that spot, by column and (when known) by range
        best, best_d = None, 1e9
        for v in self.ais.recent(max_age=300):
            if not in_view_region(v):
                continue
            px = calib.project(self.geo, v.lat, v.lon)
            if px is None:
                continue
            d = abs(px - x)
            if d < best_d:
                best, best_d = v, d
        if best is not None and best_d < 200:
            out["nearest_ship"] = {"name": best.name or str(best.mmsi), "mmsi": best.mmsi,
                                   "sog": best.sog, "off_by_px": round(best_d)}
        return out

    def locate(self, lat=None, lon=None, bearing=None):
        """Reverse lookup: where does a coordinate or bearing sit in the frame?"""
        if not self.geo:
            return {"error": "set the camera position first"}
        if bearing is None:
            bearing = calib.bearing(self.geo["lat"], self.geo["lon"], lat, lon)
        rel = (bearing - self.geo["heading"] + 540) % 360 - 180
        out = {"bearing": round(bearing % 360, 1), "compass": compass(bearing), "rel_deg": round(rel, 2)}
        if abs(rel) > 80:
            out["in_frame"] = False
            out["note"] = "outside the camera's view"
            return out
        x = W / 2 + self.geo["f"] * math.tan(math.radians(rel))
        out["x"] = round(x)
        out["in_frame"] = 0 <= x < W
        if lat is not None and self.plane:
            xy = groundplane.world_to_pixel(self.geo, self.plane, lat, lon)
            if xy:
                out["x"], out["y"] = round(xy[0]), round(xy[1])
                out["range_m"] = round(groundplane.range_m(self.geo["lat"], self.geo["lon"], lat, lon))
        return out

    def ship_for(self, t):
        """Best AIS match for a tracked box right now, or None."""
        if not self.geo:
            return None
        cx = (t.box[0] + t.box[2]) / 2
        return calib.match(self.geo, cx, [v for v in self.ais.recent(max_age=300) if in_view_region(v)])

    def name_for(self, t):
        hit = self.ship_for(t)
        return (hit.name or str(hit.mmsi)) if hit else None

    def size_range_m(self, t):
        """Stadiametric range from the matched ship's AIS length and the box width.

        Only as good as the hull being broadside: bow-on it reads far too close,
        so it is a fallback for when no ground plane has been fitted.
        """
        v = self.ship_for(t)
        if not v or not v.length:
            return None
        return groundplane.range_from_size(self.geo["f"], t.box[2] - t.box[0], v.length)

    # ---------- sightings ----------
    def on_sighting(self, t):
        now = time.time()
        x1, y1, x2, y2 = t.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Candidates: moving vessels on the midtown Hudson only. The AIS box also covers
        # the harbor and East River, which can't be in view and only add decoys to the fit.
        moving = [v for v in self.ais.recent(max_age=180) if v.sog >= 1.0 and in_view_region(v)]
        name = mmsi = None
        if self.geo:
            hit = calib.match(self.geo, cx, moving)
            if hit:
                name, mmsi = hit.name or str(hit.mmsi), hit.mmsi
        crop_path = None
        with self.lock:
            still = self.still
        if still and self.save_crops:
            img = still[1]  # same 4K frame the tiles came from, so detection coords apply directly
            pad = 120
            a = (max(0, int(x1) - pad), max(0, int(y1) - pad))
            b = (min(W, int(x2) + pad), min(H, int(y2) + pad))
            if self.view_rect:  # keep the photo inside what the dashboard is showing
                vx, vy, vw, vh = self.view_rect
                vx1, vy1 = int(vx * W), int(vy * H)
                vx2, vy2 = int((vx + vw) * W), int((vy + vh) * H)
                a = (max(a[0], vx1), max(a[1], vy1))
                b = (min(b[0], vx2), min(b[1], vy2))
            crop_path = os.path.join(OUT, "sightings", f"{int(now)}_{t.tid}.jpg")
            cv2.imwrite(crop_path, img[a[1]:b[1], a[0]:b[0]])
        self.db_lock.acquire()
        self.db.execute(
            "INSERT INTO sightings VALUES (?,?,?,?,?,?,?,?)",
            (now, t.tid, cx, cy, round(t.px_per_s, 2), name, mmsi, crop_path),
        )
        self.db.execute(
            "INSERT INTO calib_obs VALUES (?,?,?,?,?)",
            (now, t.tid, cx, cy, json.dumps([[v.mmsi, v.name, v.lat, v.lon, v.sog, v.cog] for v in moving])),
        )
        self.db.commit()
        self.db_lock.release()
        self.stats["sightings"] += 1
        if self.session:
            w = self.world_of(t)
            if not crop_path and still:  # a report wants pictures even when the dashboard does not
                pad = 120
                a = (max(0, int(x1) - pad), max(0, int(y1) - pad))
                b = (min(W, int(x2) + pad), min(H, int(y2) + pad))
                crop_path = os.path.join(self.session.dir, f"{int(now)}_{t.tid}.jpg")
                cv2.imwrite(crop_path, still[1][a[1]:b[1], a[0]:b[0]])
            self.session.add({
                "ts": now, "tid": t.tid, "name": name, "cx": cx,
                "bearing": (self.geo["heading"] + math.degrees(math.atan((cx - W / 2) / self.geo["f"]))) % 360 if self.geo else None,
                "range_km": round(w["range_m"] / 1000, 2) if w else None,
                "knots": groundplane.speed_knots(self.geo, self.plane, t) if (self.geo and self.plane) else None,
                "px_per_s": round(t.px_per_s, 1),
                "crop": self.session.keep(crop_path),
            }, cx=cx, cands=[[v.mmsi, v.name, v.lat, v.lon, v.sog, v.cog] for v in moving])
        label = name or "unidentified vessel"
        print(f"SIGHTING #{t.tid} {label} at x={cx:.0f} ({t.px_per_s:.1f}px/s)")
        subprocess.run(
            ["osascript", "-e", f'display notification "{label} on the Hudson" with title "Ship spotter"'],
            capture_output=True,
        )

    # ---------- HD video (4K H.264 straight off the camera) ----------
    def enable_hd(self):
        """Turn on the encoder branch by restarting into HD mode (pipelines are static)."""
        if self.hd:
            return True
        open(HD_FLAG, "w").close()
        print("hd mode: restarting to add the encoder")
        threading.Thread(target=lambda: (time.sleep(0.5), os.execv(sys.executable, [sys.executable, "-u", *sys.argv])), daemon=True).start()
        return False

    def start_hls(self):
        if self.hls and self.hls.poll() is None:
            return
        os.makedirs(HLS_DIR, exist_ok=True)
        for f in os.listdir(HLS_DIR):
            os.remove(os.path.join(HLS_DIR, f))
        self.hls = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS), "-i", "pipe:0",
             "-c", "copy", "-f", "hls", "-hls_time", "2", "-hls_list_size", "6",
             "-hls_flags", "delete_segments+append_list+independent_segments",
             "-hls_segment_type", "fmp4", os.path.join(HLS_DIR, "live.m3u8")],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, bufsize=0)
        print("hd video: started")

    def stop_hls(self):
        if self.hls:
            try:
                self.hls.stdin.close()
                self.hls.terminate()
            except OSError:
                pass
            self.hls = None
            print("hd video: stopped (idle)")

    def start_recording(self, seconds):
        os.makedirs(CLIPS_DIR, exist_ok=True)
        ts = int(time.time())
        raw = os.path.join(CLIPS_DIR, f"{ts}.h264")
        self.rec = {"file": open(raw, "wb"), "until": time.time() + seconds, "raw": raw,
                    "path": os.path.join(CLIPS_DIR, f"{ts}.mp4"), "started": time.time(), "frames": 0}
        return self.rec["path"]

    def _finish_recording(self):
        rec, self.rec = self.rec, None
        rec["file"].close()
        # The camera delivers fewer than FPS frames while detection runs, so mux at the
        # rate actually captured; muxing at 30 made a 5 s clip play as 1.2 s.
        elapsed = max(0.1, time.time() - rec["started"])
        fps = max(1.0, rec["frames"] / elapsed)
        rate = f"{rec['frames']}/{elapsed:.3f}".replace(".", "")  # exact fraction, e.g. 101/8000
        rate = f"{rec['frames'] * 1000}/{int(elapsed * 1000)}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-r", rate,
                        "-i", rec["raw"], "-c", "copy", "-movflags", "+faststart", "-y", rec["path"]],
                       capture_output=True)
        print(f"clip: {rec['frames']} frames over {elapsed:.1f}s -> {fps} fps")
        os.remove(rec["raw"])
        print("clip saved:", rec["path"])

    def pump_video(self):
        """Move encoded frames to whatever wants them (live HLS, recording)."""
        if self.enc_q is None:
            return
        while True:
            pkt = self.enc_q.tryGet()
            if pkt is None:
                break
            if self.hls or self.rec:
                data = pkt.getData().tobytes()
                if self.hls and self.hls.poll() is None:
                    try:
                        self.hls.stdin.write(data)
                        self.hls.stdin.flush()
                    except (BrokenPipeError, ValueError):
                        self.hls = None
                if self.rec:
                    self.rec["file"].write(data)
                    self.rec["frames"] += 1
        if self.rec and time.time() > self.rec["until"]:
            self._finish_recording()
        if self.hls and time.time() - self.hls_last > HLS_IDLE_S and not self.rec:
            self.stop_hls()

    # ---------- calibration ----------
    def calibrate(self, bands=None):
        """New location: re-sweep focus, optionally store new water bands, forget the
        old geometry, then restart the process so the tiles are rebuilt for the bands."""
        self.mode = "calibrating"
        try:
            region = bands or self.bands
            scores = []
            for lp in range(100, 141, 4):
                self.calib_msg = f"focus sweep: lens {lp}"
                self.ctl.send(apply_image(dai.CameraControl(), self.image, lp))
                time.sleep(1.0)  # lens moves + settings apply a few frames late
                s = []
                for _ in range(3):
                    time.sleep(0.15)
                    frame = self.vid
                    if frame is None:
                        continue
                    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for x1, y1, x2, y2 in region:  # bands are 4K coords; vid is 1080p
                        roi_img = g[y1 // 2:y2 // 2, x1 // 2:x2 // 2]
                        if roi_img.size:
                            s.append(cv2.Laplacian(roi_img, cv2.CV_64F).var())
                scores.append((sum(s) / len(s) if s else 0.0, lp))
            best_score, best = max(scores)
            if best_score < 50:
                # a night-time or blocked view scores single digits; 120 scored 1642 in daylight.
                # Accepting that would overwrite a good focus with noise.
                self.calib_msg = f"focus sweep inconclusive (best score {best_score:.1f}); keeping lens {self.lens}"
                print("focus:", self.calib_msg)
                best = self.lens
            with open(CAM_JSON, "w") as f:
                json.dump({**cam_json(), "lens": best, "image": self.image,
                           "sweep": [[lp, round(v, 1)] for v, lp in scores]}, f, indent=1)
            if bands:
                roi.save(bands)
                if os.path.exists(water.MASK_PATH):
                    os.remove(water.MASK_PATH)  # user drew boxes: they define the water now
            else:
                time.sleep(1.0)  # let a frame at the new focus arrive
                self.auto_water(restart=False)
            with self.db_lock:
                calib.reset(self.db)
            self.calib_msg = f"lens {best}; restarting"
            print("calibrated: lens", best, "bands", bands or "unchanged")
        except Exception as e:
            self.calib_msg = f"failed: {e}"
            self.mode = "river"
            return
        time.sleep(1.5)  # let the dashboard read the final message
        os.execv(sys.executable, [sys.executable, "-u", *sys.argv])

    def fit_plane(self):
        """Fit camera height and horizon row from sightings with one unambiguous vessel."""
        if not self.geo:
            return None
        obs = []
        for cx, cy, aj in self.q("SELECT cx, cy, ais_json FROM calib_obs"):
            try:
                cands = [v for v in json.loads(aj) if v[4] >= 1.0]
            except (ValueError, TypeError):
                continue
            if len(cands) == 1:  # only one moving ship nearby, so the match is not a guess
                obs.append((cx, cy, cands[0][2], cands[0][3]))
        self.plane_obs = len(obs)
        plane = groundplane.fit(self.geo, obs)
        if plane:
            groundplane.save(plane)
            self.plane = plane
            print("ground plane:", plane)
        return plane

    def world_of(self, t):
        """Where a tracked boat actually is, once the plane is known."""
        if not (self.geo and self.plane):
            return None
        x1, y1, x2, y2 = t.box
        return groundplane.pixel_to_world(self.geo, self.plane, (x1 + x2) / 2, y2)  # waterline, not centre

    def refit(self):
        """Known position (GPS): refine heading + lens only. Otherwise: full blind fit."""
        g = self.geo
        if g and g.get("source", "").startswith("gps"):
            geo = calib.fit_heading(g["lat"], g["lon"], g["heading"], quiet=True)
            if geo:
                calib.save(geo)
        else:
            geo = calib.fit(quiet=True)
        if geo:
            self.geo = geo
            print("geo fit:", geo)
        if self.geo:
            self.fit_plane()
        return geo

    def auto_water(self, restart=True):
        """Segment water in the current 4K still with SegFormer, save the mask, restart."""
        with self.lock:
            still = self.still
        if not still:
            self.calib_msg = "no frame yet"
            return False
        self.calib_msg = "detecting water (a few seconds)..."
        mask = water.segment(still[1])
        if mask.mean() < 0.002:
            self.calib_msg = "no water found; keeping the current water area"
            return False
        water.save(mask)
        self.calib_msg = f"water found: {mask.mean() * 100:.1f}% of frame, {len(water.bands(mask))} area(s)"
        print("auto water:", self.calib_msg)
        if restart:
            time.sleep(1.5)
            os.execv(sys.executable, [sys.executable, "-u", *sys.argv])
        return True

    def auto_fit(self):
        while True:
            time.sleep(FIT_EVERY_S)
            try:
                self.refit()
            except Exception as e:
                print("geo fit error:", e)

    def draw_trails(self, img, now, span_s=90):
        """Fading path of each moving boat over the last span_s seconds."""
        for t in list(self.boats.tracks.values()):
            if not t.moving:
                continue
            pts = [(int(x), int(y)) for ts, x, y in t.path if now - ts <= span_s]
            for i in range(1, len(pts)):
                a = i / len(pts)  # older segments fade toward the frame
                cv2.line(img, pts[i - 1], pts[i], (int(80 * a), int(220 * a), int(120 * a)), 2 + int(4 * a))
            if pts:
                cv2.circle(img, pts[-1], 7, (80, 220, 120), -1)

    def draw_ais_labels(self, img):
        """Where each in-view AIS ship should be: x from its bearing, y parked on the
        river band under it (no camera tilt model, so height is not measured)."""
        rows = []  # per row: list of occupied (x1, x2) label spans
        for v in sorted(self.ais.recent(max_age=300), key=lambda v: -v.sog):
            if not in_view_region(v):
                continue
            px = calib.project(self.geo, v.lat, v.lon)
            if px is None or not 0 <= px < W:
                continue
            x = int(px)
            band = next((b for b in self.bands if b[0] <= x <= b[2]), None)
            y_mark = band[1] if band else min(b[1] for b in self.bands)
            cog = "" if v.cog < 0 else f" {int(round(v.cog)):03d}T"
            label = f"{v.name or v.mmsi}  {v.sog:.1f}kn{cog}"
            (tw, th), _ = cv2.getTextSize(label, 0, 1.3, 3)
            span = (x - 10, x + tw + 10)
            # first row where this label's measured span doesn't overlap another label
            row = next((i for i, occ in enumerate(rows) if all(span[1] < a or span[0] > b for a, b in occ)), len(rows))
            if row == len(rows):
                rows.append([])
            rows[row].append(span)
            y_text = y_mark - 70 - row * 62
            color = (60, 200, 255) if v.sog >= 1.0 else (150, 150, 150)
            cv2.line(img, (x, y_text + 10), (x, y_mark + 40), color, 3)
            cv2.drawMarker(img, (x, y_mark + 40), color, cv2.MARKER_TRIANGLE_DOWN, 26, 3)
            cv2.rectangle(img, (x - 6, y_text - th - 10), (x + tw + 6, y_text + 8), (0, 0, 0), -1)
            cv2.putText(img, label, (x, y_text), 0, 1.3, color, 3)

    RANGE_RETICLES_M = (500, 1000, 2000, 4000)

    def draw_reticle(self, img):
        """Crosshair at the centre of frame, labelled with where it points and how far."""
        cx, cy = W // 2, H // 2
        col = (200, 200, 200)
        cv2.line(img, (cx - 70, cy), (cx - 20, cy), col, 3)
        cv2.line(img, (cx + 20, cy), (cx + 70, cy), col, 3)
        cv2.line(img, (cx, cy - 70), (cx, cy - 20), col, 3)
        cv2.line(img, (cx, cy + 20), (cx, cy + 70), col, 3)
        cv2.circle(img, (cx, cy), 6, col, -1)
        if not self.geo:
            return
        info = self.at_pixel(cx, cy)
        label = f"{info['bearing']:.0f} {info['compass']}"
        if "range_km" in info:
            label += f"   {'~' if info.get('estimated') else ''}{info['range_km']} km"
        (tw, th), _ = cv2.getTextSize(label, 0, 1.4, 3)
        cv2.rectangle(img, (cx - tw // 2 - 8, cy + 84), (cx + tw // 2 + 8, cy + 84 + th + 14), (0, 0, 0), -1)
        cv2.putText(img, label, (cx - tw // 2, cy + 84 + th + 4), 0, 1.4, col, 3)

    def draw_range_reticles(self, img):
        """Lines across the water at fixed ranges: for a range R, each column's row
        follows from the ground plane, so the line curves with sec(rel)."""
        if not (self.geo and self.plane):
            return
        f, h, y0 = self.geo["f"], self.plane["height_m"], self.plane["y_horizon"]
        for rng in self.RANGE_RETICLES_M:
            pts = []
            for x in range(0, W, 40):
                rel = math.atan((x - W / 2) / f)
                y = y0 + f * h / math.cos(rel) / rng
                if 0 <= y < H:
                    pts.append((x, int(y)))
            if len(pts) < 2:
                continue
            cv2.polylines(img, [np.array(pts, np.int32)], False, (120, 190, 255), 2)
            label = f"{'~' if self.plane.get('estimated') else ''}{rng / 1000:g} km"
            lx, ly = pts[0][0] + 10, pts[0][1] - 10
            (tw, th), _ = cv2.getTextSize(label, 0, 1.1, 2)
            cv2.rectangle(img, (lx - 5, ly - th - 8), (lx + tw + 5, ly + 5), (0, 0, 0), -1)
            cv2.putText(img, label, (lx, ly), 0, 1.1, (120, 190, 255), 2)

    def draw_bearings(self, img):
        """Compass ticks along the top edge from the fitted camera geometry."""
        g = self.geo
        cv2.rectangle(img, (0, 0), (W, 110), (0, 0, 0), -1)
        for deg in range(0, 360, 5):
            rel = (deg - g["heading"] + 540) % 360 - 180
            if abs(rel) >= 80:
                continue
            x = int(W / 2 + g["f"] * math.tan(math.radians(rel)))
            if not 0 <= x < W:
                continue
            major = deg % 10 == 0
            cv2.line(img, (x, 0), (x, 44 if major else 22), (230, 230, 230), 4 if major else 2)
            if major:
                label = f"{deg}" + (f" {compass(deg)}" if deg % 45 == 0 else "")
                cv2.putText(img, label, (x - 40, 96), 0, 1.5, (230, 230, 230), 3)
        cx = W // 2
        cv2.line(img, (cx, 0), (cx, 110), (60, 180, 255), 5)
        cv2.putText(img, f"{g['heading']:.0f} {compass(g['heading'])}", (cx + 14, 70), 0, 1.8, (60, 180, 255), 4)

    # ---------- dashboard frame ----------
    def frame_jpeg(self, zoom=1.0, cx=0.5, cy=0.5, max_w=None, quality=90, rect=None):
        """Latest still with river bands and tracks drawn on, digitally zoomed.

        zoom crops a 1/zoom window of the 4K frame centered at (cx, cy) (0-1
        fractions), clamped to the frame, and sends it at native resolution:
        1x is the full 3840x2160, 4x is a true-pixel 960x540 crop the browser
        scales up, rather than an interpolated resample.
        """
        with self.lock:
            still = self.still
        if not still:
            return None
        img = still[1].copy()
        if self.overlays["water"]:
            if self.water is not None:
                water.outline(img, self.water)
            else:
                for x1, y1, x2, y2 in self.bands:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 60), 4)
        now = time.time()
        if self.overlays["trails"]:
            self.draw_trails(img, now)
        if self.geo and self.overlays["ais"]:
            self.draw_ais_labels(img)
        for t in list(self.boats.tracks.values()) if self.overlays["boxes"] else []:
            if now - t.last > 3:  # only boxes the detector is still seeing
                continue
            x1, y1, x2, y2 = (int(v) for v in t.box)
            pad = 18  # small hulls are ~20 px; pad so the box reads at 1080p
            a, b = (x1 - pad, y1 - pad), (x2 + pad, y2 + pad)
            name = self.name_for(t) if self.geo else None
            w = self.world_of(t)
            kn = groundplane.speed_knots(self.geo, self.plane, t) if (self.geo and self.plane) else None
            tilde = "~" if (self.plane or {}).get("estimated") else ""
            if t.moving:
                cv2.rectangle(img, a, b, (80, 220, 120), 5)
                speed = f"{tilde}{kn:.1f}kn" if kn is not None else f"{t.px_per_s:.0f}px/s"
                if w:
                    dist = f"  {tilde}{w['range_m'] / 1000:.2f}km"
                else:
                    sr = self.size_range_m(t)
                    dist = f"  ~{sr / 1000:.2f}km" if sr else ""
                label = f"#{t.tid} {speed}{dist}" + (f"  {name}" if name else "")
                cv2.putText(img, label, (a[0], a[1] - 12), 0, 1.5, (80, 220, 120), 4)
            else:  # seen but not yet moved far enough to count (or moored)
                cv2.rectangle(img, a, b, (170, 170, 170), 2)
                if name:
                    cv2.putText(img, name, (a[0], a[1] - 10), 0, 1.1, (170, 170, 170), 3)
        if self.marker and time.time() - self.marker[3] < 120:
            mx, my, label, _ = self.marker
            mx, my = int(mx), int(my)
            cv2.drawMarker(img, (mx, my), (80, 200, 255), cv2.MARKER_CROSS, 60, 4)
            cv2.circle(img, (mx, my), 26, (80, 200, 255), 3)
            (tw, th), _ = cv2.getTextSize(label, 0, 1.3, 3)
            cv2.rectangle(img, (mx + 30, my - th - 16), (mx + 40 + tw, my + 6), (0, 0, 0), -1)
            cv2.putText(img, label, (mx + 35, my - 8), 0, 1.3, (80, 200, 255), 3)
        if self.overlays.get("ranges", True):
            self.draw_range_reticles(img)
        if self.overlays.get("reticle", True):
            self.draw_reticle(img)
        if self.geo and self.overlays["bearing"]:
            self.draw_bearings(img)
        if rect:  # free-aspect crop (x0, y0, w, h as 0-1 fractions of the frame)
            rx, ry, rw, rh = (min(max(v, 0.0), 1.0) for v in rect)
            x0, y0 = int(rx * W), int(ry * H)
            cw, ch = max(64, min(int(rw * W), W - x0)), max(36, min(int(rh * H), H - y0))
        else:
            zoom = min(max(zoom, 1.0), 8.0)
            cw, ch = int(W / zoom), int(H / zoom)
            x0 = int(min(max(cx * W - cw / 2, 0), W - cw))
            y0 = int(min(max(cy * H - ch / 2, 0), H - ch))
        img = img[y0:y0 + ch, x0:x0 + cw]
        if max_w and cw > max_w:
            img = cv2.resize(img, (max_w, int(ch * max_w / cw)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    # ---------- scanner ----------
    def scan(self):
        with self.lock:
            still = self.still
        if not still:
            return {"error": "no still yet"}
        ts = int(time.time())
        frame = still[1]
        if self.view_rect:  # scan what the dashboard is showing, not the whole window
            vx, vy, vw, vh = self.view_rect
            frame = frame[int(vy * H):int((vy + vh) * H), int(vx * W):int((vx + vw) * W)]
            if frame.size == 0:
                frame = still[1]
        raw = os.path.join(OUT, "scans", f"{ts}_raw.jpg")
        cv2.imwrite(raw, frame)
        flat, found = ocr.flatten(frame)
        flat_path = os.path.join(OUT, "scans", f"{ts}.jpg")
        cv2.imwrite(flat_path, flat)
        lines = ocr.ocr(flat_path)
        with open(flat_path[:-4] + ".txt", "w") as f:
            f.write("\n".join(lines))
        return {"page_found": found, "image": flat_path, "text": lines}

    # ---------- main loop ----------
    def run(self):
        print("quit Composer:", composer.quit_composer())
        threading.Thread(target=self.watch_composer, daemon=True).start()
        threading.Thread(target=lambda: asyncio.run(self.ais.run(ais.load_key())), daemon=True).start()
        threading.Thread(target=self.serve, daemon=True).start()
        self.vcam = None
        try:
            import pyvirtualcam

            self.vcam = pyvirtualcam.Camera(1920, 1080, FPS, fmt=pyvirtualcam.PixelFormat.BGR)
        except Exception as e:  # OBS Virtual Camera not installed: spotter + scanner still run
            print("webcam disabled:", e)
        dev, attempt = None, 0
        while dev is None:            # unplugged or USB reset: wait for it, however long it takes
            try:
                dev = dai.Device()
            except Exception as e:
                if attempt % 20 == 0:  # once a minute, not on every retry
                    print("waiting for the camera:", str(e)[:80])
                attempt += 1
                time.sleep(3)
        with dai.Pipeline(dev) as p:
            self.build(p)
            p.start()
            threading.Thread(target=self.watch_webcam, daemon=True).start()
            threading.Thread(target=self.auto_fit, daemon=True).start()
            if self.session:
                threading.Thread(target=self.run_session, daemon=True).start()
            threading.Thread(target=lambda: self.watch_temp(dev), daemon=True).start()
            print(f"running: {len(self.tiles)} river tiles, webcam {VCAM_NAME if self.vcam else 'off'}, "
                  f"http://127.0.0.1:{PORT}")
            while p.isRunning():   # falls out of this loop when the device link drops
                try:
                    s = self.still_q.tryGet()
                    if s is not None:
                        t0 = time.perf_counter()
                        frame = s.getCvFrame()
                        self.stats["stills"] = self.stats.get("stills", 0) + 1
                        self.stats["still_decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                        with self.lock:
                            self.still = (time.time(), frame)
                    self.pump_video()
                    v = self.vid_q.tryGet()
                    if v is not None:
                        self.vid = v.getCvFrame()
                        self.vid_ts = time.time()
                        self.stats["vid_frames"] = self.stats.get("vid_frames", 0) + 1
                        if self.vcam:
                            self.vcam.send(self.vid)
                    if self.face_q is not None:
                        f = self.face_q.tryGet()
                        dets = getattr(f, "detections", None) if f is not None else None
                        if dets:
                            b = max(dets, key=lambda d: d.confidence)
                            r = b.rotated_rect.getOuterRect() if hasattr(b, "rotated_rect") else (b.xmin, b.ymin, b.xmax, b.ymax)
                            self.face_box = tuple(r)
                    boxes = []
                    if self.mode == "river":  # auto-exposed frames make the river useless
                        for tile, q in zip(self.tiles, self.det_q):
                            d = q.tryGet()
                            if d is None:
                                continue
                            self.stats["det_msgs"] += 1
                            for det in d.detections:
                                if self.labels[det.label] not in tracker.BOAT_LIKE:
                                    continue
                                x1, y1, x2, y2 = self.tile_to_frame(tile, det)
                                cx_, cy_ = (x1 + x2) / 2, (y1 + y2) / 2
                                on_water = (water.contains(self.water, cx_, cy_) if self.water is not None
                                            else in_band(cx_, cy_, self.bands))
                                if on_water:
                                    boxes.append((x1, y1, x2, y2, det.confidence))
                    for t in self.boats.update(boxes):
                        self.on_sighting(t)
                    time.sleep(0.02)
                except Exception as e:
                    # queues close when the device link drops; treat it as a device loss
                    self.stopping = True
                    print("device error in main loop:", str(e)[:120])
                    break
        # The pipeline stopped: usually the USB link dropped. Restart the whole process,
        # which waits above for the camera to come back rather than dying silently.
        self.stopping = True
        print("pipeline stopped (device link lost); restarting")
        if self.hls:
            self.stop_hls()
        time.sleep(3)
        os.execv(sys.executable, [sys.executable, "-u", *sys.argv])

    # ---------- http ----------
    def serve(self):
        svc = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, obj, code=200):
                body = json.dumps(obj, default=str, indent=1).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def _stream(self, zoom, cx, cy, rect=None):
                """MJPEG: push a new annotated frame each time a new still arrives (~6 fps).
                The browser opens a fresh stream when the zoom window changes."""
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                while True:
                    with svc.lock:
                        ts = svc.still[0] if svc.still else None
                    if ts is None or ts == last:
                        time.sleep(0.03)
                        continue
                    last = ts
                    jpg = svc.frame_jpeg(zoom, cx, cy, max_w=STREAM_MAX_W, quality=80, rect=rect)
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")

            def _bytes(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                        self._bytes(f.read(), "text/html; charset=utf-8")
                elif path in ("/frame.jpg", "/stream.mjpg"):
                    from urllib.parse import parse_qs

                    q = {k: v[0] for k, v in parse_qs(self.path.partition("?")[2]).items()}
                    try:
                        zoom, cx, cy = float(q.get("zoom", 1)), float(q.get("cx", 0.5)), float(q.get("cy", 0.5))
                        rect = tuple(float(v) for v in q["rect"].split(",")) if "rect" in q else None
                        if rect and len(rect) != 4:
                            rect = None
                    except ValueError:
                        zoom, cx, cy, rect = 1.0, 0.5, 0.5, None
                    if path == "/stream.mjpg":
                        return self._stream(zoom, cx, cy, rect)
                    if q.get("raw") == "1":  # unannotated full 4K still (water segmentation, exports)
                        with svc.lock:
                            still = svc.still
                        ok, buf = cv2.imencode(".jpg", still[1], [cv2.IMWRITE_JPEG_QUALITY, 92]) if still else (False, None)
                        return self._bytes(buf.tobytes(), "image/jpeg") if ok else self._json({"error": "no frame yet"}, 503)
                    jpg = svc.frame_jpeg(zoom, cx, cy)
                    if jpg is None:
                        self._json({"error": "no frame yet"}, 503)
                    else:
                        self._bytes(jpg, "image/jpeg")
                elif path.startswith("/hls/"):
                    name = os.path.basename(path[len("/hls/"):])
                    svc.hls_last = time.time()
                    if not svc.hd:
                        svc.enable_hd()
                        return self._json({"error": "starting hd mode, retry in a few seconds"}, 503)
                    if not svc.hls:
                        svc.start_hls()
                    fp = os.path.join(HLS_DIR, name)
                    # ffmpeg writes the playlist only after the first full segment, which
                    # takes a few seconds at the frame rate detection leaves us
                    for _ in range(150 if name.endswith(".m3u8") else 60):
                        if os.path.exists(fp):
                            break
                        time.sleep(0.1)
                    if not os.path.exists(fp):
                        return self._json({"error": "starting hd video"}, 503)
                    ctype = ("application/vnd.apple.mpegurl" if name.endswith(".m3u8")
                             else "video/iso.segment" if name.endswith(".m4s") else "video/mp4")
                    with open(fp, "rb") as f:
                        self._bytes(f.read(), ctype)
                elif path.startswith("/clip/"):
                    name = os.path.basename(path[len("/clip/"):])
                    fp = os.path.join(CLIPS_DIR, name)
                    if name.endswith(".mp4") and os.path.isfile(fp):
                        with open(fp, "rb") as f:
                            self._bytes(f.read(), "video/mp4")
                    else:
                        self._json({"error": "not found"}, 404)
                elif path == "/clips":
                    names = sorted((f for f in os.listdir(CLIPS_DIR) if f.endswith(".mp4")), reverse=True) if os.path.isdir(CLIPS_DIR) else []
                    self._json([{"name": n, "url": f"/clip/{n}", "size": os.path.getsize(os.path.join(CLIPS_DIR, n))} for n in names[:20]])
                elif path.startswith("/crop/"):
                    name = os.path.basename(path[len("/crop/"):])  # basename blocks path traversal
                    fp = os.path.join(OUT, "sightings", name)
                    if name.endswith(".jpg") and os.path.isfile(fp):
                        with open(fp, "rb") as f:
                            self._bytes(f.read(), "image/jpeg")
                    else:
                        self._json({"error": "not found"}, 404)
                elif path == "/ais":
                    # "in view": projects inside the frame once the bearing is fitted;
                    # until then, the midtown Hudson stretch the window looks across.
                    # ?all=1 skips the frame test (needed to align on a ship a bad heading hides).
                    show_all = "all=1" in self.path
                    out = []
                    for v in sorted(svc.ais.recent(), key=lambda v: -v.sog):
                        if not in_view_region(v):
                            continue
                        px = calib.project(svc.geo, v.lat, v.lon) if svc.geo else None
                        if svc.geo and not show_all and (px is None or not 0 <= px < W):
                            continue
                        out.append({"mmsi": v.mmsi, "name": v.name, "sog": v.sog, "cog": v.cog,
                                    "lat": v.lat, "lon": v.lon, "px": round(px) if px is not None else None})
                    self._json({"exact": bool(svc.geo), "vessels": out})
                elif path == "/at":
                    from urllib.parse import parse_qs

                    q = {k: v[0] for k, v in parse_qs(self.path.partition("?")[2]).items()}
                    try:
                        x, y = float(q["x"]), float(q["y"])
                    except (KeyError, ValueError):
                        return self._json({"error": "need x and y in frame pixels"}, 400)
                    info = svc.at_pixel(x, y)
                    label = info.get("compass", "") and f"{info['bearing']:.0f} {info['compass']}"
                    if "range_km" in info:
                        label += f"  {info['range_km']}km"
                    svc.marker = (x, y, label or "?", time.time())
                    self._json(info)
                elif path == "/locate":
                    from urllib.parse import parse_qs

                    q = {k: v[0] for k, v in parse_qs(self.path.partition("?")[2]).items()}
                    try:
                        if "bearing" in q:
                            info = svc.locate(bearing=float(q["bearing"]))
                        else:
                            info = svc.locate(lat=float(q["lat"]), lon=float(q["lon"]))
                    except (KeyError, ValueError):
                        return self._json({"error": "need lat and lon, or bearing"}, 400)
                    if info.get("x") is not None:
                        svc.marker = (info["x"], info.get("y", H * 0.62), f"{info['bearing']:.0f} {info['compass']}", time.time())
                    self._json(info)
                elif path == "/map":
                    # top-down view: tracked boats placed on the water, plus AIS traffic
                    boats = []
                    for t in svc.boats.active_moving():
                        w = svc.world_of(t)
                        if w:
                            boats.append({"tid": t.tid, "name": svc.name_for(t) if svc.geo else None,
                                          "lat": w["lat"], "lon": w["lon"], "range_m": round(w["range_m"]),
                                          "bearing": round(w["bearing"], 1),
                                          "knots": groundplane.speed_knots(svc.geo, svc.plane, t)})
                    ships = []
                    if svc.geo:
                        for v in svc.ais.recent(max_age=300):
                            if in_view_region(v):
                                ships.append({"mmsi": v.mmsi, "name": v.name, "lat": v.lat, "lon": v.lon,
                                              "sog": v.sog, "cog": v.cog,
                                              "range_m": round(groundplane.range_m(svc.geo["lat"], svc.geo["lon"], v.lat, v.lon)),
                                              "bearing": round(calib.bearing(svc.geo["lat"], svc.geo["lon"], v.lat, v.lon), 1)})
                    self._json({"camera": svc.geo, "plane": svc.plane, "boats": boats, "ships": ships})
                elif path == "/status":
                    self._json({
                        **svc.stats, "mode": svc.mode, "calibrated": bool(svc.geo), "save_crops": svc.save_crops,
                        "lens": svc.lens, "calib_msg": svc.calib_msg, "bands": svc.bands, "image": svc.image, "overlays": svc.overlays,
                        "hd_live": bool(svc.hls), "recording": bool(svc.rec), "plane": svc.plane,
                        "water_auto": svc.water is not None,
                        "heading": svc.geo["heading"] if svc.geo else None,
                        "geo": svc.geo,
                        "hfov": round(math.degrees(2 * math.atan(W / 2 / svc.geo["f"])), 1) if svc.geo else None,
                        "compass": compass(svc.geo["heading"]) if svc.geo else None,
                        "calib_obs": svc.q("SELECT COUNT(*) FROM calib_obs")[0][0],
                        "tracks_moving": len(svc.boats.active_moving()), "ais_vessels": len(svc.ais.recent()),
                    })
                elif self.path == "/sightings":
                    rows = svc.q("SELECT * FROM sightings ORDER BY ts DESC LIMIT 50")
                    self._json(rows)
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                if self.path == "/scan":
                    self._json(svc.scan())
                elif self.path == "/record":
                    if svc.rec:
                        return self._json({"error": "already recording"}, 409)
                    if not svc.hd:
                        svc.enable_hd()
                        return self._json({"error": "starting hd mode, retry in a few seconds"}, 503)
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        secs = float(json.loads(self.rfile.read(n) or b"{}").get("seconds", 30))
                    except ValueError:
                        secs = 30
                    secs = max(2.0, min(300.0, secs))
                    path_out = svc.start_recording(secs)
                    self._json({"recording": True, "seconds": secs, "path": path_out,
                                "url": "/clip/" + os.path.basename(path_out)})
                elif self.path == "/view":
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                        if "rect" not in body:
                            return self._json({"view_rect": svc.view_rect})   # query, not a change
                        r = body["rect"]
                        if r is None:
                            svc.view_rect = None          # explicit clear: show the whole frame
                        elif isinstance(r, (list, tuple)) and len(r) == 4:
                            svc.view_rect = tuple(float(v) for v in r)
                        else:
                            # malformed: keep the current view rather than silently dropping it
                            return self._json({"error": "rect must be [x, y, w, h] or null"}, 400)
                    except (ValueError, TypeError):
                        return self._json({"error": "bad rect"}, 400)
                    self._json({"view_rect": svc.view_rect})
                elif self.path == "/plane/estimate":
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        height = float(json.loads(self.rfile.read(n) or b"{}")["height_m"])
                    except (ValueError, KeyError, TypeError):
                        return self._json({"error": "need height_m: the camera's height above the water"}, 400)
                    if not 2 <= height <= 400:
                        return self._json({"error": "height_m should be between 2 and 400"}, 400)
                    y_h = groundplane.horizon_from_mask(svc.water, water.SCALE) if svc.water is not None else None
                    if y_h is None:
                        return self._json({"error": "no water mask: run Auto-detect water first"}, 400)
                    plane = groundplane.estimate(y_h, height)
                    groundplane.save(plane)
                    svc.plane = plane
                    self._json({"ok": True, "plane": plane,
                                "msg": f"estimated from a {height:g} m camera height, horizon at row {y_h:.0f}. "
                                       "Ranges are approximate until the plane is fitted from AIS."})
                elif self.path == "/plane/fit":
                    plane = svc.fit_plane()
                    total = len(svc.q("SELECT 1 FROM calib_obs"))
                    usable = svc.plane_obs
                    if plane:
                        msg = f"ground plane fitted from {plane['n']} sightings"
                    elif not svc.geo:
                        msg = "set the camera position first"
                    elif usable < 8:
                        msg = f"only {usable} sightings have a single moving ship nearby (need 8; {total} logged in total)"
                    else:
                        msg = (f"{usable} usable sightings, but the fit was not trustworthy. "
                               "Align the heading first: a wrong heading makes the ranges nonsense.")
                    self._json({"ok": bool(plane), "plane": plane or svc.plane, "usable": usable, "logged": total, "msg": msg})
                elif self.path == "/overlays":
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                    except ValueError:
                        return self._json({"error": "bad input"}, 400)
                    for k, v in body.items():
                        if k in svc.overlays:
                            svc.overlays[k] = bool(v)
                    self._json({"overlays": svc.overlays})
                elif self.path == "/image":
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                        if body.get("reset"):
                            img = dict(IMAGE_DEFAULTS)
                        else:
                            img = {**svc.image, **{k: int(v) for k, v in body.items() if k in IMAGE_DEFAULTS}}
                        # ranges the RVC2 ISP accepts
                        for k, lo, hi in [("exp", 100, 33000), ("iso", 100, 1600), ("contrast", -10, 10),
                                          ("sharpness", 0, 4), ("luma", 0, 4), ("chroma", 0, 4),
                                          ("saturation", -10, 10), ("brightness", -10, 10)]:
                            img[k] = max(lo, min(hi, img[k]))
                    except (ValueError, TypeError) as e:
                        return self._json({"error": f"bad input: {e}"}, 400)
                    svc.image = img
                    if svc.mode == "river":
                        svc.ctl.send(apply_image(dai.CameraControl(), img, svc.lens))
                    if body.get("save") or body.get("reset"):
                        with open(CAM_JSON, "w") as f:
                            json.dump({**cam_json(), "lens": svc.lens, "image": img}, f, indent=1)
                    self._json({"image": img, "saved": bool(body.get("save") or body.get("reset"))})
                elif self.path == "/water/auto":
                    if svc.mode == "calibrating":
                        return self._json({"error": "calibration running"}, 409)
                    threading.Thread(target=svc.auto_water, daemon=True).start()
                    self._json({"started": True})
                elif self.path == "/fit":
                    geo = svc.refit()
                    n = svc.q("SELECT COUNT(*) FROM calib_obs")[0][0]
                    need = "5+" if svc.geo and svc.geo.get("source", "").startswith("gps") else "8+"
                    self._json({"ok": bool(geo), "geo": geo or svc.geo, "sightings": n,
                                "msg": "bearing fitted" if geo else f"no better fit yet ({n} sightings; needs {need} that beat chance)"})
                elif self.path == "/geo":
                    # manual GPS position + assumed pointing direction
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                        if "align" in body:
                            # user clicked where an AIS ship really is: solve heading (+ lens with 2+ refs)
                            if not svc.geo:
                                return self._json({"error": "set a GPS position first"}, 400)
                            v = svc.ais.vessels.get(int(body["align"]["mmsi"]))
                            if v is None or not v.lat:
                                return self._json({"error": "that ship has no recent AIS position"}, 400)
                            x = float(body["align"]["x"])
                            refs = [r for r in svc.geo.get("refs", []) if r["mmsi"] != v.mmsi]
                            refs.append({"mmsi": v.mmsi, "name": v.name, "lat": v.lat, "lon": v.lon, "x": x})
                            hd, f, err = calib.solve_refs(svc.geo["lat"], svc.geo["lon"], refs, svc.geo["f"])
                            geo = {**svc.geo, "heading": hd, "f": f, "refs": refs, "ref_err_px": err}
                        elif "clear_refs" in body:
                            geo = {k: v for k, v in (svc.geo or {}).items() if k not in ("refs", "ref_err_px")}
                            if not geo:
                                return self._json({"error": "no position set"}, 400)
                        elif "nudge" in body:
                            if not svc.geo:
                                return self._json({"error": "set a position first"}, 400)
                            geo = {**svc.geo, "heading": (svc.geo["heading"] + float(body["nudge"])) % 360}
                        else:
                            lat, lon = float(body["lat"]), float(body["lon"])
                            hd, hfov = float(body["heading"]) % 360, float(body.get("hfov") or 72)
                            if not (-90 <= lat <= 90 and -180 <= lon <= 180 and 20 <= hfov <= 150):
                                raise ValueError("out of range")
                            geo = {"lat": lat, "lon": lon, "heading": hd, "f": round(calib.f_from_hfov(hfov), 1)}
                        geo["source"] = "gps"
                    except (ValueError, KeyError, TypeError) as e:
                        return self._json({"error": f"bad input: {e}"}, 400)
                    calib.save(geo)
                    svc.geo = geo
                    self._json({"ok": True, "geo": geo})
                elif self.path == "/crops/toggle":
                    svc.save_crops = not svc.save_crops
                    self._json({"save_crops": svc.save_crops})
                elif self.path == "/calibrate":
                    if svc.mode == "calibrating":
                        return self._json({"error": "already calibrating"}, 409)
                    n = int(self.headers.get("Content-Length") or 0)
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                        bands = [tuple(int(v) for v in b) for b in body.get("bands") or []]
                        bands = [(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)) for x1, y1, x2, y2 in bands]
                        bands = [b for b in bands if b[2] - b[0] >= 64 and b[3] - b[1] >= 32
                                 and 0 <= b[0] and b[2] <= W and 0 <= b[1] and b[3] <= H]
                    except (ValueError, TypeError):
                        return self._json({"error": "bad bands"}, 400)
                    threading.Thread(target=svc.calibrate, args=(bands or None,), daemon=True).start()
                    self._json({"started": True, "bands": bands or "unchanged"})
                else:
                    self._json({"error": "not found"}, 404)

            def log_message(self, *a):
                pass

            def handle(self):
                # the dashboard aborts in-flight frame requests on zoom/refresh; not an error
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hudson ship spotter on a repurposed Opal C1")
    ap.add_argument("--face", action="store_true", help="run YuNet to meter webcam exposure on a face")
    ap.add_argument("--session", metavar="DURATION",
                    help="watch for a fixed spell (20m, 45s, 1h), write a report, then exit")
    args = ap.parse_args()
    secs = session_mod.parse_duration(args.session) if args.session else None
    Service(face=args.face, session_secs=secs).run()
