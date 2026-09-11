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
import depthai as dai

import ais
import calib
import cam_in_use
import config
import composer
import ocr
import roi
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
    def __init__(self, face=False):
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
        self.tiles = roi.tiles(self.bands, tw=512, th=384)
        self.boats = tracker.Tracker()
        self.ais = ais.Tracker()
        self.geo = calib.load()  # None until calib.py has a confident fit
        self.lock = threading.Lock()
        self.still = None  # (ts, BGR 4000x3000)
        self.mode = "river"
        self.save_crops = True  # dashboard toggle: sightings still logged when off, just no photo
        self.lens = river_lens()
        self.image = river_image()
        self.calib_msg = ""
        self.vid = None  # latest 1080p frame, used by focus calibration
        self.face_box = None
        self.stats = {"started": time.time(), "sightings": 0, "det_msgs": 0}
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
        label = name or "unidentified vessel"
        print(f"SIGHTING #{t.tid} {label} at x={cx:.0f} ({t.px_per_s:.1f}px/s)")
        subprocess.run(
            ["osascript", "-e", f'display notification "{label} on the Hudson" with title "Ship spotter"'],
            capture_output=True,
        )

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
            best = max(scores)[1]
            with open(CAM_JSON, "w") as f:
                json.dump({"lens": best, "sweep": [[lp, round(v, 1)] for v, lp in scores]}, f, indent=1)
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
        if self.water is not None:
            water.outline(img, self.water)
        else:
            for x1, y1, x2, y2 in self.bands:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 60), 4)
        now = time.time()
        self.draw_trails(img, now)
        if self.geo:
            self.draw_ais_labels(img)
        for t in list(self.boats.tracks.values()):
            if now - t.last > 3:  # only boxes the detector is still seeing
                continue
            x1, y1, x2, y2 = (int(v) for v in t.box)
            pad = 18  # small hulls are ~20 px; pad so the box reads at 1080p
            a, b = (x1 - pad, y1 - pad), (x2 + pad, y2 + pad)
            if t.moving:
                cv2.rectangle(img, a, b, (80, 220, 120), 5)
                cv2.putText(img, f"#{t.tid} {t.px_per_s:.0f}px/s", (a[0], a[1] - 12), 0, 1.5, (80, 220, 120), 4)
            else:  # seen but not yet moved far enough to count (or moored)
                cv2.rectangle(img, a, b, (170, 170, 170), 2)
        if self.geo:
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
        raw = os.path.join(OUT, "scans", f"{ts}_raw.jpg")
        cv2.imwrite(raw, still[1])
        flat, found = ocr.flatten(still[1])
        flat_path = os.path.join(OUT, "scans", f"{ts}.jpg")
        cv2.imwrite(flat_path, flat)
        lines = ocr.ocr(flat_path)
        with open(flat_path[:-4] + ".txt", "w") as f:
            f.write("\n".join(lines))
        return {"page_found": found, "image": flat_path, "text": lines}

    # ---------- main loop ----------
    def run(self):
        print("quit Composer:", composer.quit_composer())
        threading.Thread(target=lambda: asyncio.run(self.ais.run(ais.load_key())), daemon=True).start()
        threading.Thread(target=self.serve, daemon=True).start()
        self.vcam = None
        try:
            import pyvirtualcam

            self.vcam = pyvirtualcam.Camera(1920, 1080, FPS, fmt=pyvirtualcam.PixelFormat.BGR)
        except Exception as e:  # OBS Virtual Camera not installed: spotter + scanner still run
            print("webcam disabled:", e)
        with dai.Pipeline() as p:
            self.build(p)
            p.start()
            threading.Thread(target=self.watch_webcam, daemon=True).start()
            threading.Thread(target=self.auto_fit, daemon=True).start()
            print(f"running: {len(self.tiles)} river tiles, webcam {VCAM_NAME if self.vcam else 'off'}, "
                  f"http://127.0.0.1:{PORT}")
            while p.isRunning():
                s = self.still_q.tryGet()
                if s is not None:
                    t0 = time.perf_counter()
                    frame = s.getCvFrame()
                    self.stats["stills"] = self.stats.get("stills", 0) + 1
                    self.stats["still_decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                    with self.lock:
                        self.still = (time.time(), frame)
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

    # ---------- http ----------
    def serve(self):
        svc = self

        class H(BaseHTTPRequestHandler):
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
                elif path == "/status":
                    self._json({
                        **svc.stats, "mode": svc.mode, "calibrated": bool(svc.geo), "save_crops": svc.save_crops,
                        "lens": svc.lens, "calib_msg": svc.calib_msg, "bands": svc.bands, "image": svc.image,
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

        ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", action="store_true", help="run YuNet to meter webcam exposure on a face")
    Service(face=ap.parse_args().face).run()
