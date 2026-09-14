"""B-cam mode: the Opal as the second angle beside an FX30, cut together in post.

The webcam look comes from sitting two feet from the lens, not from the lens.
So this mode assumes you sit back at a normal interview distance and takes its
framing out of the 48 MP sensor instead, which gives a tight angle with flat,
unexaggerated perspective.

Where the work went
-------------------
The first build did the tracked crop on the device, with an ImageManip and
YuNet in the pipeline, and delivered 5.9 fps. Measuring one stage at a time
(probe_bcam_load*.py, probe_bcam_sustained.py, probe_bcam_nonn.py) showed two
things worth writing down:

  - The on-device NN cost about 30% of throughput. With YuNet in the pipeline
    nothing beat 21.6 fps sustained at any resolution. With it gone, 4K at 24
    holds 99.6% and 1440p at 30 holds 99.3%.
  - Short measurements lie. A 15 s run reported 97-103% for configurations that
    sustain 85-90% over 60 s, because the warmup backlog sitting in the output
    queue gets counted. Anything timing this pipeline has to drain first.

So the device now does the one thing it is good at: encode a clean full frame
and hand over a cheap second stream. No ImageManip, no NN. Face detection runs
on the host through macOS Vision at about 5 ms a frame.

That also turns out to be the better answer for an edit. The recorded file is
the full sensor frame at full rate, so you reframe on the timeline with the
whole image available instead of living with a crop baked in on the day. The
damped tracking drives the live feed, and its path is written to the sidecar as
a reframe curve you can apply in post if you want it rather than hand-keying.

Everything that could drift between the two cameras is nailed down: exposure,
focus and white balance are manual and never touched once the take starts.

The Opal records no audio. Sync by clapping once on camera at the top.

    uv run python interview.py --seconds 1200
    uv run python interview.py --preview       # frame up, record nothing
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import cv2
import depthai as dai
import numpy as np

import composer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# Recorded stream: the full sensor frame, cropped later on the timeline.
# 3840x2160 at 24 measured 99.6% of nominal sustained over 60 s.
REC_W, REC_H = 3840, 2160

# Second stream for the live feed and for face detection. Cropping a 1280x720
# window out of it is a 1.5x tighten at native resolution, no upscale.
LIVE_W, LIVE_H = 1920, 1080
VCAM_W, VCAM_H = 1280, 720

# How often to ask Vision for a face. The framer is damped over seconds, so
# there is nothing to gain from running it every frame.
DETECT_HZ = 6.0

# Where the face sits in the frame. 0.38 puts the eyes near the upper third.
FACE_Y = 0.38

# Easing, in units of the live tap. ALPHA closes that fraction of the remaining
# error per detection, DEADBAND_FRAC ignores detector jitter, MAX_STEP_PX stops
# a bad detection from snapping the frame.
ALPHA = 0.10
DEADBAND_FRAC = 0.03
MAX_STEP_PX = 8

# Recording bitrate. The default preset lands near 16 Mbps at 4K, which is thin
# beside FX30 footage and shows on gradients. 60 Mbps is closer to a match and
# still holds full rate.
BITRATE_KBPS = 60000

# Locked look. These are only fallbacks: run --calibrate once from the chair and
# the measured values land in config.toml under [interview]. The spotter's own
# values are metered against a bright window and are wrong for a face.
LOOK_DEFAULTS = {
    "exposure_us": 8000,
    "iso": 200,
    "lens_position": 145,
    "white_balance_k": 5000,
    "contrast": 0,
    "sharpness": 1,
    "saturation": 0,
}


def load_look():
    """[interview] out of config.toml, falling back per key."""
    import tomllib
    look = dict(LOOK_DEFAULTS)
    path = os.path.join(HERE, "config.toml")
    if os.path.exists(path):
        with open(path, "rb") as f:
            look.update({k: v for k, v in tomllib.load(f).get("interview", {}).items() if k in look})
    return look


def save_look(look):
    """Replace the [interview] block in config.toml, leaving the rest alone."""
    path = os.path.join(HERE, "config.toml")
    text = open(path).read() if os.path.exists(path) else ""
    out, skipping = [], False
    for line in text.splitlines():
        if line.strip().startswith("["):
            skipping = line.strip() == "[interview]"
        if not skipping:
            out.append(line)
    while out and not out[-1].strip():
        out.pop()
    out += ["", "[interview]", "# Written by interview.py --calibrate. Re-run it if the room changes."]
    out += [f"{k} = {v}" for k, v in look.items()]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class VisionFaces:
    """macOS Vision face rectangles. Normalized, origin bottom-left, y up."""

    def __init__(self):
        import Quartz
        import Vision
        from Foundation import NSData

        self._Q, self._V, self._NSData = Quartz, Vision, NSData

    def _cgimage(self, bgr):
        Q = self._Q
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        data = self._NSData.dataWithBytes_length_(rgb.tobytes(), rgb.nbytes)
        prov = Q.CGDataProviderCreateWithCFData(data)
        return Q.CGImageCreate(w, h, 8, 24, w * 3, Q.CGColorSpaceCreateDeviceRGB(),
                               Q.kCGBitmapByteOrderDefault | Q.kCGImageAlphaNone,
                               prov, None, False, Q.kCGRenderingIntentDefault)

    def largest_box(self, bgr):
        """Biggest face as (x1, y1, x2, y2) in pixels of bgr, or None."""
        V = self._V
        req = V.VNDetectFaceRectanglesRequest.alloc().init()
        handler = V.VNImageRequestHandler.alloc().initWithCGImage_options_(self._cgimage(bgr), None)
        ok, _ = handler.performRequests_error_([req], None)
        res = req.results() if ok else None
        if not res:
            return None
        h, w = bgr.shape[:2]
        f = max(res, key=lambda r: r.boundingBox().size.width * r.boundingBox().size.height)
        bb = f.boundingBox()
        x1 = bb.origin.x * w
        x2 = (bb.origin.x + bb.size.width) * w
        # Vision's y runs up from the bottom; image rows run down, so the box flips.
        y1 = (1.0 - (bb.origin.y + bb.size.height)) * h
        y2 = (1.0 - bb.origin.y) * h
        return x1, y1, x2, y2

    def largest(self, bgr):
        """Biggest face centre as (cx, cy) in pixels of bgr, or None."""
        box = self.largest_box(bgr)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2


class Framer:
    """Damped crop centre over the live tap. Holds still when no face is seen."""

    def __init__(self, crop_w, crop_h, src_w, src_h):
        self.crop_w, self.crop_h = crop_w, crop_h
        self.src_w, self.src_h = src_w, src_h
        self.cx, self.cy = src_w / 2, src_h / 2
        self.deadband = crop_w * DEADBAND_FRAC
        self.seen = False

    def ease_to_face(self, fcx, fcy):
        self.seen = True
        tx = fcx
        ty = fcy + (0.5 - FACE_Y) * self.crop_h
        for axis, target in (("cx", tx), ("cy", ty)):
            cur = getattr(self, axis)
            err = target - cur
            if abs(err) < self.deadband:
                continue
            setattr(self, axis, cur + clamp(err * ALPHA, -MAX_STEP_PX, MAX_STEP_PX))
        self.cx = clamp(self.cx, self.crop_w / 2, self.src_w - self.crop_w / 2)
        self.cy = clamp(self.cy, self.crop_h / 2, self.src_h - self.crop_h / 2)

    def rect(self):
        x = int(round(self.cx - self.crop_w / 2))
        y = int(round(self.cy - self.crop_h / 2))
        return x, y, self.crop_w, self.crop_h

    def normalized(self):
        """Crop as fractions of the frame, so it maps onto the 4K recording."""
        x, y, w, h = self.rect()
        return [round(x / self.src_w, 5), round(y / self.src_h, 5),
                round(w / self.src_w, 5), round(h / self.src_h, 5)]


def apply_look(c, look):
    c.setManualExposure(look["exposure_us"], look["iso"])
    c.setManualFocus(look["lens_position"])
    c.setManualWhiteBalance(look["white_balance_k"])
    c.setContrast(look["contrast"])
    c.setSharpness(look["sharpness"])
    c.setSaturation(look["saturation"])


def build(pipeline, fps, record, look):
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    apply_look(cam.initialControl, look)

    # Both outputs at the same rate: asking the camera for mixed rates stalls
    # the whole pipeline to a fraction of a frame per second.
    live = cam.requestOutput((LIVE_W, LIVE_H), dai.ImgFrame.Type.NV12, fps=fps)
    enc = None
    if record:
        rec = cam.requestOutput((REC_W, REC_H), dai.ImgFrame.Type.NV12, fps=fps)
        enc = pipeline.create(dai.node.VideoEncoder)
        enc.setDefaultProfilePreset(fps, dai.VideoEncoderProperties.Profile.H264_MAIN)
        enc.setKeyframeFrequency(fps)
        enc.setBitrateKbps(BITRATE_KBPS)
        enc.input.setBlocking(False)
        enc.input.setMaxSize(1)
        rec.link(enc.input)
    return live, enc


def probe_frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    n = r.stdout.strip().rstrip(",")
    return int(n) if n.isdigit() else None



def wait_for_device(timeout=40, settle=6.0):
    """Whatever held the camera leaves USB busy for a while after it dies.

    Starting a pipeline into that window crashes the device, and it recovers on
    the next run, which is how this hid twice. The device reports itself
    available before it is actually ready, so appearing in the list is not
    enough on its own: wait, then let callers confirm frames really flow.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if dai.Device.getAllAvailableDevices():
            time.sleep(settle)
            return True
        time.sleep(1.0)
    return False


def frames_flowing(q, timeout=6.0):
    """A crashed device still hands back a queue; it just never fills it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if q.tryGet() is not None:
            return True
        time.sleep(0.05)
    return False


def calibrate(fps=24, use_face=True, attempts=3):
    for attempt in range(1, attempts + 1):
        if _calibrate_once(fps, use_face):
            return
        if attempt < attempts:
            print(f"retrying ({attempt + 1} of {attempts}); letting the device settle")
            time.sleep(8)
    print("gave up: the camera would not come up cleanly")


def _calibrate_once(fps=24, use_face=True):
    """Measure the look from the chair: expose for the face, focus on the face.

    Auto exposure is only used as an instrument. It is pointed at the face,
    allowed to settle, read back off the frame metadata, and then thrown away;
    the take itself runs on the fixed numbers this writes to config.toml.

    The 180 degree mount means an exposure region in sensor coordinates is not
    the box Vision found, so both orientations are tried and the one that
    actually brightens the face wins. This is the same ambiguity probe_face.py
    turned up, settled by measurement rather than by reasoning about it.
    """
    faces = VisionFaces()
    if composer.quit_composer():
        print("closed Opal Composer to take the camera")
    if not wait_for_device():
        print("no camera showed up; is the spotter still holding it?")
        return False

    with dai.Pipeline() as p:
        cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        cam.initialControl.setAutoExposureEnable()
        cam.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        live = cam.requestOutput((LIVE_W, LIVE_H), dai.ImgFrame.Type.NV12, fps=fps)
        q = live.createOutputQueue(maxSize=3, blocking=False)
        ctl = cam.inputControl.createInputQueue()
        p.start()
        if not frames_flowing(q):
            print("camera came up but no frames arrived: it crashed on start")
            return False

        def grab(n=12):
            for _ in range(n):
                f = q.get()
            return f

        def find_face(timeout=30):
            print("sit where the subject sits and look at the lens...")
            deadline = time.time() + timeout
            hits, tries = [], 0
            while time.time() < deadline:
                tries += 1
                box = faces.largest_box(grab(3).getCvFrame())
                if box:
                    hits.append(box)
                    if len(hits) >= 6:
                        # Median over the window, so one bad box cannot set the
                        # room up and one missed frame does not reset the count.
                        return tuple(float(np.median([h[i] for h in hits[-6:]])) for i in range(4))
                if tries % 20 == 0:
                    print(f"  still looking ({len(hits)} detections so far)...")
            return None

        def centre_box():
            """Stand-in region: whatever is in the middle of the frame."""
            w, h = LIVE_W * 0.22, LIVE_H * 0.33
            return (LIVE_W / 2 - w / 2, LIVE_H / 2 - h / 2,
                    LIVE_W / 2 + w / 2, LIVE_H / 2 + h / 2)

        box = find_face() if use_face else centre_box()
        if box is None:
            print("no face found in 30 s; nothing measured, config.toml untouched")
            print("if you are setting up with a stand-in or a chart, use --no-face")
            return True
        if not use_face:
            print("using the centre of frame as the subject; put a stand-in in the chair")
        x1, y1, x2, y2 = box
        fw, fh = x2 - x1, y2 - y1
        print(f"face {fw:.0f}x{fh:.0f} px in the {LIVE_W}x{LIVE_H} tap")

        def face_mean(frame):
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return g[int(y1):int(y2), int(x1):int(x2)].mean()

        # --- exposure, metered on the face ---
        sx, sy = REC_W / LIVE_W, REC_H / LIVE_H
        best = None
        for label, (rx1, ry1, rx2, ry2) in {
            "direct": (x1, y1, x2, y2),
            "flipped": (LIVE_W - x2, LIVE_H - y2, LIVE_W - x1, LIVE_H - y1),
        }.items():
            c = dai.CameraControl()
            c.setAutoExposureEnable()
            c.setAutoExposureRegion(int(rx1 * sx), int(ry1 * sy),
                                    max(1, int((rx2 - rx1) * sx)), max(1, int((ry2 - ry1) * sy)))
            ctl.send(c)
            time.sleep(3.0)
            f = grab(20)
            mean = face_mean(f.getCvFrame())
            print(f"  AE region {label:8} face brightness {mean:5.1f} "
                  f"(exp {f.getExposureTime().total_seconds() * 1e6:.0f} us, iso {f.getSensitivity()})")
            # 128 is a mid grey; the region that lands nearest it is the one
            # the sensor actually understood.
            if best is None or abs(mean - 128) < abs(best[0] - 128):
                best = (mean, f, label)
        mean, f, label = best
        exposure_us = int(f.getExposureTime().total_seconds() * 1e6)
        iso = int(f.getSensitivity())
        wb = int(f.getColorTemperature())
        print(f"exposure locked from the {label} region: {exposure_us} us, iso {iso}, {wb} K")
        if mean < 60:
            print("  warning: the face is still dark. Something bright is behind you;")
            print("  move the subject off the window before trusting these numbers.")
        elif mean > 200:
            print("  warning: the face is near clipping; add negative fill or close a blind.")

        # --- focus, scored inside the face box only ---
        c = dai.CameraControl()
        c.setManualExposure(exposure_us, iso)
        c.setManualWhiteBalance(wb)
        ctl.send(c)
        time.sleep(1.5)

        def sharpness_at(lp):
            c = dai.CameraControl()
            c.setManualFocus(lp)
            ctl.send(c)
            g = cv2.cvtColor(grab(14).getCvFrame(), cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(g[int(y1):int(y2), int(x1):int(x2)], cv2.CV_64F).var()

        print("sweeping focus, hold still...")
        coarse = [(lp, sharpness_at(lp)) for lp in range(80, 211, 10)]
        for lp, sc in coarse:
            print(f"  lens {lp:3d}: {sc:8.1f}", flush=True)
        peak = max(coarse, key=lambda t: t[1])[0]
        fine = [(lp, sharpness_at(lp)) for lp in range(max(0, peak - 8), min(255, peak + 9), 2)]
        for lp, sc in fine:
            print(f"  lens {lp:3d}: {sc:8.1f}", flush=True)
        lens, score = max(fine, key=lambda t: t[1])
        print(f"focus locked at lens {lens} (sharpness {score:.0f})")

        # --- framing check, so CROP_SPAN and FACE_Y get a sanity read ---
        c = dai.CameraControl()
        c.setManualFocus(lens)
        ctl.send(c)
        time.sleep(1.0)
        frame = grab(10).getCvFrame()
        framer = Framer(VCAM_W, VCAM_H, LIVE_W, LIVE_H)
        for _ in range(400):
            framer.ease_to_face((x1 + x2) / 2, (y1 + y2) / 2)
        cx, cy, cw, ch = framer.rect()
        crop = np.ascontiguousarray(frame[cy:cy + ch, cx:cx + cw])
        cv2.imwrite(os.path.join(OUT, "calibrate-framing.jpg"), crop)
        head_frac = fh / ch
        eye_frac = ((y1 + fh * 0.4) - cy) / ch
        print(f"\nframing: head height {head_frac * 100:.0f}% of frame, "
              f"eyeline {eye_frac * 100:.0f}% down")
        if head_frac < 0.18:
            print("  you are framed wide for a B-cam; sit closer or lower CROP_SPAN")
        elif head_frac > 0.55:
            print("  that is a big close-up; sit back or raise CROP_SPAN")
        print("  check out/calibrate-framing.jpg before you shoot")

    look = dict(LOOK_DEFAULTS)
    look.update({"exposure_us": exposure_us, "iso": iso,
                 "lens_position": lens, "white_balance_k": wb})
    save_look(look)
    print(f"\nwrote [interview] to config.toml: {look}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=24, help="must match the A-cam; 24 and 30 both hold")
    ap.add_argument("--seconds", type=float, default=1200)
    ap.add_argument("--preview", action="store_true", help="frame up only, record nothing")
    ap.add_argument("--no-virtualcam", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-face", action="store_true",
                    help="with --calibrate, meter the centre of frame instead of a face")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure exposure and focus from the chair, write config.toml")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.calibrate:
        calibrate(args.fps, use_face=not args.no_face)
        return
    record = not args.preview
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(OUT, f"{args.name + '-' if args.name else ''}bcam-{stamp}")

    if composer.quit_composer():
        print("closed Opal Composer to take the camera")

    look = load_look()
    faces = VisionFaces()
    framer = Framer(VCAM_W, VCAM_H, LIVE_W, LIVE_H)

    vcam = None
    if not args.no_virtualcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=VCAM_W, height=VCAM_H, fps=args.fps,
                                       fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"virtual camera: {vcam.device}")
        except Exception as e:
            print(f"no virtual camera ({e}); continuing without it")

    print(f"record {REC_W}x{REC_H} full frame at {args.fps} fps; live {VCAM_W}x{VCAM_H} tracked crop")
    print("clap once on camera at the top of the take for sync")

    curve = []
    with dai.Pipeline() as p:
        live, enc = build(p, args.fps, record, look)
        ql = live.createOutputQueue(maxSize=3, blocking=False)
        qb = enc.bitstream.createOutputQueue(maxSize=120, blocking=False) if record else None

        p.start()
        time.sleep(3)
        # Drop the warmup backlog, or the measured rate is flattered by frames
        # that were encoded before the clock started.
        while ql.tryGet() is not None:
            pass
        if qb is not None:
            while qb.tryGet() is not None:
                pass

        clip = open(base + ".h264", "wb") if record else None
        shown = packets = 0
        ts_first = ts_last = None
        last_crop = None
        next_detect = 0.0
        t0 = time.time()
        try:
            while time.time() - t0 < args.seconds:
                if qb is not None:
                    b = qb.tryGet()
                    if b is not None:
                        clip.write(b.getData().tobytes())
                        packets += 1
                        ts = b.getTimestampDevice()
                        ts_first = ts_first if ts_first is not None else ts
                        ts_last = ts

                f = ql.tryGet()
                if f is not None:
                    shown += 1
                    frame = f.getCvFrame()
                    now = time.time()
                    if now >= next_detect:
                        next_detect = now + 1.0 / DETECT_HZ
                        hit = faces.largest(frame)
                        if hit:
                            framer.ease_to_face(*hit)
                            curve.append([round(now - t0, 3), *framer.normalized()])
                    x, y, w, h = framer.rect()
                    cropped = np.ascontiguousarray(frame[y:y + h, x:x + w])
                    last_crop = cropped
                    if vcam is not None:
                        vcam.send(cropped)
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            elapsed = time.time() - t0
            if clip is not None:
                drain = time.time()
                while time.time() - drain < 1.5:
                    b = qb.tryGet()
                    if b is not None:
                        clip.write(b.getData().tobytes())
                    time.sleep(0.005)
                clip.close()
            if vcam is not None:
                vcam.close()

    print(f"\n{elapsed:.1f}s, live {shown / elapsed:.2f} fps, {len(curve)} reframe samples")
    if not framer.seen:
        print("warning: no face was ever detected, the crop stayed centred")
    if last_crop is not None:
        cv2.imwrite(base + "-frame.jpg", last_crop)
        print(f"last live crop written to {base}-frame.jpg ({last_crop.shape[1]}x{last_crop.shape[0]})")

    if not record:
        return

    frames = probe_frames(base + ".h264")
    if frames is None:
        print(f"could not read back {base}.h264; leaving it unmuxed")
        return
    # Wall clock is contaminated by the drain, which keeps encoding after the
    # take stops. Device timestamps on the encoded packets are the honest span.
    span = (ts_last - ts_first).total_seconds() if ts_first is not None and ts_last != ts_first else 0.0
    measured = (packets - 1) / span if span > 0 else frames / elapsed
    off = abs(measured - args.fps) / args.fps
    print(f"recorded {frames} frames ({packets} packets), {measured:.3f} fps "
          f"({measured / args.fps * 100:.1f}% of nominal)")
    if frames > packets:
        print(f"the clip runs {frames - packets} frames past the take: the encoder keeps going "
              "during the drain, which is harmless")
    if off > 0.02:
        print(f"delivered rate is {off * 100:.1f}% off nominal: muxing at the measured rate so the")
        print("angles do not slide apart, but expect a conform step in the edit")
    rate = args.fps if off <= 0.02 else measured

    mp4 = base + ".mp4"
    cmd = ["ffmpeg", "-y", "-r", f"{rate:.5f}", "-i", base + ".h264", "-c", "copy", mp4]
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        subprocess.run(cmd, capture_output=True)
        print(f"wrote {mp4}")
    else:
        print("ffmpeg not found, mux by hand:\n  " + " ".join(cmd))

    with open(base + ".json", "w") as f:
        json.dump({
            "clip": os.path.basename(mp4),
            "started_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
            "seconds": round(elapsed, 3),
            "frames": frames,
            "fps_nominal": args.fps,
            "fps_measured": round(measured, 4),
            "mux_rate": round(rate, 5),
            "recorded": f"{REC_W}x{REC_H}",
            "bitrate_kbps": BITRATE_KBPS,
            "look": look,
            "audio": None,
            "sync": "clap on camera, no audio track",
            "reframe_curve": {
                "note": "t seconds from start, then x y w h as fractions of the recorded frame",
                "samples": curve,
            },
        }, f, indent=2)
    print(f"wrote {base}.json")


if __name__ == "__main__":
    main()
