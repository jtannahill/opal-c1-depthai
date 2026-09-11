"""River region-of-interest config and tiling.

The water band is stored in out/roi.json as a list of rectangles in 4K sensor
coordinates (after the 180 deg rotation). Each rectangle is cut into tiles
close to the detector's native input size so small boats keep their pixels.
`uv run python roi.py` saves a gridded frame to out/roi_grid.jpg for picking
the band, and `uv run python roi.py x1 y1 x2 y2 [...]` writes the config.
"""
import json
import math
import os
import sys

PATH = os.path.join(os.path.dirname(__file__), "out", "roi.json")
FRAME_W, FRAME_H = 3840, 2160


def load():
    with open(PATH) as f:
        return [tuple(r) for r in json.load(f)["bands"]]


def save(bands):
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w") as f:
        json.dump({"bands": [list(b) for b in bands]}, f, indent=1)


def tiles(bands, tw=768, th=576, overlap=0.15):
    """Split bands into overlapping tiles (x, y, w, h) about tw x th pixels.

    768x576 downsamples 1.5x into a 512x384 model input: a compromise between
    tile count (each tile is one inference on the camera) and small-boat pixels.
    """
    out = []
    for x1, y1, x2, y2 in bands:
        w, h = x2 - x1, y2 - y1
        # ceil so a band slightly wider than one tile still gets full coverage
        nx = max(1, math.ceil((w - tw * overlap) / (tw * (1 - overlap))))
        ny = max(1, math.ceil((h - th * overlap) / (th * (1 - overlap))))
        cw, ch = min(w, tw), min(h, th)
        for iy in range(ny):
            for ix in range(nx):
                x = x1 + (0 if nx == 1 else round(ix * (w - cw) / (nx - 1)))
                y = y1 + (0 if ny == 1 else round(iy * (h - ch) / (ny - 1)))
                out.append((x, y, cw, ch))
    return out


def grab_grid():
    import cv2
    import depthai as dai

    with dai.Pipeline() as p:
        cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
        cam.initialControl.setManualExposure(1000, 100)
        cam.initialControl.setManualFocus(120)
        q = cam.requestOutput((FRAME_W, FRAME_H), dai.ImgFrame.Type.NV12, fps=5).createOutputQueue()
        p.start()
        for _ in range(15):
            f = q.get()
    img = cv2.resize(f.getCvFrame(), (1920, 1080))
    for x in range(0, 1920, 80):
        cv2.line(img, (x, 0), (x, 1080), (0, 0, 255), 1)
        if x % 160 == 0:
            cv2.putText(img, str(x * 2), (x + 3, 30), 0, 0.7, (0, 0, 255), 2)
    for y in range(0, 1080, 60):
        cv2.line(img, (0, y), (1920, y), (0, 0, 255), 1)
        if y % 120 == 0:
            cv2.putText(img, str(y * 2), (3, y + 25), 0, 0.7, (0, 0, 255), 2)
    out = os.path.join(os.path.dirname(PATH), "roi_grid.jpg")
    cv2.imwrite(out, img)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        grab_grid()
    else:
        vals = [int(v) for v in sys.argv[1:]]
        bands = [tuple(vals[i:i + 4]) for i in range(0, len(vals), 4)]
        save(bands)
        print("bands:", bands, "->", len(tiles(bands)), "tiles")
