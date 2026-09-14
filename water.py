"""Automatic water mask for the river view.

SegFormer-B2 trained on ADE20K labels every pixel; its water / sea / river / lake
classes give a pixel-accurate river outline. Raw output bleeds onto the dark
window mullions at pane edges, so the mask is cleaned: drop near-black pixels,
open to cut thin bridges, keep only sizable blobs. Runs on demand (a few
seconds on CPU), not per frame. Manual boxes from roi.json are unioned in.

out/water_mask.png holds the mask at 960x540 (1/4 of 4K) so lookups stay cheap.
`uv run python water.py <image>` writes the mask for a saved frame.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MASK_PATH = os.path.join(HERE, "out", "water_mask.png")
SCALE = 4                      # mask pixel = 4x4 block of the 4K frame
MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
WATER_LABELS = ("water", "sea", "river", "lake")
MIN_BLOB_FRAC = 0.002          # of frame area; smaller blobs are noise (puddles, glints)
_model = None


def _load():
    global _model
    if _model is None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        proc = SegformerImageProcessor.from_pretrained(MODEL)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL).eval()
        ids = [i for i, l in model.config.id2label.items() if l in WATER_LABELS]
        _model = (proc, model, ids)
    return _model


def window_panes(bgr, min_frac=0.02):
    """Bright regions of the frame: the view through the glass, pane by pane.

    Shot from indoors, the frame is mostly dark mullion and reveal, and the scene
    parser reads the whole image as a window: 43% wall, 42% windowpane, 0.4% water
    on this view. Segmenting each lit pane instead removes that confusion entirely.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lit = (cv2.GaussianBlur(g, (21, 21), 0) > 70).astype(np.uint8)
    lit = cv2.morphologyEx(lit, cv2.MORPH_CLOSE, np.ones((60, 60), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lit)
    panes = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if w * h >= min_frac * g.size and w > 200 and h > 200:
            panes.append((x, y, w, h))
    return panes or [(0, 0, bgr.shape[1], bgr.shape[0])]


def _segment_region(bgr, ids, proc, model):
    """Raw water mask for one region, at full region resolution."""
    import torch

    with torch.no_grad():
        logits = model(**proc(images=np.ascontiguousarray(bgr[:, :, ::-1]), return_tensors="pt")).logits
    h, w = max(1, bgr.shape[0] // SCALE), max(1, bgr.shape[1] // SCALE)
    seg = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear").argmax(1)[0].numpy()
    return np.isin(seg, ids)


def segment(bgr):
    """Boolean water mask at 1/SCALE resolution of the input frame."""
    proc, model, ids = _load()
    h, w = bgr.shape[0] // SCALE, bgr.shape[1] // SCALE
    mask = np.zeros((h, w), bool)
    for (x, y, pw, ph) in window_panes(bgr):
        sub = _segment_region(bgr[y:y + ph, x:x + pw], ids, proc, model)
        sy, sx = y // SCALE, x // SCALE
        target = mask[sy:sy + sub.shape[0], sx:sx + sub.shape[1]]
        target |= sub[:target.shape[0], :target.shape[1]]

    small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    # window mullions are near-black; shadowed water is dark but not that dark
    mask &= cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) > 8
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # fill wakes and glints
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))    # cut thin bridges
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB_FRAC * h * w:
            keep[lab == i] = 1
    return keep.astype(bool)


def with_boxes(mask, boxes):
    """Union manual 4K boxes into a 1/SCALE mask."""
    out = mask.copy()
    for x1, y1, x2, y2 in boxes:
        out[y1 // SCALE:y2 // SCALE, x1 // SCALE:x2 // SCALE] = True
    return out


def save(mask):
    os.makedirs(os.path.dirname(MASK_PATH), exist_ok=True)
    cv2.imwrite(MASK_PATH, mask.astype(np.uint8) * 255)


def load():
    m = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def contains(mask, x, y):
    """Is the 4K pixel (x, y) on water?"""
    my, mx = int(y) // SCALE, int(x) // SCALE
    return 0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1] and bool(mask[my, mx])


def bands(mask, pad=24):
    """Bounding boxes (4K coords) of each water blob, padded; these drive the tiles."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])  # plain ints: these go to JSON and roi.tiles
        out.append((max(0, x * SCALE - pad), max(0, y * SCALE - pad),
                    min(mask.shape[1] * SCALE, (x + w) * SCALE + pad), min(mask.shape[0] * SCALE, (y + h) * SCALE + pad)))
    return out


def outline(img, mask, color=(255, 160, 60), thickness=4):
    """Draw the mask boundary onto a 4K frame."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, [c * SCALE for c in cnts], -1, color, thickness)


if __name__ == "__main__":
    frame = cv2.imread(sys.argv[1])
    m = segment(frame)
    save(m)
    vis = frame.copy()
    outline(vis, m)
    out = os.path.join(HERE, "out", "water_mask_vis.jpg")
    cv2.imwrite(out, cv2.resize(vis, (1920, 1080)))
    print(f"water {m.mean() * 100:.1f}% of frame, {len(bands(m))} blobs -> {out}")
