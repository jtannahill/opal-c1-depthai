"""Document scanner post-processing: find the page, flatten it, OCR with Apple Vision."""
import sys, cv2, numpy as np, Quartz, Vision
from Foundation import NSURL

def flatten(img):
    """Perspective-correct the largest 4-corner contour (the page); returns img unchanged if none found."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(g, 50, 150), None)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.15 * img.shape[0] * img.shape[1]:
            pts = approx.reshape(4, 2).astype("float32")
            s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
            tl, br, tr, bl = pts[s.argmin()], pts[s.argmax()], pts[d.argmin()], pts[d.argmax()]
            w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
            return cv2.warpPerspective(img, M, (w, h)), True
    return img, False

def ocr(path):
    """Return recognized text lines (accurate mode, language correction on)."""
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate); req.setUsesLanguageCorrection_(True)
    ok, err = handler.performRequests_error_([req], None)
    if not ok: raise RuntimeError(err)
    return [o.topCandidates_(1)[0].string() for o in (req.results() or [])]

if __name__ == "__main__":
    src = sys.argv[1]; img = cv2.imread(src)
    flat, found = flatten(img); out = src.rsplit(".", 1)[0] + "_flat.jpg"; cv2.imwrite(out, flat)
    print("page found:", found, "->", out)
    for line in ocr(out): print("  ", line)
