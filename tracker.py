"""Host-side boat tracker with static-object suppression.

YOLO on the river tiles fires constantly on things that never move: the docked
cruise ship, pier sheds, the far shore. A track only counts as a vessel
sighting once its center has travelled far enough, so a moored hull or a pier
stays silent however confident the detector is.
"""
import itertools
import time
from dataclasses import dataclass, field

BOAT_LIKE = {"boat"}          # COCO classes treated as vessels
MOVE_PX = 25                  # 4K pixels of center travel before a track is "moving"
MAX_PX_S = 80                 # boats cross at <20 px/s on the bare lens; faster tracks are
                              # occluders, IoU jumps, or shake (tighten this behind a telescope)
MAX_BOX_W = 500               # 4K px; anything wider is a hand/person in front of the lens
MAX_GAP_S = 8.0               # drop tracks unseen this long
IOU_MATCH = 0.2


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


@dataclass
class Track:
    tid: int
    box: tuple
    conf: float
    first: float = field(default_factory=time.time)
    last: float = field(default_factory=time.time)
    path: list = field(default_factory=list)   # (t, cx, cy)
    reported: bool = False

    @property
    def travel(self):
        if len(self.path) < 2:
            return 0.0
        (_, x0, y0), (_, x1, y1) = self.path[0], self.path[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    @property
    def moving(self):
        return self.travel >= MOVE_PX and self.px_per_s <= MAX_PX_S

    @property
    def px_per_s(self):
        dt = self.path[-1][0] - self.path[0][0] if len(self.path) > 1 else 0
        return self.travel / dt if dt else 0.0


class Tracker:
    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, boxes, now=None):
        """boxes: [(x1, y1, x2, y2, conf)] in frame pixels, boat class only.

        Returns tracks that just became confirmed moving vessels (report once).
        """
        now = now or time.time()
        free = dict(self.tracks)
        for *box, conf in sorted(boxes, key=lambda b: -b[4]):
            box = tuple(box)
            if box[2] - box[0] > MAX_BOX_W:
                continue
            best = max(free.values(), key=lambda t: iou(t.box, box), default=None)
            if best and iou(best.box, box) >= IOU_MATCH:
                t = free.pop(best.tid)
            else:
                t = Track(next(self._ids), box, conf, first=now)
                self.tracks[t.tid] = t
            t.box, t.conf, t.last = box, max(t.conf, conf), now
            t.path.append((now, *center(box)))
        for tid in [tid for tid, t in self.tracks.items() if now - t.last > MAX_GAP_S]:
            del self.tracks[tid]
        fresh = [t for t in self.tracks.values() if t.moving and not t.reported]
        for t in fresh:
            t.reported = True
        return fresh

    def active_moving(self):
        return [t for t in self.tracks.values() if t.moving]
