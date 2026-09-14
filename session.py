"""Session mode: run for a fixed spell, then leave something behind.

Built for a camera that is only on now and then. Every run is a cold start, so a
session verifies the stored calibration against live AIS before trusting it,
watches for the requested time, and writes a dated report: what was seen, what
was named, how far away it was, and the frames worth keeping.

The calibration check matters more than it sounds. Between sessions the camera
gets unplugged, carried and knocked, and a heading that is 20 degrees stale
still produces confident-looking bearings and ranges. Each sighting whose AIS
match is unambiguous predicts where that ship should have appeared; the
difference, in pixels, says whether the stored geometry still holds.
"""
import json
import math
import os
import re
import shutil
import time

import calib

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "sessions")
CONFIRM_PX = 70        # median residual below this: the stored heading still holds
MIN_CHECKS = 2         # unambiguous sightings needed before saying anything at all


def parse_duration(text):
    """'20m', '45s', '1h', or a bare number of minutes."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", str(text).lower())
    if not m:
        raise ValueError(f"cannot read a duration from {text!r} (try 20m, 45s, 1h)")
    n, unit = float(m.group(1)), m.group(2) or "m"
    return n * {"s": 1, "m": 60, "h": 3600}[unit]


def verify(geo, obs):
    """Does the stored geometry still predict where ships actually appeared?

    obs: [(cx_px, [[mmsi, name, lat, lon, sog, cog], ...])] as logged per sighting.
    Returns a dict with status: confirmed | drifted | unknown.
    """
    if not geo:
        return {"status": "unknown", "reason": "no camera position stored", "n": 0}
    resid = []
    for cx, cands in obs:
        movers = [v for v in cands if v[4] >= 1.0]
        if len(movers) != 1:            # ambiguous: several ships could be this boat
            continue
        px = calib.project(geo, movers[0][2], movers[0][3])
        if px is not None:
            resid.append(px - cx)
    if len(resid) < MIN_CHECKS:
        return {"status": "unknown", "n": len(resid),
                "reason": f"only {len(resid)} unambiguous sighting(s) this session"}
    resid.sort()
    median = resid[len(resid) // 2]
    out = {"n": len(resid), "median_px": round(median, 1),
           "spread_px": round(resid[-1] - resid[0], 1)}
    if abs(median) <= CONFIRM_PX:
        out["status"] = "confirmed"
        out["reason"] = f"ships landed within {abs(median):.0f} px of where the stored heading put them"
    else:
        # a consistent offset is a rotated camera; convert it back to degrees
        drift = math.degrees(math.atan(median / geo["f"]))
        out["status"] = "drifted"
        out["drift_deg"] = round(drift, 2)
        out["suggested_heading"] = round((geo["heading"] + drift) % 360, 1)
        out["reason"] = (f"ships appeared {abs(median):.0f} px from the prediction, about "
                         f"{abs(drift):.1f} deg of heading error")
    return out


class Session:
    """Collects what happened, then writes it up."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = time.time()
        self.dir = os.path.join(REPORTS, time.strftime("%Y-%m-%d_%H%M", time.localtime(self.started)))
        os.makedirs(self.dir, exist_ok=True)
        self.sightings = []
        self.checks = []          # (cx, candidates) for the calibration check
        self.verdict = None
        self.notes = []

    @property
    def remaining(self):
        return self.seconds - (time.time() - self.started)

    def add(self, row, cx=None, cands=None):
        self.sightings.append(row)
        if cx is not None and cands:
            self.checks.append((cx, cands))

    def save_frame(self, name, jpeg_bytes):
        if not jpeg_bytes:
            return None
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        return path

    def keep(self, src):
        """Copy a sighting crop into the session folder so the report is self-contained."""
        if not src or not os.path.exists(src):
            return None
        dst = os.path.join(self.dir, os.path.basename(src))
        shutil.copy2(src, dst)
        return os.path.basename(dst)

    def write(self, geo=None, plane=None, extra=None):
        named = [s for s in self.sightings if s.get("name")]
        mins = (time.time() - self.started) / 60
        lines = [
            f"# River session, {time.strftime('%d %B %Y, %H:%M', time.localtime(self.started))}",
            "",
            f"Watched for {mins:.0f} minutes. {len(self.sightings)} vessel"
            f"{'' if len(self.sightings) == 1 else 's'} seen, {len(named)} identified.",
            "",
            "## Calibration",
            "",
        ]
        v = self.verdict or {"status": "unknown", "reason": "no check ran"}
        status_text = {
            "confirmed": "Stored geometry still holds.",
            "drifted": "The camera has moved since it was calibrated.",
            "unknown": "Not enough evidence to check the calibration.",
        }[v["status"]]
        reason = v.get("reason", "")
        reason = reason[:1].upper() + reason[1:] if reason else ""
        lines += [f"**{status_text}** {reason}".rstrip(), ""]
        if v.get("status") == "drifted":
            lines += [f"Bearings and ranges in this report are off by roughly "
                      f"{abs(v['drift_deg']):.1f} degrees. Re-align, or set the heading to "
                      f"{v['suggested_heading']} and refit.", ""]
        if geo:
            lines += [f"Camera: {geo['lat']:.5f}, {geo['lon']:.5f}, facing {geo['heading']:.0f} deg true.", ""]
        if plane:
            est = " (estimated from an assumed height)" if plane.get("estimated") else ""
            lines += [f"Ground plane: {plane['height_m']:.0f} m above the water{est}.", ""]

        lines += ["## Vessels", ""]
        if self.sightings:
            lines += ["| Time | Vessel | Bearing | Range | Speed | Frame |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for s in self.sightings:
                lines.append("| {} | {} | {} | {} | {} | {} |".format(
                    time.strftime("%H:%M:%S", time.localtime(s["ts"])),
                    s.get("name") or "unidentified",
                    f"{s['bearing']:.0f} deg" if s.get("bearing") is not None else "-",
                    f"{s['range_km']:.2f} km" if s.get("range_km") is not None else "-",
                    f"{s['knots']:.1f} kn" if s.get("knots") is not None else
                    (f"{s['px_per_s']:.0f} px/s" if s.get("px_per_s") is not None else "-"),
                    f"![]({s['crop']})" if s.get("crop") else "-"))
        else:
            lines.append("Nothing moved on the water during this session.")
        lines.append("")

        if extra:
            lines += ["## Conditions", ""] + [f"- {k}: {v}" for k, v in extra.items()] + [""]
        if self.notes:
            lines += ["## Notes", ""] + [f"- {n}" for n in self.notes] + [""]

        path = os.path.join(self.dir, "report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        with open(os.path.join(self.dir, "session.json"), "w") as f:
            json.dump({"started": self.started, "seconds": self.seconds, "verdict": self.verdict,
                       "geo": geo, "plane": plane, "sightings": self.sightings}, f, indent=1)
        return path
