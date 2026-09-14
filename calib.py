"""Auto-calibration: learn where the camera is and which way it looks from AIS.

Every sighting logs the boat's pixel position plus every AIS vessel moving on
the river at that moment (table calib_obs). With no location given, we brute
force the camera model  x_px = cx + f * tan(bearing - heading)  over a grid of
observer positions on Manhattan's west side, headings and focal lengths. Each
candidate model scores how many sightings have an AIS vessel within TOL_PX of
the predicted pixel; the best model wins once it explains enough sightings.

`uv run python calib.py` fits and writes out/geo.json; spotter.py loads it.
"""
import json
import math
import os
import sqlite3

import numpy as np

import config

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "out", "spotter.db")
GEO = os.path.join(HERE, "out", "geo.json")
FRAME_CX = 1920
TOL_PX = 60
MIN_SIGHTINGS = 8          # need this many explained sightings before trusting a fit
MIN_INLIER_FRAC = 0.5

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (ts REAL, tid INT, cx REAL, cy REAL, px_per_s REAL, name TEXT, mmsi INT, crop TEXT);
CREATE TABLE IF NOT EXISTS calib_obs (ts REAL, tid INT, cx REAL, cy REAL, ais_json TEXT);
"""


def bearing(lat0, lon0, lat, lon):
    """Initial bearing in degrees from (lat0, lon0) to (lat, lon); flat-earth is fine at 5 km."""
    dx = (lon - lon0) * math.cos(math.radians(lat0))
    dy = lat - lat0
    return math.degrees(math.atan2(dx, dy)) % 360


def project(geo, lat, lon):
    """Predicted pixel x for a vessel, or None if behind the camera / outside +-80 deg."""
    rel = (bearing(geo["lat"], geo["lon"], lat, lon) - geo["heading"] + 540) % 360 - 180
    if abs(rel) > 80:
        return None
    return FRAME_CX + geo["f"] * math.tan(math.radians(rel))


def match(geo, cx, vessels):
    """Closest moving AIS vessel to pixel x within tolerance, or None."""
    best, best_d = None, TOL_PX
    for v in vessels:
        px = project(geo, v.lat, v.lon)
        if px is not None and abs(px - cx) < best_d:
            best, best_d = v, abs(px - cx)
    return best


def load():
    if os.path.exists(GEO):
        with open(GEO) as f:
            return json.load(f)
    return None


def reset(db):
    """Forget the old location: drop calibration observations and the fitted geometry."""
    db.execute("DELETE FROM calib_obs")
    db.commit()
    if os.path.exists(GEO):
        os.remove(GEO)


NULL_TRIALS = 5         # shuffled-data fits used to measure how many hits chance alone buys
NULL_MARGIN = 4         # real fit must beat the best chance fit by this many sightings

# Where the camera might be and which way it might point (config.toml).
LATS = np.arange(config.CAL_LAT_RANGE[0], config.CAL_LAT_RANGE[1], 0.002)
LONS = np.arange(config.CAL_LON_RANGE[0], config.CAL_LON_RANGE[1], 0.002)
HEADINGS = np.arange(config.CAL_HEADING_RANGE[0], config.CAL_HEADING_RANGE[1], 2.0)
FOCALS = np.concatenate([                      # px at 4K; unknown optics, so search widely
    np.arange(1600, 4001, 200.0),              # bare lens, roughly 50-100 deg of field
    np.arange(5000, 60001, 2500.0),            # behind a telescope or binocular, down to ~4 deg
])


def _search(cxs, vlat, vlon, starts):
    """Best (hits, geo) over the grid for flattened (sighting, candidate) pairs."""
    lats, lons, headings, focals = LATS, LONS, HEADINGS, FOCALS
    best = (0, None)
    for lat in lats:
        for lon in lons:
            dx = (vlon - lon) * math.cos(math.radians(lat))
            b = np.degrees(np.arctan2(dx, vlat - lat)) % 360
            rel = (b[:, None] - headings[None, :] + 540) % 360 - 180            # pairs x headings
            px = FRAME_CX + focals[None, None, :] * np.tan(np.radians(rel))[:, :, None]
            hit = (np.abs(rel)[:, :, None] < 80) & (np.abs(px - cxs[:, None, None]) < TOL_PX)
            # a sighting is explained if ANY of its candidate vessels lands within tolerance
            per_obs = np.maximum.reduceat(hit.astype(np.int8), starts, axis=0)
            counts = per_obs.sum(axis=0)                                         # headings x focals
            i, j = np.unravel_index(np.argmax(counts), counts.shape)
            if counts[i, j] > best[0]:
                best = (int(counts[i, j]), {"lat": round(float(lat), 4), "lon": round(float(lon), 4),
                                            "heading": float(headings[i]), "f": float(focals[j])})
    return best


def f_from_hfov(hfov_deg):
    """Focal length in 4K pixels for a horizontal field of view."""
    return (FRAME_CX) / math.tan(math.radians(hfov_deg) / 2)


def save(geo):
    with open(GEO, "w") as f:
        json.dump(geo, f, indent=1)


def solve_refs(lat, lon, refs, f_prior):
    """Heading (and focal, with 2+ spread refs) from user-clicked reference points.

    refs: [{"lat", "lon", "x"}] where x is the 4K pixel column the user clicked on
    the real object. One ref: heading only, focal kept. Two or more: least squares
    over heading and focal.
    """
    bears = np.array([bearing(lat, lon, r["lat"], r["lon"]) for r in refs])
    xs = np.array([r["x"] for r in refs], float)

    def heading_for(f):
        # each ref implies a heading; circular mean handles the 359/0 wrap
        hs = np.radians(bears - np.degrees(np.arctan((xs - FRAME_CX) / f)))
        return math.degrees(math.atan2(np.sin(hs).mean(), np.cos(hs).mean())) % 360

    if len(refs) == 1 or np.ptp(xs) < 600:  # refs too close together to pin the lens
        return heading_for(f_prior), f_prior, None
    best = None
    for f in np.arange(1200, 60001, 25.0):
        hd = heading_for(f)
        rel = (bears - hd + 540) % 360 - 180
        err = np.sqrt(np.mean((FRAME_CX + f * np.tan(np.radians(rel)) - xs) ** 2))
        if best is None or err < best[2]:
            best = (hd, float(f), float(err))
    return best


def _load_obs(db_path):
    db = sqlite3.connect(db_path)
    obs = [(cx, json.loads(a)) for cx, a in db.execute("SELECT cx, ais_json FROM calib_obs")]
    return [(cx, a) for cx, a in obs if a]


def fit_heading(lat, lon, prior_heading, db_path=DB, quiet=False, span=40):
    """Known camera position: solve only heading (within +-span of the prior) and focal.

    Two free parameters instead of four, so a handful of sightings is enough.
    Same chance test as the full fit. Returns refined geo or None.
    """
    say = (lambda *a: None) if quiet else print
    obs = _load_obs(db_path)
    if len(obs) < 5:
        say(f"only {len(obs)} usable sightings; need 5 with a known position")
        return None
    counts_per = [len(a) for _, a in obs]
    vlat = np.array([v[2] for _, a in obs for v in a])
    vlon = np.array([v[3] for _, a in obs for v in a])
    starts = np.cumsum([0] + counts_per[:-1])
    obs_cx = np.array([cx for cx, _ in obs])
    headings = np.arange(prior_heading - span, prior_heading + span + 0.01, 0.5) % 360
    dx = (vlon - lon) * math.cos(math.radians(lat))
    b = np.degrees(np.arctan2(dx, vlat - lat)) % 360

    def score(cxs):
        rel = (b[:, None] - headings[None, :] + 540) % 360 - 180
        px = FRAME_CX + FOCALS[None, None, :] * np.tan(np.radians(rel))[:, :, None]
        hit = (np.abs(rel)[:, :, None] < 80) & (np.abs(px - cxs[:, None, None]) < TOL_PX)
        counts = np.maximum.reduceat(hit.astype(np.int8), starts, axis=0).sum(axis=0)
        i, j = np.unravel_index(np.argmax(counts), counts.shape)
        return int(counts[i, j]), float(headings[i]), float(FOCALS[j])

    hits, hd, f = score(np.repeat(obs_cx, counts_per))
    rng = np.random.default_rng(0)
    null_best = max(score(np.repeat(rng.permutation(obs_cx), counts_per))[0] for _ in range(NULL_TRIALS))
    say(f"heading fit {hd} f {f}: {hits}/{len(obs)} sightings (chance alone: {null_best})")
    if hits >= 5 and hits >= null_best + 3 and hits / len(obs) >= MIN_INLIER_FRAC:
        return {"lat": lat, "lon": lon, "heading": hd, "f": f, "inliers": hits, "n": len(obs),
                "null_hits": null_best, "source": "gps+fit"}
    return None


def fit(db_path=DB, quiet=False):
    say = (lambda *a: None) if quiet else print
    db = sqlite3.connect(db_path)
    obs = [(cx, json.loads(a)) for cx, a in db.execute("SELECT cx, ais_json FROM calib_obs")]
    obs = [(cx, a) for cx, a in obs if a]
    if len(obs) < MIN_SIGHTINGS:
        say(f"only {len(obs)} usable sightings; need {MIN_SIGHTINGS}")
        return None
    # Flatten every (sighting, candidate vessel) pair; starts[i] = first pair of sighting i.
    counts_per = [len(a) for _, a in obs]
    vlat = np.array([v[2] for _, a in obs for v in a])
    vlon = np.array([v[3] for _, a in obs for v in a])
    starts = np.cumsum([0] + counts_per[:-1])
    obs_cx = np.array([cx for cx, _ in obs])
    hits, geo = _search(np.repeat(obs_cx, counts_per), vlat, vlon, starts)

    # Null: with many AIS candidates per sighting and 4 free parameters, junk can
    # "explain" most sightings by chance. Shuffle which pixel goes with which AIS
    # snapshot; whatever that scores is luck, and the real fit must clearly beat it.
    rng = np.random.default_rng(0)
    null_best = max(_search(np.repeat(rng.permutation(obs_cx), counts_per), vlat, vlon, starts)[0]
                    for _ in range(NULL_TRIALS))
    say(f"best fit explains {hits}/{len(obs)} sightings (chance alone: {null_best}): {geo}")
    on_edge = geo and (geo["heading"] in (HEADINGS[0], HEADINGS[-1]) or geo["f"] in (FOCALS[0], FOCALS[-1])
                       or geo["lat"] in (round(float(LATS[0]), 4), round(float(LATS[-1]), 4))
                       or geo["lon"] in (round(float(LONS[0]), 4), round(float(LONS[-1]), 4)))
    if on_edge:
        say("rejected: best fit sits on the edge of the search grid")
        return None
    if hits < null_best + NULL_MARGIN:
        say("rejected: not better than chance; keep collecting")
        return None
    if hits >= MIN_SIGHTINGS and hits / len(obs) >= MIN_INLIER_FRAC:
        geo["null_hits"] = null_best
        geo["inliers"], geo["n"] = hits, len(obs)
        with open(GEO, "w") as f:
            json.dump(geo, f, indent=1)
        say("wrote", GEO)
        return geo
    say("not confident yet; keep collecting")
    return None


if __name__ == "__main__":
    fit()
