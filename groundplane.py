"""Turn the water in the frame into a map, using AIS as the survey.

Once the camera's position, heading and focal length are known (calib.py), a
boat's column already gives its bearing. This adds the missing half: how far
away it is, from how far down the frame it sits.

For a level camera at height h above the water, a point at bearing offset rel
and range R projects to

    x = cx + f * tan(rel)
    y = y_horizon + f * h * sec(rel) / R

so y is linear in h once y_horizon is known, and both fall out of least squares
over sightings whose range we know from AIS. Inverting it turns any water pixel
into a real latitude and longitude, which is what makes a top-down map possible.

Assumes flat water (fine), no roll, and square pixels. Tide changes h by a metre
or two out of tens of metres, which is lost in the noise at this range.
"""
import json
import math
import os

import numpy as np

import calib

HERE = os.path.dirname(os.path.abspath(__file__))
PLANE = os.path.join(HERE, "out", "plane.json")
FRAME_CX, FRAME_CY = 1920, 1080
MIN_OBS = 8
MAX_RESID_PX = 40          # a fit worse than this is not worth trusting
EARTH_M_PER_DEG = 111_320


def range_m(lat0, lon0, lat, lon):
    """Flat-earth range in metres; exact enough over a few kilometres."""
    dy = (lat - lat0) * EARTH_M_PER_DEG
    dx = (lon - lon0) * EARTH_M_PER_DEG * math.cos(math.radians(lat0))
    return math.hypot(dx, dy)


def load():
    if os.path.exists(PLANE):
        with open(PLANE) as f:
            return json.load(f)
    return None


def save(plane):
    with open(PLANE, "w") as f:
        json.dump(plane, f, indent=1)


def fit(geo, obs, min_obs=MIN_OBS):
    """obs: [(x_px, y_px, lat, lon)] of boats whose AIS position we trust.

    Returns {"y_horizon", "height_m", "rms_px", "n"} or None.
    """
    if len(obs) < min_obs:
        return None
    rel = np.array([math.radians((calib.bearing(geo["lat"], geo["lon"], lat, lon) - geo["heading"] + 540) % 360 - 180)
                    for _, _, lat, lon in obs])
    rng = np.array([range_m(geo["lat"], geo["lon"], lat, lon) for _, _, lat, lon in obs])
    y = np.array([o[1] for o in obs], float)
    keep = (rng > 50) & (np.abs(rel) < math.radians(80))
    if keep.sum() < min_obs:
        return None
    rel, rng, y = rel[keep], rng[keep], y[keep]

    # y = y_horizon + h * u, with u = f * sec(rel) / R
    u = geo["f"] / np.cos(rel) / rng
    for _ in range(3):  # a couple of rounds of trimming; AIS antennas are not hull centres
        A = np.column_stack([np.ones_like(u), u])
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = np.abs(A @ sol - y)
        if len(u) <= min_obs or resid.max() < MAX_RESID_PX:
            break
        keep = resid < np.percentile(resid, 80)
        u, y, rel, rng = u[keep], y[keep], rel[keep], rng[keep]
    y_horizon, height = float(sol[0]), float(sol[1])
    rms = float(np.sqrt(np.mean((np.column_stack([np.ones_like(u), u]) @ sol - y) ** 2)))
    if not (0.5 < height < 500) or rms > MAX_RESID_PX:
        return None
    return {"y_horizon": y_horizon, "height_m": height, "rms_px": round(rms, 1), "n": int(len(u))}


def pixel_to_world(geo, plane, x, y):
    """Water pixel -> (lat, lon, range_m, bearing_deg), or None above the horizon."""
    rel = math.atan((x - FRAME_CX) / geo["f"])
    dy = y - plane["y_horizon"]
    if dy <= 1:
        return None
    rng = geo["f"] * plane["height_m"] / math.cos(rel) / dy
    brg = (geo["heading"] + math.degrees(rel)) % 360
    lat = geo["lat"] + (rng * math.cos(math.radians(brg))) / EARTH_M_PER_DEG
    lon = geo["lon"] + (rng * math.sin(math.radians(brg))) / (EARTH_M_PER_DEG * math.cos(math.radians(geo["lat"])))
    return {"lat": lat, "lon": lon, "range_m": rng, "bearing": brg}


def world_to_pixel(geo, plane, lat, lon):
    """(lat, lon) -> (x, y) in the 4K frame, or None if behind the camera."""
    rel_deg = (calib.bearing(geo["lat"], geo["lon"], lat, lon) - geo["heading"] + 540) % 360 - 180
    if abs(rel_deg) > 80:
        return None
    rel = math.radians(rel_deg)
    rng = max(1.0, range_m(geo["lat"], geo["lon"], lat, lon))
    return (FRAME_CX + geo["f"] * math.tan(rel),
            plane["y_horizon"] + geo["f"] * plane["height_m"] / math.cos(rel) / rng)


def speed_knots(geo, plane, track):
    """Track speed in knots from its pixel path, once the plane is known."""
    if len(track.path) < 2:
        return None
    (t0, x0, y0), (t1, x1, y1) = track.path[0], track.path[-1]
    a = pixel_to_world(geo, plane, x0, y0)
    b = pixel_to_world(geo, plane, x1, y1)
    if not a or not b or t1 <= t0:
        return None
    d = range_m(a["lat"], a["lon"], b["lat"], b["lon"])
    return d / (t1 - t0) * 1.94384


def horizon_from_mask(mask, scale):
    """Topmost water row in the mask, in frame pixels: a stand-in for the horizon.

    The far shore sits slightly below the true horizon, so ranges estimated from
    this run long at distance. Good enough to put reticles on the water.
    """
    import numpy as np

    rows = np.flatnonzero(mask.any(axis=1))
    return None if rows.size == 0 else float(rows[0] * scale)


def estimate(y_horizon, height_m):
    """A plane from an assumed camera height instead of fitted from AIS."""
    return {"y_horizon": float(y_horizon), "height_m": float(height_m),
            "rms_px": None, "n": 0, "estimated": True}


def range_from_size(f, box_w_px, length_m):
    """Stadiametric range: a vessel of known length subtending box_w_px pixels.

    Assumes the hull is roughly broadside; bow-on it reads far too close.
    """
    if box_w_px <= 1 or length_m <= 0:
        return None
    return f * length_m / box_w_px


if __name__ == "__main__":
    # Self-test: invent a camera, generate boats, check the fit recovers it.
    import random

    geo = {"lat": 40.7614, "lon": -73.9975, "heading": 283.0, "f": 2600.0}
    truth = {"y_horizon": 1180.0, "height_m": 62.0}
    random.seed(3)
    obs = []
    for _ in range(25):
        brg = geo["heading"] + random.uniform(-30, 30)
        rng = random.uniform(300, 2500)
        lat = geo["lat"] + rng * math.cos(math.radians(brg)) / EARTH_M_PER_DEG
        lon = geo["lon"] + rng * math.sin(math.radians(brg)) / (EARTH_M_PER_DEG * math.cos(math.radians(geo["lat"])))
        xy = world_to_pixel(geo, truth | {"y_horizon": truth["y_horizon"]}, lat, lon)
        obs.append((xy[0] + random.gauss(0, 6), xy[1] + random.gauss(0, 6), lat, lon))
    got = fit(geo, obs)
    print("truth :", truth)
    print("fitted:", got)
    if got:
        p = pixel_to_world(geo, got, 1500, 1350)
        print(f"pixel (1500,1350) -> {p['range_m']:.0f} m at {p['bearing']:.1f} deg true")
