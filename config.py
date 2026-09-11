"""Site configuration: which water this camera watches, and where it can be.

Everything location-specific lives in config.toml (gitignored). Copy
config.example.toml to config.toml and edit it for your window.
"""
import os
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "config.toml")

if not os.path.exists(PATH):
    raise SystemExit("config.toml missing: copy config.example.toml to config.toml and edit it for your view")

with open(PATH, "rb") as f:
    _cfg = tomllib.load(f)

_ais = _cfg.get("ais", {})
# Feed subscription box [[lat1, lon1], [lat2, lon2]]: the AIS stream sends everything inside it.
AIS_BBOX = [list(p) for p in _ais["bbox"]]
# Water actually visible from the window [lat_min, lat_max, lon_min, lon_max]: candidates
# for naming and for the "AIS in view" list come from here.
VIEW_LAT_MIN, VIEW_LAT_MAX, VIEW_LON_MIN, VIEW_LON_MAX = _ais["view_region"]

_cal = _cfg.get("calibration", {})
# Blind-fit search space for the camera itself (used when you do not enter GPS).
CAL_LAT_RANGE = _cal.get("lat_range", [VIEW_LAT_MIN, VIEW_LAT_MAX])
CAL_LON_RANGE = _cal.get("lon_range", [VIEW_LON_MIN, VIEW_LON_MAX])
CAL_HEADING_RANGE = _cal.get("heading_range", [0, 360])

PORT = _cfg.get("service", {}).get("port", 8810)


def in_view_region(lat, lon):
    return VIEW_LAT_MIN <= lat <= VIEW_LAT_MAX and VIEW_LON_MIN <= lon <= VIEW_LON_MAX
