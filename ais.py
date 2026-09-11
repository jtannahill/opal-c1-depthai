"""Live AIS vessel tracker for the Hudson River view, fed by aisstream.io.

Keeps the latest position and static data per MMSI in memory and in SQLite
(out/ais.db) so the spotter can ask "what ships are near the river right now?".
Run standalone to watch the feed: `uv run python ais.py`.
"""
import asyncio
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field

import websockets

import config

URL = "wss://stream.aisstream.io/v0/stream"
BBOX = config.AIS_BBOX
DB_PATH = os.path.join(os.path.dirname(__file__), "out", "ais.db")


def load_key():
    key = os.environ.get("AISSTREAM_API_KEY")
    env = os.path.join(os.path.dirname(__file__), ".env")
    if not key and os.path.exists(env):
        for line in open(env):
            if line.startswith("AISSTREAM_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("AISSTREAM_API_KEY missing (env or ~/opal-lab/.env)")
    return key


@dataclass
class Vessel:
    mmsi: int
    name: str = ""
    ship_type: int = 0
    length: int = 0
    lat: float = 0.0
    lon: float = 0.0
    sog: float = 0.0
    cog: float = 0.0
    seen: float = field(default_factory=time.time)


class Tracker:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS pos (ts REAL, mmsi INT, name TEXT, lat REAL, lon REAL, sog REAL, cog REAL)"
        )
        self.vessels: dict[int, Vessel] = {}

    def handle(self, msg: dict):
        meta, kind = msg.get("MetaData", {}), msg.get("MessageType")
        mmsi = meta.get("MMSI")
        if not mmsi:
            return
        v = self.vessels.setdefault(mmsi, Vessel(mmsi))
        if meta.get("ShipName"):
            v.name = meta["ShipName"].strip()
        body = msg.get("Message", {}).get(kind, {})
        if kind == "PositionReport" and body.get("Valid", True):
            v.lat, v.lon = body["Latitude"], body["Longitude"]
            sog, cog = body.get("Sog", 0.0), body.get("Cog", 0.0)
            # AIS sentinels: 102.3 kn = speed unavailable, 360 deg = course unavailable
            v.sog = 0.0 if sog >= 102.2 else sog
            v.cog = -1.0 if cog >= 360 else cog
            v.seen = time.time()
            self.db.execute(
                "INSERT INTO pos VALUES (?,?,?,?,?,?,?)", (v.seen, mmsi, v.name, v.lat, v.lon, v.sog, v.cog)
            )
            self.db.commit()
        elif kind == "ShipStaticData":
            v.ship_type = body.get("Type", 0)
            dim = body.get("Dimension", {})
            v.length = dim.get("A", 0) + dim.get("B", 0)

    def recent(self, max_age=600, moving_only=False):
        now = time.time()
        return [
            v for v in self.vessels.values()
            if now - v.seen < max_age and v.lat and (not moving_only or v.sog > 0.5)
        ]

    async def run(self, key: str):
        sub = {
            "APIKey": key,
            "BoundingBoxes": [BBOX],
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        }
        while True:
            try:
                async with websockets.connect(URL) as ws:
                    await ws.send(json.dumps(sub))
                    async for raw in ws:
                        self.handle(json.loads(raw))
            except Exception as e:  # feed drops are routine; back off and reconnect
                print("ais: reconnect after", type(e).__name__, e)
                await asyncio.sleep(5)


async def _watch():
    t = Tracker()
    task = asyncio.create_task(t.run(load_key()))
    while True:
        await asyncio.sleep(15)
        for v in sorted(t.recent(), key=lambda v: -v.sog):
            print(f"{v.mmsi} {v.name or '?':24} {v.sog:4.1f}kn cog {v.cog:5.1f} len {v.length}m @ {v.lat:.4f},{v.lon:.4f}")
        print("--", len(t.recent()), "vessels")
    await task


if __name__ == "__main__":
    asyncio.run(_watch())
