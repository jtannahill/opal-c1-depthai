# Opal Lab

Repurposing an **Opal C1 webcam** as a programmable smart camera with the
open-source **DepthAI** SDK, instead of the vendor app.

The C1 turns out to be Luxonis-class hardware: a Movidius Myriad X (RVC2) vision
processor behind a 48 MP autofocus sensor. The open DepthAI stack talks to it
directly, so you can run your own neural networks on the camera, drive the
sensor manually, and build whatever you want on top.

What this repo runs on one camera pointed out a window at a river:

- **Ship spotter** - YOLOv6-nano runs on the camera over the water only, tracks
  boats, ignores anything that does not move, and names vessels from a live AIS
  feed once the camera's position and heading are known.
- **Document scanner** - a 4K still on demand, page detection, perspective
  flattening, then text recognition through Apple Vision.
- **Webcam** - the 1080p stream published as a virtual camera for calls, with
  exposure metered on a face when one appears.
- **Dashboard** - a local page with the live view, water outline, bearing
  overlay, AIS labels, sightings, calibration and scanning.

Nothing is flashed to the camera. DepthAI loads firmware into RAM; unplug it,
plug it back in, open the vendor app and the camera is stock again.

## Why bother: the C1 is on its way off the shelf

On 1 June 2026 the company (now Opal Electronics Inc.) wrote: "Our cameras are
old and will soon leave the shelf. But they will not leave you, we will service
and support them for many years." (<https://op.al>)

So the C1 is being retired from sale rather than abandoned, and Opal is still
trading. Still, a camera whose model is winding down is exactly the hardware
worth owning outright. The Myriad X inside it is a general-purpose vision
processor with an open SDK, so the camera stays programmable and useful for as
long as the hardware lasts, whatever happens to the vendor app, the product
line or the company behind it. Everything here runs locally: no cloud, no
account, no dependency on anyone's servers.

## What the Opal C1 actually is

Findings from taking one apart in software (September 2026, firmware as shipped
with Opal Composer 2.0). Useful if you are pointing DepthAI at one yourself.

| Property | Value |
| --- | --- |
| USB vendor ID | `0x03E7` (Intel Movidius), product `0xF63D` |
| Platform | Luxonis RVC2 / Myriad X, USB 3 SuperSpeed, bootloader 0.0.15 |
| Sensor | `LCM48`, 48 MP (8000x6000), colour, autofocus |
| Modes via DepthAI | 4K up to 42 fps, 4000x3000 at 30 fps, 5312x6000 at 10 fps |
| Modes as a plain webcam | 720p / 1080p / 1440p / 4K, 30 fps only |
| Stereo, IMU | none - single camera, no motion sensor |
| Factory calibration | EEPROM empty (`EEPROM_INVALID_DATA`) |
| Chip temperature | around 35 C idle, 50 C under load |

Opal Composer ships `libdepthai-core`, the DepthAI bootloader binaries and the
camera firmware blobs; that is what gave the game away.

## Beyond the Composer defaults

The vendor app is a webcam app: it decides how the camera is used, and most of
the hardware never comes into play. Measured on the same camera, same session.

| | Opal Composer / plain webcam | Direct with DepthAI |
| --- | --- | --- |
| Resolution and rate | 720p / 1080p / 1440p / 4K, 30 fps only | 4K up to 42 fps, 12 MP (4000x3000) at 30 fps, about 32 MP (5312x6000) at 10 fps |
| Sensor used | 8 MP of a 48 MP sensor | any mode the sensor offers, including near-full readout for stills |
| Exposure | automatic, no exposure time or ISO of your own | manual exposure in microseconds and ISO, or auto metered on a region you choose |
| Focus | autofocus, hunts on a window view | manual lens position 0-255, or autofocus you trigger and region you set |
| Framing | one fixed output | several outputs at once, plus native-resolution crops taken on the camera |
| On-camera compute | idle apart from image processing; Composer's own face and background models run as CoreML on your Mac | your neural networks run on the camera's 4 TOPS chip, plus small scripts on the device |
| Control | sliders in the app | every setting, at runtime, from code |

Concretely, that is what makes this project possible: detection on the camera
instead of the laptop, exposure fixed for a bright window from a dark room,
focus locked at infinity, several crops of the river cut out at full resolution
and fed to a network, all at the same time as a webcam feed.

### Tuning the picture yourself

The ISP knobs are all reachable at runtime: exposure time, ISO, contrast,
sharpness, luma and chroma denoise, saturation, brightness, white balance mode,
anti-banding, and a lens position for focus. Sweeping them against measurements
on the water beat guessing; on this window view, per 4K frame:

| Setting | Effect measured on the water |
| --- | --- |
| Exposure 1000 -> 1500 us | mean 42 -> 66, crushed shadows 9.4% -> 2.1%, no clipping |
| Contrast 0 -> +4 | contrast 44 -> 57, detail 712 -> 1280, clipped highlights 3.5% |
| Contrast +8 | contrast 61 but 8% clipped: the bright shore starts burning out |
| Sharpness 0 -> 1 | detail 412 -> 1283; 3 and 4 mostly add noise |
| Luma denoise 4 | detail collapses (3300 -> 566); keep it at 0-2 |
| setHdr | no effect on this sensor, on or off |
| White balance modes | negligible here; daylight matches auto |

Shipped default (`IMAGE_DEFAULTS`, adjustable live from the dashboard and saved
to `out/cam.json`): exposure 1500 us, ISO 100, contrast +4, sharpness 1, luma
and chroma denoise 1. Against the camera's own auto settings, that is the
difference between a white rectangle where the river should be and a picture
with the shoreline, piers and wakes all readable.

Two things the vendor app still does better: it ships a tuned image pipeline
(nicer colour and noise handling for faces in a room), and it works out of the
box. Taking the camera over means doing that tuning yourself.

### Gotchas worth knowing

1. **The sensor is mounted upside down.** In DepthAI mode every frame arrives
   rotated 180 degrees; the vendor firmware corrects it silently. Use
   `setImageOrientation(ROTATE_180_DEG)`.
2. **Mixed frame rates stall the whole pipeline.** Camera outputs at different
   `fps` values (4K at 5 plus 1080p at 30, or anything at 1 fps) drop every
   stream to about 0.1 fps. Request every output at the same rate and throttle
   downstream instead.
3. **Blocking node inputs stall it too.** Any `ImageManip` or `NeuralNetwork`
   input left blocking back-pressures the camera and freezes all branches. Set
   `setBlocking(False)` and `setMaxSize(1)` on every branch.
4. **The UVC node does not work on macOS.** The device enumerates as a webcam
   but never delivers frames, and its unread input stalls the pipeline. Publish
   frames from the host instead (this repo uses `pyvirtualcam`).
5. **The ISP budget is real.** About 600 MP/s. 4K30 plus 1080p30 is fine; adding
   a 12 MP stream is not. Take stills from the 4K stream.
6. **Under load the camera does not hit 30 fps.** With five detection crops it
   settles around 12.5 fps at 4K and 14 fps at 1080p.
7. **Autofocus hunts on a window view.** Fix the lens: on this mount position
   120 was a sharp peak (sharpness 1642 versus 678 at 124 and 320 at 116).
8. **Auto-exposure cannot handle a bright window from a dark room.** Manual
   exposure (1000 us, ISO 100 here) is the difference between a blown-out white
   rectangle and a usable river.
9. **The vendor app fights you for the camera.** Opal Composer relaunches itself
   through a login item; this service quits it on startup and leaves the login
   item alone.

## How it works

```
Opal C1 (Myriad X)                         Mac
  4K30 -> ImageManip crops -> YOLOv6-nano ---> boat boxes -> tracker -> sightings
       -> Script (1 frame in 2) ------------> 4K stills -> MJPEG dashboard, crops, scans
  1080p30 ------------------------------------> virtual camera, face detector
```

- **Water, not rectangles.** SegFormer (ADE20K) segments water in a still on the
  Mac; the mask decides which crops the camera runs and whether a detection
  counts. Runs on demand, not per frame.
- **Movement, not confidence.** Piers, moored hulls and the far shore light up a
  boat detector constantly. A track is only a sighting once its centre has moved
  far enough, and absurd speeds or oversized boxes are rejected.
- **Naming by geometry.** Enter the camera's GPS and rough facing, or click
  "Align" on a ship in the AIS list and then on that ship in the image; two
  ships far apart also solve the lens field of view. A blind fit from sightings
  alone works too, guarded by a shuffled-data null test so noise cannot produce
  a confident wrong answer.

## Setup

Requires a Mac (Apple Vision is used for text recognition), Python 3.12+ and
[uv](https://docs.astral.sh/uv/).

```sh
git clone <this repo> && cd opal-lab
uv sync
cp config.example.toml config.toml   # edit for your view
echo "AISSTREAM_API_KEY=..." > .env  # free key from https://aisstream.io
uv run python spotter.py             # dashboard at http://127.0.0.1:8810
```

First run, in the dashboard:

1. **Auto-detect water** - outlines the water the detector should watch.
2. **Camera position** - paste GPS, enter the facing direction, then align on
   two ships to pin the heading and lens.
3. **Recalibrate** any time the camera moves: it re-sweeps focus, re-detects the
   water and resets the bearing fit.

Without a camera position, sightings are still logged and photographed; they
just stay unnamed.

## Files

| File | Purpose |
| --- | --- |
| `spotter.py` | The service: pipeline, tracking, HTTP API, dashboard |
| `dashboard.html` | Live view, zoom and crop, calibration, scanner, AIS list |
| `water.py` | SegFormer water mask, cleanup, water areas |
| `calib.py` | Bearing fits: blind, GPS-anchored, and click-to-align |
| `tracker.py` | Boat tracking with static-object suppression |
| `ais.py` | aisstream.io client and vessel store |
| `ocr.py` | Page detection, flattening, Apple Vision text recognition |
| `roi.py` | Manual water boxes and tiling |
| `cam_in_use.py` | CoreMediaIO check for whether an app is using a camera |
| `composer.py` | Quits the vendor app so the camera is free |

## Returning the camera to stock

Stop the service, unplug the camera, plug it back in, open Opal Composer. The
DepthAI firmware only ever lives in RAM; flash memory is never written.
