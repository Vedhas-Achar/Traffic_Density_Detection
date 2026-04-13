# Traffic Density Check — YOLOv8 Vehicle Counter
Detect and count vehicles (**car**, **bus**, **truck**) inside a user-defined ROI polygon using YOLOv8.
Supports both **single-image** and **video** pipelines.

---

## Demo Videos

| # | Traffic Level | Link |
|---|---------------|------|
| 1 | 🟡 Medium Traffic | [Output — Medium Traffic Detection](https://youtu.be/rWM24iDUImY) |
| 2 | 🔴 Heavy Traffic | [Output — Heavy Traffic Detection](https://youtu.be/J7Ny3Lo9jFs) |
| 3 | 🟢 Low Traffic | [Output — Low Traffic Detection](https://youtube.com/shorts/n9aImIrHtNs?feature=share) |

---

## Install
```bash
pip install -r requirements.txt
```

---

## 1) Video Pipeline — `traffic_density.py`
Process every frame of a video, annotate it, and save the output alongside an optional JSON report.

### Full usage
```bash
python traffic_density.py \
  --input  input.mp4 \
  --output output.mp4 \
  --weights yolov8n.pt \
  --roi    "x1,y1 x2,y2 x3,y3 x4,y4" \
  --conf   0.4 \
  --iou    0.45 \
  --classes 2 5 7 \
  --json-out out.json
```

### Arguments
| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | ✅ | — | Path to the source video (`.mp4`, `.avi`, etc.) |
| `--output` | ✅ | — | Path for the annotated output video |
| `--weights` | — | `yolov8n.pt` | YOLOv8 model weights file |
| `--roi` | — | interactive | Four polygon vertices as `"x1,y1 x2,y2 x3,y3 x4,y4"`. Omit to select interactively on the first frame |
| `--conf` | — | `0.4` | Confidence threshold (lower = more detections) |
| `--iou` | — | `0.45` | NMS IoU threshold |
| `--classes` | — | `2 5 7` | COCO class IDs to count (car=2, bus=5, truck=7) |
| `--json-out` | — | — | Save a JSON report with per-frame counts and summary stats |

### Interactive ROI Selection
If `--roi` is omitted, the first frame opens in a window:
- **Left-click** — place a point (4 required)
- **r** — reset all points
- **Enter** — confirm selection
- **q / Esc** — cancel

---

## 2) Image Pipeline — `count_cars_roi.py`
Count vehicles in a single aerial highway image.

```bash
python count_cars_roi.py \
  --image highway.jpg \
  --roi "100,150 1200,140 1230,620 90,640" \
  --model yolov8n.pt \
  --conf 0.25 \
  --save out.jpg \
  --json out.json
```

Or use interactive ROI:
```bash
python count_cars_roi.py --image highway.jpg --click-roi
```

---

## Notes
- Counted classes: `car` (2), `bus` (5), `truck` (7) from the COCO dataset.
- The video pipeline draws a semi-transparent ROI overlay, colour-coded bounding boxes, and a real-time HUD with vehicle count, frame number, and processing FPS.
- The JSON report includes per-frame detections and summary statistics (min/max/mean vehicles).
