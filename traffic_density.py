#!/usr/bin/env python3
"""
Traffic Density Check — Video Pipeline
=======================================
Detects and counts vehicles inside a user-defined ROI polygon across every
frame of a video using YOLOv8.

Usage
-----
python traffic_density.py \
    --input  input.mp4 \
    --output output.mp4 \
    --weights yolov8n.pt \
    --roi "x1,y1 x2,y2 x3,y3 x4,y4" \
    --conf 0.4 \
    --iou 0.45 \
    --classes 2 5 7 \
    --json-out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class-id → human-readable name (vehicles only, for reference)
COCO_VEHICLE_NAMES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def parse_roi(roi_text: str) -> np.ndarray:
    """Parse ROI from string ``"x1,y1 x2,y2 x3,y3 x4,y4"`` → Nx2 int32."""
    points: List[Tuple[int, int]] = []
    for part in roi_text.strip().split():
        x_str, y_str = part.split(",")
        points.append((int(float(x_str)), int(float(y_str))))
    if len(points) < 3:
        raise ValueError("ROI polygon needs at least 3 vertices.")
    return np.array(points, dtype=np.int32)


def bbox_center(xyxy: np.ndarray) -> Tuple[float, float]:
    """Return (cx, cy) of an xyxy bounding box."""
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def point_in_polygon(pt: Tuple[float, float], polygon: np.ndarray) -> bool:
    """True when *pt* is inside or on the edge of *polygon*."""
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


# ──────────────────────────────────────────────────────────────────────
# Interactive ROI selection (fallback when --roi is omitted)
# ──────────────────────────────────────────────────────────────────────


def select_roi_interactive(
    frame: np.ndarray, n_points: int = 4
) -> np.ndarray:
    """Open a window and let the user click *n_points* vertices.

    Controls
    --------
    - **Left-click** — add a point
    - **r** — reset all points
    - **Enter** — confirm (only when enough points are placed)
    - **q / Esc** — abort
    """
    points: List[Tuple[int, int]] = []
    win = "Select ROI - click points, Enter=confirm, r=reset, q=quit"

    h, w = frame.shape[:2]
    # Fit the preview on screen for high-resolution videos while preserving coordinates.
    max_w = 1600
    max_h = 900
    scale = min(max_w / w, max_h / h, 1.0)
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    frame_display = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()
    display = frame_display.copy()

    def _redraw() -> None:
        nonlocal display
        display = frame_display.copy()
        for i, (px, py) in enumerate(points, 1):
            dx = int(round(px * scale))
            dy = int(round(py * scale))
            cv2.circle(display, (dx, dy), 6, (0, 255, 255), -1)
            cv2.putText(
                display, str(i), (dx + 8, dy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
        if len(points) >= 2:
            pts_display = np.array(
                [[int(round(px * scale)), int(round(py * scale))] for px, py in points],
                dtype=np.int32,
            )
            cv2.polylines(
                display,
                [pts_display],
                isClosed=(len(points) == n_points),
                color=(0, 255, 255),
                thickness=2,
            )
        cv2.putText(
            display,
            f"Points: {len(points)}/{n_points}  |  Enter=confirm  r=reset  q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    def _on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            ox = int(round(x / scale))
            oy = int(round(y / scale))
            ox = min(max(ox, 0), w - 1)
            oy = min(max(oy, 0), h - 1)
            points.append((ox, oy))
            _redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _on_mouse)
    _redraw()

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(win)
            raise RuntimeError("ROI selection cancelled by user.")
        if key == ord("r"):
            points.clear()
            _redraw()
        if key in (13, 10) and len(points) == n_points:
            cv2.destroyWindow(win)
            return np.array(points, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

# Colour palette for different class IDs
_CLASS_COLOURS: Dict[int, Tuple[int, int, int]] = {
    2: (0, 200, 0),     # car  — green
    3: (200, 100, 0),   # motorcycle — teal
    5: (0, 140, 255),   # bus  — orange
    7: (255, 80, 80),   # truck — blue-ish
}
_DEFAULT_COLOUR = (180, 180, 180)


def draw_overlay(
    frame: np.ndarray,
    roi: np.ndarray,
    boxes: List[np.ndarray],
    class_ids: List[int],
    confidences: List[float],
    names: Dict[int, str],
    count: int,
    frame_idx: int,
    fps_val: float,
    density_text: Optional[str] = None,
) -> np.ndarray:
    """Draw ROI, bounding boxes, count HUD, and FPS on *frame* (in-place)."""
    # Semi-transparent ROI fill
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi], (0, 255, 255))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [roi], True, (0, 255, 255), 2)

    # Bounding boxes
    for box, cid, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        colour = _CLASS_COLOURS.get(cid, _DEFAULT_COLOUR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{names.get(cid, str(cid))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 2, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # HUD — top-left panel
    hud_lines = [
        f"Vehicles in ROI: {count}",
        f"Frame: {frame_idx}",
        f"FPS: {fps_val:.1f}",
    ]
    if density_text is not None:
        hud_lines.append(density_text)
    y0 = 28
    for line in hud_lines:
        cv2.putText(
            frame, line, (10, y0),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA,
        )
        y0 += 28

    return frame


# ──────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────


def process_frame(
    model: YOLO,
    frame: np.ndarray,
    roi: np.ndarray,
    target_classes: set,
    conf_thr: float,
    iou_thr: float,
) -> Tuple[int, List[np.ndarray], List[int], List[float], Dict[int, str]]:
    """Run detection on a single *frame* and filter to ROI.

    Returns
    -------
    count, kept_boxes, kept_class_ids, kept_confs, names_map
    """
    results = model.predict(
        source=frame,
        conf=conf_thr,
        iou=iou_thr,
        classes=list(target_classes),
        verbose=False,
    )[0]

    names: Dict[int, str] = results.names  # type: ignore[assignment]
    boxes_obj = results.boxes

    kept_boxes: List[np.ndarray] = []
    kept_cids: List[int] = []
    kept_confs: List[float] = []

    if boxes_obj is not None and len(boxes_obj) > 0:
        xyxy = boxes_obj.xyxy.cpu().numpy()
        cls = boxes_obj.cls.cpu().numpy().astype(int)
        confs = boxes_obj.conf.cpu().numpy()

        for box, cid, score in zip(xyxy, cls, confs):
            if cid not in target_classes:
                continue
            if point_in_polygon(bbox_center(box), roi):
                kept_boxes.append(box)
                kept_cids.append(int(cid))
                kept_confs.append(float(score))

    return len(kept_boxes), kept_boxes, kept_cids, kept_confs, names


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Traffic Density Check — count vehicles inside an ROI "
                    "across every frame of a video using YOLOv8.",
    )

    # Required
    p.add_argument("--input", required=True, help="Path to the input video file.")
    p.add_argument("--output", required=True, help="Path for the annotated output video.")
    p.add_argument(
        "--weights", default="yolov8n.pt",
        help="YOLOv8 model weights file (default: yolov8n.pt).",
    )

    # ROI
    p.add_argument(
        "--roi", default=None,
        help='ROI polygon as "x1,y1 x2,y2 x3,y3 x4,y4". '
             "Omit to select interactively on the first frame.",
    )

    # Detection tuning
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4).")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45).")
    p.add_argument(
        "--classes", nargs="+", type=int, default=[2, 5, 7],
        help="COCO class IDs to count (default: 2 5 7 = car bus truck).",
    )

    # Output
    p.add_argument(
        "--json-out", default=None,
        help="Path to save a JSON report alongside the video.",
    )
    p.add_argument(
        "--density", action="store_true",
        help="Compute and display vehicle density in ROI and include it in JSON output.",
    )
    p.add_argument(
        "--px-per-meter", type=float, default=None,
        help="Optional pixel-per-meter scale for real-world density (vehicles/m^2).",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    # ── Validate input ─────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"[ERROR] Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {input_path}  ({width}x{height}, {src_fps:.2f} fps, {total_frames} frames)")

    # ── ROI ────────────────────────────────────────────────────────
    if args.roi:
        roi = parse_roi(args.roi)
        print(f"[INFO] ROI from CLI: {roi.tolist()}")
    else:
        print("[INFO] No --roi provided; opening interactive selector on the first frame…")
        ret, first_frame = cap.read()
        if not ret:
            sys.exit("[ERROR] Cannot read the first frame for ROI selection.")
        roi = select_roi_interactive(first_frame, n_points=4)
        print(f"[INFO] ROI selected: {roi.tolist()}")
        # Reset capture back to frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    roi_area_px2 = float(cv2.contourArea(roi.astype(np.float32)))
    if roi_area_px2 <= 0:
        sys.exit("[ERROR] ROI area must be greater than zero.")

    density_area = roi_area_px2
    density_unit = "vehicles/px^2"
    if args.px_per_meter is not None:
        if args.px_per_meter <= 0:
            sys.exit("[ERROR] --px-per-meter must be greater than 0.")
        density_area = roi_area_px2 / float(args.px_per_meter ** 2)
        density_unit = "vehicles/m^2"

    if args.density:
        print(f"[INFO] ROI area: {roi_area_px2:.2f} px^2")
        if density_unit == "vehicles/m^2":
            print(f"[INFO] Density scale: {args.px_per_meter:.4f} px/m  -> area {density_area:.4f} m^2")

    # ── Model ──────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {args.weights}")
    model = YOLO(args.weights)
    target_classes = set(args.classes)
    class_names = {cid: COCO_VEHICLE_NAMES.get(cid, str(cid)) for cid in target_classes}
    print(f"[INFO] Tracking classes: {class_names}")

    # ── Video writer ───────────────────────────────────────────────
    output_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Cannot create output video: {output_path}")

    # ── Frame-by-frame processing ─────────────────────────────────
    frame_reports: List[Dict] = []
    frame_idx = 0
    t_start = time.time()

    print(f"[INFO] Processing {total_frames} frames …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_frame = time.time()

        count, boxes, cids, confs, names = process_frame(
            model, frame, roi, target_classes, args.conf, args.iou,
        )

        elapsed = time.time() - t_frame
        fps_val = 1.0 / elapsed if elapsed > 0 else 0.0

        density_value: Optional[float] = None
        density_text: Optional[str] = None
        if args.density:
            density_value = float(count) / density_area
            density_text = f"Density: {density_value:.6f} {density_unit}"

        # Annotate
        draw_overlay(frame, roi, boxes, cids, confs, names, count, frame_idx, fps_val, density_text)
        writer.write(frame)

        # Per-frame JSON data
        frame_payload = {
            "frame": frame_idx,
            "vehicle_count": count,
            "detections": [
                {
                    "class_id": int(cid),
                    "class_name": names.get(cid, str(cid)),
                    "confidence": round(float(c), 4),
                    "bbox_xyxy": [round(float(v), 1) for v in box],
                }
                for box, cid, c in zip(boxes, cids, confs)
            ],
        }
        if args.density:
            frame_payload["density"] = round(float(density_value), 8) if density_value is not None else None
            frame_payload["density_unit"] = density_unit
        frame_reports.append(frame_payload)

        frame_idx += 1

        # Progress every 100 frames
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            pct = frame_idx / total_frames * 100 if total_frames else 0
            print(f"  [{frame_idx}/{total_frames}] {pct:5.1f}%  —  vehicles in ROI: {count}")

    total_time = time.time() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    cap.release()
    writer.release()

    print(f"\n[DONE] Processed {frame_idx} frames in {total_time:.1f}s  (avg {avg_fps:.1f} fps)")
    print(f"[DONE] Output video saved: {output_path}")

    # ── JSON report ────────────────────────────────────────────────
    if args.json_out:
        counts = [fr["vehicle_count"] for fr in frame_reports]
        summary = {
            "input_video": str(input_path),
            "output_video": str(output_path),
            "model_weights": args.weights,
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "classes_filtered": {str(c): COCO_VEHICLE_NAMES.get(c, str(c)) for c in sorted(target_classes)},
            "roi_polygon": roi.tolist(),
            "roi_area_px2": round(roi_area_px2, 4),
            "density_enabled": bool(args.density),
            "total_frames": frame_idx,
            "source_fps": round(src_fps, 2),
            "processing_time_s": round(total_time, 2),
            "avg_processing_fps": round(avg_fps, 2),
            "summary": {
                "min_vehicles": int(np.min(counts)) if counts else 0,
                "max_vehicles": int(np.max(counts)) if counts else 0,
                "mean_vehicles": round(float(np.mean(counts)), 2) if counts else 0.0,
            },
            "per_frame": frame_reports,
        }
        if args.density:
            density_values = [fr["density"] for fr in frame_reports if fr.get("density") is not None]
            summary["density_unit"] = density_unit
            if density_unit == "vehicles/m^2":
                summary["px_per_meter"] = args.px_per_meter
                summary["roi_area_m2"] = round(density_area, 6)
            summary["summary"]["min_density"] = round(float(np.min(density_values)), 8) if density_values else 0.0
            summary["summary"]["max_density"] = round(float(np.max(density_values)), 8) if density_values else 0.0
            summary["summary"]["mean_density"] = round(float(np.mean(density_values)), 8) if density_values else 0.0

        json_path = Path(args.json_out)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[DONE] JSON report saved: {json_path}")


if __name__ == "__main__":
    main()
