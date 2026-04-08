import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

TARGET_CLASS_IDS = {2, 5, 7}  # COCO ids: car, bus, truck


def parse_roi(roi_text: str) -> np.ndarray:
    """
    Parse ROI from string format: "x1,y1 x2,y2 x3,y3 ..."
    Returns Nx2 int32 numpy array.
    """
    points: List[Tuple[int, int]] = []
    for part in roi_text.strip().split():
        x_str, y_str = part.split(",")
        points.append((int(float(x_str)), int(float(y_str))))

    if len(points) < 3:
        raise ValueError("ROI must have at least 3 points.")

    return np.array(points, dtype=np.int32)


def bbox_center_xyxy(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    # cv2.pointPolygonTest returns >= 0 when point is inside or on edge.
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def select_roi_by_click(image: np.ndarray, required_points: int = 4) -> np.ndarray:
    points: List[Tuple[int, int]] = []
    window_name = "Select ROI: click 4 points, Enter=confirm, r=reset, q=quit"

    h, w = image.shape[:2]
    # Fit the preview on screen for high-resolution images while preserving coordinates.
    max_w = 1600
    max_h = 900
    scale = min(max_w / w, max_h / h, 1.0)
    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    image_display = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else image.copy()

    display = image_display.copy()

    def redraw() -> None:
        nonlocal display
        display = image_display.copy()
        for idx, (x, y) in enumerate(points, start=1):
            dx = int(round(x * scale))
            dy = int(round(y * scale))
            cv2.circle(display, (dx, dy), 5, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(idx),
                (dx + 6, dy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(points) >= 2:
            pts = np.array(
                [[int(round(px * scale)), int(round(py * scale))] for px, py in points],
                dtype=np.int32,
            )
            cv2.polylines(display, [pts], False, (0, 255, 255), 2)

        if len(points) == required_points:
            pts = np.array(
                [[int(round(px * scale)), int(round(py * scale))] for px, py in points],
                dtype=np.int32,
            )
            cv2.polylines(display, [pts], True, (0, 255, 255), 2)

        cv2.putText(
            display,
            f"Points: {len(points)}/{required_points}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < required_points:
            ox = int(round(x / scale))
            oy = int(round(y / scale))
            ox = min(max(ox, 0), w - 1)
            oy = min(max(oy, 0), h - 1)
            points.append((ox, oy))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            cv2.destroyWindow(window_name)
            raise RuntimeError("ROI selection cancelled.")

        if key == ord("r"):
            points.clear()
            redraw()

        if key in (13, 10) and len(points) == required_points:
            cv2.destroyWindow(window_name)
            return np.array(points, dtype=np.int32)


def draw_results(
    image: np.ndarray,
    roi_polygon: np.ndarray,
    kept_boxes: List[np.ndarray],
    labels: List[str],
    confidences: List[float],
    count: int,
) -> np.ndarray:
    output = image.copy()

    cv2.polylines(output, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    for box, label, conf in zip(kept_boxes, labels, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            output,
            f"{label} {conf:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"Vehicles in ROI: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


def count_vehicles_in_roi(
    image_path: Path,
    model_path: str,
    roi_polygon: np.ndarray,
    conf_threshold: float,
) -> Tuple[int, np.ndarray, List[np.ndarray], List[str], List[float]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = YOLO(model_path)
    result = model.predict(source=image, conf=conf_threshold, verbose=False)[0]

    boxes = result.boxes
    kept_boxes: List[np.ndarray] = []
    kept_labels: List[str] = []
    kept_conf: List[float] = []
    names = result.names

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for box, class_id, score in zip(xyxy, cls, conf):
            if class_id not in TARGET_CLASS_IDS:
                continue
            center = bbox_center_xyxy(box)
            if point_in_polygon(center, roi_polygon):
                kept_boxes.append(box)
                kept_labels.append(str(names.get(class_id, class_id)))
                kept_conf.append(float(score))

    return len(kept_boxes), image, kept_boxes, kept_labels, kept_conf


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count cars, buses, and trucks inside an ROI in an aerial highway image using YOLOv8."
    )
    parser.add_argument("--image", required=True, help="Input image path (jpg/png/jpeg/etc).")
    parser.add_argument(
        "--roi",
        default="",
        help='ROI polygon points: "x1,y1 x2,y2 x3,y3 ..."',
    )
    parser.add_argument(
        "--click-roi",
        action="store_true",
        help="Select ROI interactively by clicking 4 points on the image.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 weights file or model name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output path for annotated image.",
    )
    parser.add_argument(
        "--json",
        default="",
        help="Optional output path for JSON results.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if args.click_roi or not args.roi:
        roi_polygon = select_roi_by_click(raw_image, required_points=4)
    else:
        roi_polygon = parse_roi(args.roi)

    count, image, kept_boxes, kept_labels, kept_conf = count_vehicles_in_roi(
        image_path=image_path,
        model_path=args.model,
        roi_polygon=roi_polygon,
        conf_threshold=args.conf,
    )

    print(f"Vehicles (car/bus/truck) in ROI: {count}")

    if args.save:
        annotated = draw_results(image, roi_polygon, kept_boxes, kept_labels, kept_conf, count)
        cv2.imwrite(args.save, annotated)
        print(f"Saved annotated image: {args.save}")

    if args.json:
        payload = {
            "image": str(image_path),
            "model": args.model,
            "confidence_threshold": args.conf,
            "vehicle_count_in_roi": count,
            "classes_counted": ["car", "bus", "truck"],
            "roi": roi_polygon.tolist(),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON output: {args.json}")


if __name__ == "__main__":
    main()
