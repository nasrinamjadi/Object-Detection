import os
import numpy as np
from typing import List
from PIL import Image

def iou_xyxy(a: List[float], b: List[float]) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0.0:
        return 0.0

    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    denom = areaA + areaB - inter
    return inter / (denom + 1e-6)


def point_in_rect(cx, cy, r):
    return (r[0] <= cx <= r[2]) and (r[1] <= cy <= r[3])

def in_zone(point, zone_px):
    x, y = point
    x1, y1, x2, y2 = zone_px
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def count_people_in_zone(tracks, zone_px):
    #current presence
    return sum(1 for t in tracks if in_zone(t.center(), zone_px))

    
    
def xyxy_to_yolo_line(x_min, y_min, x_max, y_max, image_width, image_height, class_id=0):
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + box_width / 2
    y_center = y_min + box_height / 2

    # Normalize to 0 and 1
    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def match_boxes_xyxy(gt_boxes, pred_boxes, iou_thr=0.5):
    """greedy match between ground truth and predicted boxes"""
    matched_gt = set()
    TP = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = iou_xyxy(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou >= iou_thr:
            TP += 1
            matched_gt.add(best_idx)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - len(matched_gt)
    return TP, FP, FN
    

def yolo_to_xyxy_abs(cx, cy, w, h, W, H):
    cx *= W; cy *= H; w *= W; h *= H
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def convert_to_yolo(boxes: np.ndarray, width: int, height: int, class_id: int = 0):
    yolo_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box.tolist()
        w, h = x_max - x_min, y_max - y_min
        x_center = x_min + w / 2
        y_center = y_min + h / 2

        y_center /= height
        x_center /= width
        w /= width
        h /= height

        yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return yolo_boxes


def save_yolo_labels(yolo_boxes: List[str], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(yolo_boxes))


def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) >= 6 else 1.0
    return cls_id, cx, cy, w, h, conf


def find_image_path(img_dir: str, stem: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None



#metric functions

def compute_metrics(TP, FP, FN):
    """Compute precision - recall - F1"""
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1



def ap_at_iou(gt_boxes, pred_boxes, iou_thr):
    TP, FP, FN = match_boxes_xyxy(gt_boxes, pred_boxes, iou_thr)
    precision, _, _ = compute_metrics(TP, FP, FN)
    return precision

def mAP_50_95(gt_boxes, pred_boxes, MAP_THRESHOLDS):
    if not gt_boxes and not pred_boxes:
        return 1.0
    vals = [ap_at_iou(gt_boxes, pred_boxes, thr) for thr in MAP_THRESHOLDS]
    return float(np.mean(vals))
