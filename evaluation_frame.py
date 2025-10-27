from utils import *
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from ultralytics import YOLO
import time
import statistics as stats

#Define models

IMAGE_DIR = "images"
GT_DIR = "labels"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "vlm"
LABEL_DIR = "predictionlabel"

IOU_MATCH_PRF1 = 0.5
MAP_THRESHOLDS = np.arange(0.5, 0.96, 0.05)

print(f"model: {MODEL_NAME}")


device = torch.device("cuda")

if MODEL_NAME == "vlm":
    MODEL_ID = "IDEA-Research/grounding-dino-tiny"
    TEXT_PROMPT = "person."  
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    TEXT_THRESHOLD = 0.25
    BOX_THRESHOLD = 0.25

elif MODEL_NAME == "yolo":
    model = YOLO("yolov8n.pt")

else:
    raise ValueError(f"Unknown model: {MODEL_NAME}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_inference(model_name: str, image_dir: str, output_dir: str):

    model_name = model_name
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n inference with model: {model_name.upper()} on {DEVICE}")

    # model
    if model_name == "vlm":
        MODEL_ID = "IDEA-Research/grounding-dino-tiny"
        TEXT_PROMPT = "person."
        BOX_THRESHOLD = 0.25
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)
        model.eval()

    elif model_name == "yolo":
        model = YOLO("teacher_student-Copy1/runs_student/yolov8n_ft/weights/best.pt")

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        t0 = time.perf_counter()

        # vlm
        if model_name == "vlm":
            inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([[h, w]]).to(DEVICE)
            results = processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes, threshold=BOX_THRESHOLD
            )[0]

            boxes = results["boxes"]
            yolo_boxes = convert_to_yolo(boxes, w, h, class_id=0)

        # yolo
        else:
            res = model(image_path, conf=0.25, iou=0.25, classes=[0], imgsz=960)[0]
            yolo_boxes = []
            for box in res.boxes:
                cls = int(box.cls)
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                yolo_line = xyxy_to_yolo_line(x_min, y_min, x_max, y_max, w, h, class_id=cls)
                yolo_boxes.append(yolo_line)

                
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        print(latency_ms)
        # save prediction 
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(output_dir, label_filename)
        save_yolo_labels(yolo_boxes, label_path)
        print(f" saved predictions in {label_path}")

    print(f"\n every detections converted to yolo labels for model: {model_name.upper()}\n")


def eval_image_pair(gt_path, pred_path, img_path):
    with Image.open(img_path) as im:
        W, H = im.size

    gt_lines = open(gt_path).read().strip().splitlines() if os.path.exists(gt_path) else []
    pred_lines = open(pred_path).read().strip().splitlines() if os.path.exists(pred_path) else []

    gt_xyxy, pred_xyxy = [], []
    for ln in gt_lines:
        parsed = parse_yolo_line(ln)
        if parsed:
            _, cx, cy, w, h, _ = parsed
            gt_xyxy.append(yolo_to_xyxy_abs(cx, cy, w, h, W, H))

    for ln in pred_lines:
        parsed = parse_yolo_line(ln)
        if parsed:
            _, cx, cy, w, h, _ = parsed
            pred_xyxy.append(yolo_to_xyxy_abs(cx, cy, w, h, W, H))

    TP, FP, FN = match_boxes_xyxy(gt_xyxy, pred_xyxy, IOU_MATCH_PRF1)
    P, R, F1 = compute_metrics(TP, FP, FN)
    map50 = ap_at_iou(gt_xyxy, pred_xyxy, 0.5)
    map5095 = mAP_50_95(gt_xyxy, pred_xyxy, MAP_THRESHOLDS)
    return P, R, F1, map50, map5095, TP, FP, FN


def evaluate_dataset(gt_dir, pred_dir, img_dir, model_name: str):
    print(f"\n evaluating results for model: {model_name.upper()}")
    total_TP = total_FP = total_FN = 0
    results = []

    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0]
        img_path = find_image_path(img_dir, stem)
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        P, R, F1, m50, m5095, TP, FP, FN = eval_image_pair(gt_path, pred_path, img_path)
        total_TP += TP
        total_FP += FP
        total_FN += FN
        results.append([P, R, F1, m50, m5095])

        print(f"{fname}: P={P:.3f}, R={R:.3f}, F1={F1:.3f}, mAP@0.5={m50:.3f}, mAP@0.5:0.95={m5095:.3f}")

    if results:
        mean = np.mean(np.array(results), axis=0)
        print("\n image mean")
        print(f"P={mean[0]:.3f}, R={mean[1]:.3f}, F1={mean[2]:.3f}, mAP@0.5={mean[3]:.3f}, mAP@0.5:0.95={mean[4]:.3f}")
    else:
        print("No ground-truth .txt files found.")

    Pg, Rg, F1g = compute_metrics(total_TP, total_FP, total_FN)
    print("\n average metrics over images")
    print(f"P={Pg:.4f}, R={Rg:.4f}, F1={F1g:.4f}\n")


if __name__ == "__main__":
    model_name = "yolo"

    label_dir = f"predictions_{model_name}"

    run_inference( model_name, IMAGE_DIR, label_dir)
    evaluate_dataset(GT_DIR, label_dir, IMAGE_DIR, model_name)