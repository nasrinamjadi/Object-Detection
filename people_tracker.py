import os
import cv2
import time
import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional
from collections import deque
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO
from utils import *


VIDEO_IN  = "input.mp4"
VIDEO_OUT = "output_dino.mp4"
MODEL_ID = "IDEA-Research/grounding-dino-tiny"


#prompt for detection in vlm model 
TEXT_PROMPT = "person."



device = torch.device("cuda")
ZONE = (0.1, 0.10, 0.80, 0.80) #count peeson that there is in the zone

BOX_THRESHOLD = 0.4     
N_DET        = 1        
IOU_ASSOC_TH = 0.5        
MAX_AGE      = 15   


class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = np.array(box, dtype=np.float32) 
        self.miss = 0
        self.history = deque(maxlen=5)

    def center(self):
        x1, y1, x2, y2 = self.box
        return np.array([(x1+x2)/2.0, (y1+y2)/2.0], dtype=np.float32)

    def predict(self):

        if len(self.history) >= 2:
            v = self.history[-1] - self.history[-2]
            self.box[[0,2]] += v[0]
            self.box[[1,3]] += v[1]
        self.miss += 1
        self.history.append(self.center())

    def update(self, box):
        # update smooth
        self.box = 0.7*self.box + 0.3*np.array(box, dtype=np.float32)
        self.miss = 0
        self.history.append(self.center())


class SimpleTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def step(self, dets_xyxy):

        for t in self.tracks:
            t.predict()

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets   = set(range(len(dets_xyxy)))
        matches = []

        if self.tracks and dets_xyxy:
            iou_mat = np.zeros((len(self.tracks), len(dets_xyxy)), dtype=np.float32)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(dets_xyxy):
                    iou_mat[i, j] = iou_xyxy(t.box, d[:4])

            while True:
                i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[i, j] < IOU_ASSOC_TH:
                    break
                if i in unmatched_tracks and j in unmatched_dets:
                    matches.append((i, j))
                    unmatched_tracks.remove(i)
                    unmatched_dets.remove(j)
                iou_mat[i, :] = -1
                iou_mat[:, j] = -1

        for i, j in matches:
            self.tracks[i].update(dets_xyxy[j][:4])

        for j in unmatched_dets:
            self.tracks.append(Track(self.next_id, dets_xyxy[j][:4]))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.miss <= MAX_AGE]
        return self.tracks 


def main(model_name):
    cap = cv2.VideoCapture(VIDEO_IN)
    assert cap.isOpened(), f"Cannot open {VIDEO_IN}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    zone_px = (int(ZONE[0]*w), int(ZONE[1]*h), int(ZONE[2]*w), int(ZONE[3]*h))
    out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = SimpleTracker()
    frame_id = 0
    t0 = time.time()

    # select model
    if model_name == "vlm":
        print("Using Grounding-DINO (VLM) model")
        MODEL_ID = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device).eval()
        INFER_W, INFER_H = 960, 540

    elif model_name == "yolo":
        print("Using YOLOv8 model")
        model = YOLO("teacher_student/runs_student/yolov8n_ft/weights/best.pt").to(device)
        processor = None  
        INFER_W, INFER_H = None, None

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        dets = []

        if frame_id % N_DET == 0:
            if model_name == "vlm":
                small = cv2.resize(frame, (INFER_W, INFER_H))
                img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                with torch.no_grad():
                    inputs = processor(images=img_rgb, text=TEXT_PROMPT, return_tensors="pt").to(device)
                    outputs = model(**inputs)

                target_sizes = torch.tensor([[INFER_H, INFER_W]], device=device)
                results = processor.post_process_grounded_object_detection(
                    outputs, target_sizes=target_sizes, threshold=0.4
                )[0]

                boxes = results["boxes"].detach().cpu().numpy()
                scores = results["scores"].detach().cpu().numpy()

                for b, s in zip(boxes, scores):
                    x1, y1, x2, y2 = b
                    dets.append([
                        (x1 / INFER_W) * w,
                        (y1 / INFER_H) * h,
                        (x2 / INFER_W) * w,
                        (y2 / INFER_H) * h,
                        float(s)
                    ])

                del outputs, inputs, results
                torch.cuda.empty_cache()

            else:  # 
                res = model(frame, conf=0.5, classes=[0], imgsz=960, verbose=False)[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    dets.append([x1, y1, x2, y2, conf])

        # ---- Tracking ----
        tracks = tracker.step(dets)
        current_count = count_people_in_zone(tracks, zone_px)

        # ---- Visualization ----
        vis = frame.copy()
        x1, y1, x2, y2 = zone_px
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, f"People in zone: {current_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 50), 2)

        for t in tracks:
            bx1, by1, bx2, by2 = map(int, t.box)
            cx, cy = map(int, t.center())
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(vis, f"ID {t.id}", (bx1, max(20, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

        out.write(vis)

    cap.release()
    out.release()
    dt = time.time() - t0
    print(f"saved {VIDEO_OUT}, FPS={frame_id/dt:.2f}")


if __name__ == "__main__":
    main(model_name = "yolo")