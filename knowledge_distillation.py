import os, random, shutil, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import cv2
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from utils import *
import yaml
from pathlib import Path

try:
    from ultralytics.utils.loss import v8DetectionLoss
except Exception:
    from ultralytics.yolo.utils.loss import v8DetectionLoss
device = torch.device("cuda")

unlabeled_img_dir = "dataset"
project_dir       = "teacher_student"
pseudo_label_dir  = f"{project_dir}/pseudo_labels"
dataset_dir       = f"{project_dir}/dataset_yolo"
im_train = f"{dataset_dir}/images/train"
im_val   = f"{dataset_dir}/images/val"
lb_train = f"{dataset_dir}/labels/train"
lb_val   = f"{dataset_dir}/labels/val"

# teacher (vlm as teacher model)
model_id       = "IDEA-Research/grounding-dino-tiny"
text_prompt    = "person."
box_threshold  = 0.25
text_threshold = 0.25
cls_id         = 0
class_names    = ["person"]

# student (YOLO model as student)
stu_weights = "yolov8n.pt"
imgsize       = 960
batch_size  = 2
epochs      = 50
lr          = 1e-3
num_workers = 2
conf_thres  = 0.25
iou_nms     = 0.25

# split
val_ratio = 0.1
random.seed(42)

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_id)
teacher = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()



# make folders
for d in [pseudo_label_dir, im_train, im_val, lb_train, lb_val]:
    Path(d).mkdir(parents=True, exist_ok=True)

def list_images(folder):
    exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
    return [str(Path(folder)/f) for f in os.listdir(folder) if f.lower().endswith(exts)]

# === first: make pseudo labels  using vlm model for all  unlabeld images ===
for filename in sorted(os.listdir(unlabeled_img_dir)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(unlabeled_img_dir, filename)
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    inputs = processor(images = image, text = text_prompt, return_tensors = "pt").to(device)
    with torch.no_grad():
        outputs = teacher(**inputs)

    target_sizes = torch.tensor([[h, w]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs, target_sizes = target_sizes, threshold = box_threshold
    )[0]

    boxes = results["boxes"]

    yolo_boxes = convert_to_yolo(boxes, w, h, class_id = cls_id)
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(pseudo_label_dir, label_filename)
    save_yolo_labels(yolo_boxes, label_path)


print("\n all images labeled by teacher(vlm) model.\n")

imgs = list_images(unlabeled_img_dir)
len(imgs)
pairs = []
for p in imgs:
    t = Path(pseudo_label_dir)/(Path(p).stem + ".txt")
    if t.exists():
        pairs.append((p, str(t)))

random.shuffle(pairs)
val_n = max(1, int(len(pairs) * val_ratio))
val_pairs = pairs[:val_n]
train_pairs = pairs[val_n:]

def copy_pairs(pairs, split):
    imdst = Path(im_train if split == "train" else im_val)
    lbdst = Path(lb_train if split == "train" else lb_val)
    moved = 0
    for im, lb in pairs:
        shutil.copy2(im, imdst/Path(im).name)
        shutil.copy2(lb, lbdst/(Path(im).stem + ".txt"))
        moved += 1
    print(f" Moved : {moved} and pairs : {split}")

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs,   "val")

print("\n unlabeled_img ready for yolo finetuning")


# preprocessing dataset for yolo model
imgsize = 960
class YoloV8TxtDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=960):
        self.img_paths = sorted([str(Path(img_dir)/f) for f in os.listdir(img_dir)
                                 if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
        self.lbl_dir = Path(lbl_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        img0 = cv2.imread(ip)
        img = cv2.resize(img0, (self.img_size, self.img_size))
        tl = Path(ip).stem + ".txt"
        labels = []
        lp = self.lbl_dir / tl
        if lp.exists():
            with open(lp) as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.strip().split())
                    labels.append([cls, cx, cy, bw, bh])
        labels = np.array(labels, dtype=np.float32) if len(labels) else np.zeros((0,5), dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img = np.transpose(img, (2,0,1))
        return torch.from_numpy(img), torch.from_numpy(labels), ip

def collate_y8(batch):
    imgs, labels, paths = list(zip(*batch))
    imgs = torch.stack(imgs, 0).float()
    cls_list, box_list, bid_list = [], [], []
    for i, lb in enumerate(labels):
        if lb.numel():
            c = lb[:, 0:1]            
            b = lb[:, 1:5]            
            bid = torch.full((lb.shape[0],), i)
            cls_list.append(c); box_list.append(b); bid_list.append(bid)
    if len(cls_list):
        cls = torch.cat(cls_list, 0).float()
        bboxes = torch.cat(box_list, 0).float()
        batch_idx = torch.cat(bid_list, 0).long()
    else:
        cls = torch.zeros((0,1), dtype=torch.float32)
        bboxes = torch.zeros((0,4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.long)
    return {"img": imgs, "cls": cls, "bboxes": bboxes, "batch_idx": batch_idx, "im_file": list(paths)}

train_ds = YoloV8TxtDataset(im_train, lb_train, img_size = imgsize)
val_ds   = YoloV8TxtDataset(im_val,   lb_val,   img_size = imgsize)

train_loader = DataLoader(train_ds, batch_size = 2, shuffle=True,  num_workers = 2, pin_memory=True, collate_fn=collate_y8)
val_loader   = DataLoader(val_ds,   batch_size = 2, shuffle=False, num_workers = 2, pin_memory=True, collate_fn=collate_y8)

print(f"train images: {len(train_ds)}, val images: {len(val_ds)}")



data_yaml_path = Path(project_dir) / "data_kd.yaml"
data_cfg = {
    "path": f"{project_dir}/dataset_yolo",   # base path
    "train": "images/train",
    "val": "images/val",
    "nc": 1,
    "names": ["person"]
}

data_yaml_path.write_text(yaml.safe_dump(data_cfg, sort_keys=False), encoding="utf-8")
print(f"created data.yaml at: {data_yaml_path}")



yolo_student = YOLO(stu_weights)

save_root = Path(project_dir) / "runs_student"
save_root.mkdir(parents=True, exist_ok=True)

results = yolo_student.train(
    data=str(data_yaml_path),
    epochs= epochs,
    imgsz = 960,
    batch = 2,
    lr0 = lr ,                 
    optimizer = "SGD",
    devic e= "cuda" ,
    workers = 2,
    project = str(save_root),
    name="yolov8n_ft",
    seed = 42,
    verbose = True,
    val = False,             
    plots = True,
)

try:
    run_dir = Path(yolo_student.trainer.save_dir)  
except Exception:
    run_dir = max(save_root.glob("yolov8n_kd*"), key=lambda p: p.stat().st_mtime)

weights_dir = run_dir / "weights"
best_ckpt = weights_dir / "best.pt"
last_ckpt = weights_dir / "last.pt"
