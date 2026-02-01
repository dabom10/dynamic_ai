import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

IOU_THRESHOLDS = [0.5, 0.75]

BASE_DIR = "all_dataset/dataset_kfold"
RUN_DIR = "runs/kfold"
IMG_SIZE = 640

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def evaluate_fold(fold):
    model = YOLO(
        f"{RUN_DIR}/yolov11n_fold_{fold}/weights/best.pt"
    )

    val_dir = os.path.join(
        BASE_DIR, f"fold_{fold}", "val/images"
    )

    stats = {
        "tp_05": 0,
        "tp_075": 0,
        "fn": 0,
        "center_err": [],
        "iou_075": 0,
        "total_gt": 0
    }

    results = model.predict(
        source=val_dir,
        imgsz=IMG_SIZE,
        conf=0.001,
        iou=0.7,
        device=0,
        stream=True
    )

    for r in tqdm(results):
        gt = r.boxes.gt_boxes  # xyxy
        pred = r.boxes.xyxy.cpu().numpy()

        for g in gt:
            stats["total_gt"] += 1
            best_iou = 0
            best_p = None

            for p in pred:
                i = iou(g, p)
                if i > best_iou:
                    best_iou = i
                    best_p = p

            if best_iou >= 0.5:
                stats["tp_05"] += 1
            else:
                stats["fn"] += 1
                continue

            if best_iou >= 0.75:
                stats["tp_075"] += 1
                stats["iou_075"] += 1

            c_gt = center(g)
            c_p = center(best_p)
            stats["center_err"].append(
                np.linalg.norm(
                    np.array(c_gt) - np.array(c_p)
                )
            )

    return stats


if __name__ == "__main__":
    for fold in range(1, 6):
        s = evaluate_fold(fold)

        print(f"\n📊 Fold {fold}")
        print(f"Recall@0.5  : {s['tp_05']/s['total_gt']:.4f}")
        print(f"Recall@0.75 : {s['tp_075']/s['total_gt']:.4f}")
        print(
            f"Center Err  : "
            f"{np.mean(s['center_err']):.2f} ± "
            f"{np.std(s['center_err']):.2f}"
        )
        print(
            f"IoU≥0.75 비율: "
            f"{s['iou_075']/s['total_gt']:.4f}"
        )
