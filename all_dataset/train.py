import os
import csv
from ultralytics import YOLO

# =========================
# 설정
# =========================
K = 5
BASE_DIR = "all_dataset/dataset_kfold"

MODEL_NAME = "yolo26n.pt"  

IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 120
DEVICE = 0
WORKERS = 8
SEED = 42

RESULT_CSV = "results/kfold_basic_metrics_yolov26n.csv"
os.makedirs("results", exist_ok=True)

# =========================
# Train + 기본 metric 수집
# =========================
def train_kfold():
    with open(RESULT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            "mAP50",
            "mAP50-95",
            "Recall@0.5",
            "Precision@0.5"
        ])

        for fold in range(1, K + 1):
            print(f"\n🚀 Training Fold {fold}")

            data_yaml = os.path.join(
                BASE_DIR, f"fold_{fold}", "data.yaml"
            )

            model = YOLO(MODEL_NAME)

            model.train(
                data=data_yaml,
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH_SIZE,
                device=DEVICE,
                workers=WORKERS,
                seed=SEED,
                project="runs/kfold",
                name=f"yolov26n_fold_{fold}",
                exist_ok=True,
                patience=25,
                close_mosaic=15,
                amp=True
            )

            metrics = model.val(
                data=data_yaml,
                imgsz=IMG_SIZE,
                device=DEVICE
            )

            writer.writerow([
                fold,
                metrics.box.map50,
                metrics.box.map,
                metrics.box.recall,
                metrics.box.precision
            ])

            print(
                f"Fold {fold} | "
                f"R@0.5={metrics.box.recall:.4f}, "
                f"mAP50={metrics.box.map50:.4f}"
            )

    print("✅ K-fold 학습 완료")


if __name__ == "__main__":
    train_kfold()
