# train.py
from ultralytics import YOLO
import yaml
import os

K = 5
BASE_YAML = "data.yaml"

for fold in range(K):
    print(f"\n🚀 Training fold {fold}")

    with open(BASE_YAML, "r") as f:
        data_cfg = yaml.safe_load(f)

    # 실행 위치와 상관없이 데이터셋을 찾을 수 있도록 절대 경로 사용
    data_cfg["path"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fold_{fold}")

    fold_yaml = f"data_fold_{fold}.yaml"
    with open(fold_yaml, "w") as f:
        yaml.dump(data_cfg, f)

    model = YOLO("yolov8s.pt")

    model.train(
        data=fold_yaml,
        epochs=180,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        patience=30,
        project="runs/kfold",
        name=f"fold_{fold}",
        verbose=True
    )
