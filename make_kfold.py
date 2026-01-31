# make_kfold.py
import os
import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "datasets_kfold"
K = 5

images = sorted([f for f in os.listdir(f"{DATASET_DIR}/images") if f.endswith(".jpg")])

def get_main_class(label_path):
    with open(label_path, "r") as f:
        classes = [int(line.split()[0]) for line in f.readlines()]
    return max(set(classes), key=classes.count)

y = []
for img in images:
    label_path = f"{DATASET_DIR}/labels/{img.replace('.jpg', '.txt')}"
    y.append(get_main_class(label_path))

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(images, y)):
    for split, idxs in zip(["train", "val"], [train_idx, val_idx]):
        img_out = f"{OUT_DIR}/fold_{fold}/{split}/images"
        lbl_out = f"{OUT_DIR}/fold_{fold}/{split}/labels"
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for i in idxs:
            img = images[i]
            lbl = img.replace(".jpg", ".txt")

            shutil.copy(f"{DATASET_DIR}/images/{img}", img_out)
            shutil.copy(f"{DATASET_DIR}/labels/{lbl}", lbl_out)

print("✅ Stratified K-Fold split 완료")
