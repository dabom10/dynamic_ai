from pathlib import Path
from collections import defaultdict
import yaml
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================
KFOLD_DIR = Path("MERGED_DATASET_KFOLD")
K = 5

# =========================
# 클래스 정보 로드
# =========================
with open(KFOLD_DIR / "fold_0" / "data.yaml") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml["names"]
num_classes = len(class_names)

# =========================
# 클래스 카운트 함수
# =========================
def count_classes(label_dir):
    counts = defaultdict(int)

    for lbl_file in label_dir.iterdir():
        with open(lbl_file) as f:
            for line in f:
                cls = int(line.split()[0])
                counts[cls] += 1

    return counts

# =========================
# Fold별 시각화
# =========================
for k in range(K):
    fold_path = KFOLD_DIR / f"fold_{k}"

    for split in ["train", "val"]:
        label_dir = fold_path / split / "labels"
        counts = count_classes(label_dir)

        values = [counts[i] for i in range(num_classes)]

        plt.figure()
        plt.bar(class_names, values)
        plt.title(f"Fold {k} - {split} class distribution")
        plt.xlabel("Class")
        plt.ylabel("Object count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
