import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# =========================
# 설정
# =========================
# Make BASE_DIR the project/script root so paths work regardless of cwd.
BASE_DIR = Path(__file__).resolve().parent
# Output folder inside the project root
OUT_DIR = BASE_DIR / "MERGED_DATASET_KFOLD"
K = 5
SEED = 42

random.seed(SEED)

IMG_DIR = BASE_DIR / "train/images"
LBL_DIR = BASE_DIR / "train/labels"

# =========================
# 1. 클래스 정보 로드
# =========================
with open(BASE_DIR / "data.yaml") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml["names"]
num_classes = len(class_names)

# =========================
# 2. 이미지별 클래스 집합 생성
# =========================
image_infos = []  # (img_path, label_path, class_set)

for img_path in IMG_DIR.iterdir():
    lbl_path = LBL_DIR / f"{img_path.stem}.txt"
    class_set = set()

    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                cls = int(line.split()[0])
                class_set.add(cls)

    image_infos.append((img_path, lbl_path, class_set))

random.shuffle(image_infos)

# =========================
# 3. Greedy stratified 분배
# =========================
folds = [[] for _ in range(K)]
fold_class_counts = [defaultdict(int) for _ in range(K)]

for item in image_infos:
    _, _, cls_set = item

    # 각 fold에 넣었을 때 class imbalance 최소인 곳 선택
    scores = []
    for i in range(K):
        score = sum(fold_class_counts[i][c] for c in cls_set)
        scores.append(score)

    best_fold = scores.index(min(scores))
    folds[best_fold].append(item)

    for c in cls_set:
        fold_class_counts[best_fold][c] += 1

# =========================
# 4. Fold 디렉토리 생성
# =========================
for k in range(K):
    for split in ["train", "val"]:
        (OUT_DIR / f"fold_{k}" / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"fold_{k}" / split / "labels").mkdir(parents=True, exist_ok=True)

# =========================
# 5. 파일 복사
# =========================
for k in range(K):
    # make a set of image paths for quick membership checks (Paths are hashable)
    val_set = set(item[0] for item in folds[k])

    for j in range(K):
        split = "val" if j == k else "train"

        for img_path, lbl_path, _ in folds[j]:
            shutil.copy(
                img_path,
                OUT_DIR / f"fold_{k}" / split / "images" / img_path.name
            )
            shutil.copy(
                lbl_path,
                OUT_DIR / f"fold_{k}" / split / "labels" / lbl_path.name
            )

# =========================
# 6. fold별 data.yaml 생성
# =========================
for k in range(K):
    fold_yaml = {
        "path": str(OUT_DIR / f"fold_{k}"),
        "train": "train/images",
        "val": "val/images",
        "nc": num_classes,
        "names": class_names
    }

    with open(OUT_DIR / f"fold_{k}" / "data.yaml", "w") as f:
        yaml.dump(fold_yaml, f, sort_keys=False)

print("✅ K-Fold 분할 완료")
print(f"K = {K}")
print(f"총 이미지 수 = {len(image_infos)}")
