import os
import shutil
import yaml
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 설정
# =========================
K = 5
SEED = 42

BASE_DIR = "all_dataset"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
OUT_DIR = os.path.join(BASE_DIR, "dataset_kfold")
DATA_YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

IMG_EXTS = [".jpg", ".jpeg", ".png"]

np.random.seed(SEED)

# =========================
# 유틸 함수
# =========================
def load_data_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_data_yaml(base_yaml, save_path, train_img_path, val_img_path):
    new_yaml = base_yaml.copy()
    new_yaml["train"] = train_img_path
    new_yaml["val"] = val_img_path
    with open(save_path, "w") as f:
        yaml.dump(new_yaml, f, sort_keys=False)


def get_box_vector(label_path, num_classes):
    """
    이미지 하나에 대해 클래스별 박스 수 벡터 생성
    """
    vec = np.zeros(num_classes)
    with open(label_path) as f:
        for line in f:
            cls = int(line.split()[0])
            vec[cls] += 1
    return vec


# =========================
# 메인
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base_yaml = load_data_yaml(DATA_YAML_PATH)
    class_names = base_yaml["names"]
    num_classes = base_yaml["nc"]

    images = []
    box_vectors = []

    # -------------------------
    # 데이터 로드
    # -------------------------
    for img in os.listdir(IMG_DIR):
        if os.path.splitext(img)[1].lower() not in IMG_EXTS:
            continue

        label_path = os.path.join(LBL_DIR, img.rsplit(".", 1)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        images.append(img)
        box_vectors.append(get_box_vector(label_path, num_classes))

    box_vectors = np.array(box_vectors)
    num_images = len(images)

    print(f"총 이미지 수: {num_images}")

    # -------------------------
    # Greedy box-balanced K-fold
    # -------------------------
    fold_indices = {i: [] for i in range(K)}
    fold_box_sum = {i: np.zeros(num_classes) for i in range(K)}

    # 희소 클래스 우선 배치
    class_rarity = box_vectors.sum(axis=0)
    img_order = np.argsort([
        -box_vectors[i][np.argmin(class_rarity)] for i in range(num_images)
    ])

    for idx in img_order:
        img_boxes = box_vectors[idx]

        best_fold = min(
            range(K),
            key=lambda f: np.max(fold_box_sum[f] + img_boxes)
        )

        fold_indices[best_fold].append(idx)
        fold_box_sum[best_fold] += img_boxes

    # -------------------------
    # fold별 train/val 생성
    # -------------------------
    for f in range(K):
        fold_dir = os.path.join(OUT_DIR, f"fold_{f+1}")

        train_img_dir = os.path.join(fold_dir, "train/images")
        train_lbl_dir = os.path.join(fold_dir, "train/labels")
        val_img_dir = os.path.join(fold_dir, "val/images")
        val_lbl_dir = os.path.join(fold_dir, "val/labels")

        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)

        val_idx = fold_indices[f]
        train_idx = [i for k in range(K) if k != f for i in fold_indices[k]]

        for idx in train_idx:
            img = images[idx]
            shutil.copy(os.path.join(IMG_DIR, img), train_img_dir)
            shutil.copy(
                os.path.join(LBL_DIR, img.rsplit(".", 1)[0] + ".txt"),
                train_lbl_dir
            )

        for idx in val_idx:
            img = images[idx]
            shutil.copy(os.path.join(IMG_DIR, img), val_img_dir)
            shutil.copy(
                os.path.join(LBL_DIR, img.rsplit(".", 1)[0] + ".txt"),
                val_lbl_dir
            )

        save_data_yaml(
            base_yaml,
            os.path.join(fold_dir, "data.yaml"),
            "./train/images",
            "./val/images"
        )

    # -------------------------
    # 박스 수 기준 분포 시각화
    # -------------------------
    for f in range(K):
        train_boxes = np.zeros(num_classes)
        val_boxes = np.zeros(num_classes)

        val_idx = fold_indices[f]
        train_idx = [i for k in range(K) if k != f for i in fold_indices[k]]

        for idx in train_idx:
            train_boxes += box_vectors[idx]

        for idx in val_idx:
            val_boxes += box_vectors[idx]

        plt.figure(figsize=(12, 4))
        plt.bar(class_names, train_boxes, alpha=0.7, label="train")
        plt.bar(class_names, val_boxes, alpha=0.7, label="val")
        plt.title(f"Fold {f+1} - Box Count Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("✅ Box-balanced K-Fold 분할 + 시각화 완료")


if __name__ == "__main__":
    main()
