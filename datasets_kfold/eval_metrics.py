# eval_metrics.py
from ultralytics import YOLO

model = YOLO("runs/kfold/fold_0/weights/best.pt")

metrics = model.val(conf=0.3)

print("📊 Recall per class:", metrics.box.recall)
print("📊 Precision:", metrics.box.precision)

# FN 계산 (개념적)
# FN = GT - TP
tp = metrics.box.tp
gt = metrics.box.gt
fn = gt - tp

print("📉 FN per class:", fn)
