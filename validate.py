from ultralytics import YOLO

model = YOLO("runs/detect/modelo_v2/weights/best.pt")

metrics = model.val(data="dataset/data.yaml")

print(metrics)
