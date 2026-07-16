from ultralytics import YOLO

# Modeli indir (ilk çalıştırmada otomatik indirilir, birkaç MB)
model = YOLO('yolov8n-pose.pt')

print("Model başarıyla yüklendi!")
print(model.info())