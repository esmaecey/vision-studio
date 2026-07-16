from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset/data.yaml',
    epochs=30,          # CPU'da makul sürede bitmesi için düşürüldü
    imgsz=416,           # 640 yerine daha küçük boyut, CPU'yu rahatlatır
    batch=8,              # CPU için daha küçük batch
    name='ppe_detection_local',
    device='cpu'          # CPU kullanılacağını açıkça belirtiyoruz
)