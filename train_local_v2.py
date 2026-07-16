from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='dataset_eski/data.yaml',
    epochs=20,
    imgsz=320,
    batch=8,
    device='cpu',
    patience=10,
    cache='ram',      # görselleri RAM'e önceden yükler, disk gecikmesini azaltır
    workers=4,         # CPU'ya göre ayarlanmış paralel veri yükleme
    mosaic=0.5,        # augmentation hesaplama yükünü hafifletir
    name='ppe_detection_balanced'
)