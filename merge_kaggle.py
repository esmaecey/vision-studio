import os
import shutil

# Kaynak: Kaggle veri seti
kaggle_images_dir = "dataset_kaggle/train/images"
kaggle_labels_dir = "dataset_kaggle/train/labels"

# Hedef: bizim ana veri setimiz
target_images_dir = "dataset_eski/train/images"
target_labels_dir = "dataset_eski/train/labels"

# Kaggle ID -> Bizim ID eşlemesi
CLASS_MAP = {
    "0": "5",  # no-safety-glove -> no-gloves
    "1": "7",  # no-safety-helmet -> no-helmet
    "2": "4",  # no-safety-shoes -> no-boots
    "3": "6",  # no-welding-glass -> no-goggles
    "4": "1",  # safety-glove -> gloves
    "5": "3",  # safety-helmet -> helmet
    "6": "0",  # safety-shoes -> boots
    "7": "2",  # welding-glass -> goggles
}

# Sadece bu sınıfları içeren görselleri hedefliyoruz (en zayıf 3 sınıf)
PRIORITY_KAGGLE_CLASSES = {"2", "3", "7"}  # no-safety-shoes, no-welding-glass, welding-glass

# Her sınıf için maksimum kaç görsel ekleyeceğimiz
MAX_PER_CLASS = 700

class_counts = {"2": 0, "3": 0, "7": 0}
added_count = 0

for label_file in os.listdir(kaggle_labels_dir):
    label_path = os.path.join(kaggle_labels_dir, label_file)
    
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        continue
    
    kaggle_ids_in_file = {line.split()[0] for line in lines}
    
    # Bu görselde öncelikli sınıflarımızdan biri var mı, ve o sınıfın kotası dolmadı mı?
    relevant_priority = kaggle_ids_in_file & PRIORITY_KAGGLE_CLASSES
    if not relevant_priority:
        continue
    
    # Kota kontrolü — herhangi biri hâlâ limitin altındaysa devam et
    if all(class_counts[cid] >= MAX_PER_CLASS for cid in relevant_priority):
        continue
    
    # Görsel dosyasını bul
    base_name = os.path.splitext(label_file)[0]
    image_file = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = base_name + ext
        if os.path.exists(os.path.join(kaggle_images_dir, candidate)):
            image_file = candidate
            break
    
    if image_file is None:
        continue
    
    # Etiketleri bizim sistemimize göre yeniden yaz
    new_lines = []
    for line in lines:
        parts = line.split()
        kaggle_id = parts[0]
        if kaggle_id in CLASS_MAP:
            new_id = CLASS_MAP[kaggle_id]
            new_lines.append(" ".join([new_id] + parts[1:]))
    
    if not new_lines:
        continue
    
    # Kaydet
    new_label_name = "kg_" + label_file
    new_image_name = "kg_" + image_file
    
    with open(os.path.join(target_labels_dir, new_label_name), "w") as f:
        f.write("\n".join(new_lines) + "\n")
    
    shutil.copy(os.path.join(kaggle_images_dir, image_file), os.path.join(target_images_dir, new_image_name))
    
    for cid in relevant_priority:
        class_counts[cid] += 1
    
    added_count += 1

print(f"Toplam {added_count} yeni görsel eklendi.")
print(f"Sınıf bazlı katkı: no-safety-shoes(no-boots)={class_counts['2']}, "
      f"no-welding-glass(no-goggles)={class_counts['3']}, welding-glass(goggles)={class_counts['7']}")