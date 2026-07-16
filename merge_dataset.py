import os
import shutil

# Kaynak: augmented veri setinin train klasörü
aug_images_dir = "dataset/train/images"
aug_labels_dir = "dataset/train/labels"

# Hedef: orijinal veri setinin train klasörü (buraya ekleyeceğiz)
target_images_dir = "dataset_eski/train/images"
target_labels_dir = "dataset_eski/train/labels"

# İstediğimiz zayıf sınıfların index numaraları
TARGET_CLASSES = {"4", "5", "7"}  # no-boots, no-gloves, no-helmet

added_count = 0

for label_file in os.listdir(aug_labels_dir):
    label_path = os.path.join(aug_labels_dir, label_file)
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    # Bu etiket dosyasında hedef sınıflardan biri var mı kontrol et
    has_target_class = any(line.split()[0] in TARGET_CLASSES for line in lines if line.strip())
    
    if has_target_class:
        # Karşılık gelen görsel dosyasını bul
        base_name = os.path.splitext(label_file)[0]
        
        # Görsel uzantısını bul (.jpg olabilir)
        image_file = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = base_name + ext
            if os.path.exists(os.path.join(aug_images_dir, candidate)):
                image_file = candidate
                break
        
        if image_file is None:
            print(f"UYARI: {base_name} için görsel bulunamadı, atlanıyor.")
            continue
        
        # Çakışmayı önlemek için "aug_" ön eki ekleyerek kopyala
        new_label_name = "aug_" + label_file
        new_image_name = "aug_" + image_file
        
        shutil.copy(label_path, os.path.join(target_labels_dir, new_label_name))
        shutil.copy(os.path.join(aug_images_dir, image_file), os.path.join(target_images_dir, new_image_name))
        
        added_count += 1

print(f"\nToplam {added_count} yeni görsel+etiket eklendi.")
print(f"Yeni train klasörü toplam görsel sayısı: {len(os.listdir(target_images_dir))}")