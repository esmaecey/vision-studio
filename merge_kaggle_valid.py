import os
import shutil

# Kaggle'ın VALID klasöründen çekiyoruz (train'den değil!)
kaggle_images_dir = "dataset_kaggle/valid/images"
kaggle_labels_dir = "dataset_kaggle/valid/labels"

# Hedef: bizim valid setimiz
target_images_dir = "dataset_eski/valid/images"
target_labels_dir = "dataset_eski/valid/labels"

# Kaggle ID -> Bizim ID eşlemesi (aynı mapping)
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

# Val setinde zayıf olan sınıfları hedefliyoruz
PRIORITY_KAGGLE_CLASSES = {"2", "3", "7", "0"}  # no-shoes, no-glass, glass, no-glove

# Val seti train'den küçük olmalı, o yüzden daha az ekliyoruz
MAX_PER_CLASS = 200

class_counts = {cid: 0 for cid in PRIORITY_KAGGLE_CLASSES}
added_count = 0

for label_file in os.listdir(kaggle_labels_dir):
    label_path = os.path.join(kaggle_labels_dir, label_file)

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        continue

    kaggle_ids_in_file = {line.split()[0] for line in lines}
    relevant = kaggle_ids_in_file & PRIORITY_KAGGLE_CLASSES
    if not relevant:
        continue

    if all(class_counts[cid] >= MAX_PER_CLASS for cid in relevant):
        continue

    base_name = os.path.splitext(label_file)[0]
    image_file = None
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = base_name + ext
        if os.path.exists(os.path.join(kaggle_images_dir, candidate)):
            image_file = candidate
            break

    if image_file is None:
        continue

    new_lines = []
    for line in lines:
        parts = line.split()
        kaggle_id = parts[0]
        if kaggle_id in CLASS_MAP:
            new_lines.append(" ".join([CLASS_MAP[kaggle_id]] + parts[1:]))

    if not new_lines:
        continue

    new_label_name = "kgval_" + label_file
    new_image_name = "kgval_" + image_file

    with open(os.path.join(target_labels_dir, new_label_name), "w") as f:
        f.write("\n".join(new_lines) + "\n")

    shutil.copy(os.path.join(kaggle_images_dir, image_file),
                os.path.join(target_images_dir, new_image_name))

    for cid in relevant:
        class_counts[cid] += 1

    added_count += 1

print(f"Valid setine {added_count} yeni görsel eklendi.")
print("Sınıf katkıları:", class_counts)