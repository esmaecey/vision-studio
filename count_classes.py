import os
from collections import Counter

labels_dir = "dataset_eski/valid/labels"

class_names = {
    "0": "boots", "1": "gloves", "2": "goggles", "3": "helmet",
    "4": "no-boots", "5": "no-gloves", "6": "no-goggles",
    "7": "no-helmet", "8": "no-vest", "9": "vest"
}

counter = Counter()

for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, "r") as f:
        for line in f:
            if line.strip():
                class_id = line.split()[0]
                counter[class_id] += 1

print("Sınıf bazlı örnek (instance) sayıları:\n")
for class_id, name in class_names.items():
    count = counter.get(class_id, 0)
    print(f"{name:12s}: {count}")

print(f"\nToplam etiket dosyası: {len(os.listdir(labels_dir))}")