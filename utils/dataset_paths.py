"""
Ortak veri seti yol çözümleme yardımcıları.

Vision Studio'nun birden fazla modülü (Augmentation, Dataset Split, Detect/Keypoint
Labeling) YOLO veri seti yapısıyla çalışır. Bu modül, "görsel klasörü nerede,
eşleşen etiket klasörü nerede" mantığını tek yerde toplar:

  - Standart YOLO kuralı: yoldaki 'images' bileşeni 'labels' ile eşleşir.
    Örn: .../hand-mini/images/train  ->  .../hand-mini/labels/train
  - Zaten bölünmüş veri setlerinde (images/train, images/val ...) görseller bir
    kademe derinde olabilir; collect_dataset_items bunları da toplar.
"""
import os

VALID_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def dir_has_images(d):
    """Klasörde doğrudan (alt klasör değil) görsel dosyası var mı?"""
    return bool(d) and os.path.isdir(d) and any(
        f.lower().endswith(VALID_IMG_EXT) and os.path.isfile(os.path.join(d, f))
        for f in os.listdir(d)
    )


def derive_labels_dir(images_dir):
    """
    YOLO kuralı: yoldaki son 'images' bileşenini 'labels' ile değiştirir.
    Karşılığı olan klasör mevcutsa döner, yoksa None.
    Örn: .../images/train -> .../labels/train
    """
    if not images_dir:
        return None
    norm = os.path.normpath(images_dir)
    parts = norm.split(os.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            cand = list(parts)
            cand[i] = "labels"
            cand_path = os.sep.join(cand)
            if os.path.isdir(cand_path):
                return cand_path
    return None


def resolve_dataset_dirs(folder):
    """
    Kullanıcının seçtiği klasörden (images_dir, labels_dir) çiftini çözer.

    - folder içinde 'images/' alt klasörü varsa: images_dir=folder/images,
      labels_dir=folder/labels (yoksa images->labels kuralıyla türetilir).
    - Aksi halde folder'ın kendisi görsel klasörü kabul edilir (örn.
      .../images/train); labels images->labels kuralıyla türetilir.

    Not: images_dir doğrudan görsel içermeyebilir (train/val alt klasörleri gibi);
    görselleri toplamak için collect_dataset_items kullanın.
    """
    if not folder:
        return None, None
    img_sub = os.path.join(folder, "images")
    lbl_sub = os.path.join(folder, "labels")
    if os.path.isdir(img_sub):
        images_dir = img_sub
        labels_dir = lbl_sub if os.path.isdir(lbl_sub) else (derive_labels_dir(img_sub) or lbl_sub)
    else:
        images_dir = folder
        labels_dir = derive_labels_dir(folder) or folder
    return images_dir, labels_dir


def default_output_dir(source_dir, name):
    """
    Bir kaynak (görsel) klasörü için varsayılan çıktı klasörü yolu.
    Çıktı, kullanıcının seçtiği klasörün İÇİNE konur (üst dizine değil), böylece
    .../images/train seçilince çıktı .../images/train/<name> olur.
    """
    return os.path.join(source_dir, name)


def label_path_for(image_path, images_dir, labels_dir):
    """Bir görselin etiket yolunu, images_dir'e göreli alt-yolu koruyarak döner."""
    rel = os.path.relpath(image_path, images_dir)
    rel_txt = os.path.splitext(rel)[0] + ".txt"
    return os.path.join(labels_dir, rel_txt)


def collect_dataset_items(images_dir, labels_dir):
    """
    images_dir altındaki görselleri (doğrudan ya da train/val gibi alt klasörlerde)
    toplayıp eşleşen etiketleriyle döner.

    Dönüş: (items, used_subdirs)
      items: [(image_path, label_path_or_None, out_name), ...]
             out_name çakışmayı önlemek için alt-yol düzleştirilerek üretilir
             (train/x.jpg -> "train_x.jpg"); doğrudan görsellerde sade dosya adı.
      used_subdirs: görseller alt klasörlerden toplandıysa ilk kademe klasör adları
                    (örn. ['train', 'val']); doğrudan toplandıysa boş liste.
    """
    items = []
    if not images_dir or not os.path.isdir(images_dir):
        return items, []

    # 1) Doğrudan görseller (bölünmemiş düz veri seti)
    direct = [f for f in os.listdir(images_dir)
              if f.lower().endswith(VALID_IMG_EXT)
              and os.path.isfile(os.path.join(images_dir, f))]
    if direct:
        for f in sorted(direct):
            ip = os.path.join(images_dir, f)
            lp = label_path_for(ip, images_dir, labels_dir)
            items.append((ip, lp if os.path.exists(lp) else None, f))
        return items, []

    # 2) Alt klasörlerdeki görseller (zaten bölünmüş veri seti: train/val/test ...)
    used_subdirs = set()
    for root, _, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(VALID_IMG_EXT):
                continue
            ip = os.path.join(root, f)
            rel = os.path.relpath(ip, images_dir)
            used_subdirs.add(rel.split(os.sep)[0])
            lp = label_path_for(ip, images_dir, labels_dir)
            out_name = rel.replace(os.sep, "_")
            items.append((ip, lp if os.path.exists(lp) else None, out_name))
    items.sort(key=lambda x: x[2])
    return items, sorted(used_subdirs)


# ---------------- Etiket içeriği yardımcıları ----------------
def _iter_label_lines(label_paths):
    for lp in label_paths:
        try:
            with open(lp) as f:
                for line in f:
                    yield line
        except OSError:
            continue


def keypoint_count_from_label_files(label_paths):
    """Etiket dosyalarından pose olup olmadığını ve keypoint sayısını (K) belirler."""
    for line in _iter_label_lines(label_paths):
        p = line.split()
        if len(p) == 5:
            return 0
        if len(p) > 5 and (len(p) - 5) % 3 == 0:
            return (len(p) - 5) // 3
    return 0


def keypoint_count_from_dir(labels_dir):
    """Bir etiket klasöründeki (düz) .txt dosyalarından K belirler (0 = detect/boş)."""
    if not labels_dir or not os.path.isdir(labels_dir):
        return 0
    paths = [os.path.join(labels_dir, f) for f in sorted(os.listdir(labels_dir))
             if f.lower().endswith(".txt")]
    return keypoint_count_from_label_files(paths)


def max_class_from_label_files(label_paths):
    """Etiketlerdeki en büyük sınıf indeksini bulur (-1 = etiket yok)."""
    max_id = -1
    for line in _iter_label_lines(label_paths):
        p = line.split()
        if p:
            try:
                max_id = max(max_id, int(float(p[0])))
            except ValueError:
                continue
    return max_id
