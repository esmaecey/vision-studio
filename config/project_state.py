"""
Vision Studio — merkezi proje durumu.
Aktif data.yaml, sınıf listesi ve keypoint şablonunu tüm modüller için tutar.
"""
import os
import yaml


# ---------- Varsayılan COCO 17 keypoint şablonu ----------
DEFAULT_KEYPOINT_NAMES = [
    "burun", "sol göz", "sağ göz", "sol kulak", "sağ kulak",
    "sol omuz", "sağ omuz", "sol dirsek", "sağ dirsek",
    "sol bilek", "sağ bilek", "sol kalça", "sağ kalça",
    "sol diz", "sağ diz", "sol ayak bileği", "sağ ayak bileği"
]

DEFAULT_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)
]


class ProjectState:
    """Uygulama genelinde paylaşılan proje durumu (singleton gibi kullanılır)."""

    def __init__(self):
        self.data_yaml_path = None
        self.class_names = []
        self.dataset_root = None
        self.default_model_path = None
        # Eğitim çıktılarının kaydedileceği sabit klasör (kullanıcı Settings'ten
        # seçebilir). None ise data.yaml'ın bulunduğu klasör kullanılır.
        self.output_dir = None

        # Keypoint ayarları (data.yaml'da tanımlıysa oradan, yoksa varsayılan)
        self.keypoint_names = list(DEFAULT_KEYPOINT_NAMES)
        self.skeleton = list(DEFAULT_SKELETON)
        # Yatay çevirmede sol/sağ keypoint takası için indeks eşlemesi
        self.flip_idx = []

        # Değişiklikleri dinleyecek callback'ler
        self._listeners = []

    # ---------- Dinleyici mekanizması ----------
    def add_listener(self, callback):
        """Proje değiştiğinde çağrılacak fonksiyon kaydeder."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def _notify(self):
        for cb in self._listeners:
            try:
                cb()
            except Exception as e:
                print(f"Listener hatası: {e}")

    # ---------- data.yaml yükleme ----------
    def load_data_yaml(self, path):
        """data.yaml dosyasını okuyup sınıfları ve keypoint ayarlarını günceller."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"data.yaml bulunamadı: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("data.yaml geçerli bir YAML sözlüğü değil.")

        # Sınıf isimleri: liste veya sözlük olabilir
        names = data.get("names", [])
        if isinstance(names, dict):
            # {0: 'boots', 1: 'gloves'} formatı
            self.class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
        elif isinstance(names, list):
            self.class_names = list(names)
        else:
            self.class_names = []

        self.data_yaml_path = path
        self.dataset_root = os.path.dirname(path)

        # Keypoint şablonu (opsiyonel — data.yaml'da varsa kullan)
        kp_names = data.get("kpt_names")
        if isinstance(kp_names, list) and kp_names:
            self.keypoint_names = list(kp_names)
        else:
            # kpt_shape varsa nokta sayısını ondan al
            kpt_shape = data.get("kpt_shape")
            if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
                n = int(kpt_shape[0])
                if n != len(DEFAULT_KEYPOINT_NAMES):
                    self.keypoint_names = [f"nokta {i}" for i in range(n)]
                else:
                    self.keypoint_names = list(DEFAULT_KEYPOINT_NAMES)
            else:
                self.keypoint_names = list(DEFAULT_KEYPOINT_NAMES)

        skeleton = data.get("skeleton")
        if isinstance(skeleton, list) and skeleton:
            try:
                self.skeleton = [tuple(pair) for pair in skeleton]
            except Exception:
                self.skeleton = list(DEFAULT_SKELETON)
        else:
            # Varsayılan iskelet sadece 17 nokta ise anlamlı
            if len(self.keypoint_names) == len(DEFAULT_KEYPOINT_NAMES):
                self.skeleton = list(DEFAULT_SKELETON)
            else:
                self.skeleton = []

        # flip_idx: yatay çevirmede keypoint indeks takası (uzunluk keypoint sayısıyla eşleşmeli)
        flip_idx = data.get("flip_idx")
        if isinstance(flip_idx, list) and len(flip_idx) == len(self.keypoint_names):
            try:
                self.flip_idx = [int(x) for x in flip_idx]
            except (ValueError, TypeError):
                self.flip_idx = []
        else:
            self.flip_idx = []

        self._notify()
        return self.class_names

    # ---------- Yardımcılar ----------
    def has_project(self):
        return self.data_yaml_path is not None and len(self.class_names) > 0

    def project_name(self):
        if not self.data_yaml_path:
            return "Proje seçilmedi"
        return os.path.basename(os.path.dirname(self.data_yaml_path))

    def summary(self):
        if not self.has_project():
            return "Aktif proje yok — Settings'ten bir data.yaml seçin"
        return f"{self.project_name()} · {len(self.class_names)} sınıf"

    def set_output_dir(self, path):
        """Eğitim çıktıları için sabit klasör belirler (boş verilirse varsayılana döner)."""
        self.output_dir = path or None
        self._notify()

    def training_base_dir(self, fallback_data_yaml=None):
        """
        Eğitim çıktılarının kök klasörünü döner.
        Öncelik: Settings'te seçilen output_dir → aktif data.yaml klasörü →
        parametreyle verilen data.yaml klasörü.
        """
        if self.output_dir:
            return self.output_dir
        yaml_path = self.data_yaml_path or fallback_data_yaml
        if yaml_path:
            return os.path.dirname(os.path.abspath(yaml_path))
        return None

    def add_class(self, name):
        """Kullanıcının arayüzden yeni sınıf eklemesi (geçici, dosyaya yazmaz)."""
        if name and name not in self.class_names:
            self.class_names.append(name)
            self._notify()

    def num_keypoints(self):
        return len(self.keypoint_names)


# Uygulama genelinde tek bir örnek kullanılır
project_state = ProjectState()