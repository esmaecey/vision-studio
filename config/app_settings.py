"""
Kalıcı kullanıcı ayarları — uygulama kapansa da korunması gereken tercihler.

Aktif proje (data.yaml yolu), tema, varsayılan model yolu ve çıktı klasörü gibi
ayarlar basit bir JSON dosyasında saklanır. PyQt bağımlılığı yoktur; hem
config.project_state hem de arayüz modülleri güvenle içe aktarabilir.
"""
import json
import os

# Ayar dosyası proje kökünde tutulur (config/ bir üst klasör).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(_PROJECT_ROOT, "user_settings.json")


def load_settings():
    """Ayar sözlüğünü döndürür; dosya yoksa ya da bozuksa boş sözlük."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(data):
    """Tüm ayar sözlüğünü diske yazar. Hata olursa sessizce yutar (kritik değil)."""
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def save_setting(key, value):
    """Tek bir ayarı günceller (diğerlerini koruyarak)."""
    data = load_settings()
    if value is None:
        data.pop(key, None)
    else:
        data[key] = value
    save_settings(data)


def get_setting(key, default=None):
    return load_settings().get(key, default)
