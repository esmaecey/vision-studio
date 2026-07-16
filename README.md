# Vision Studio

**YOLO tabanlı bilgisayarlı görü projeleri için uçtan uca masaüstü platformu.**

Vision Studio; veri etiketlemeden model eğitimine, veri artırmadan canlı teste kadar bir bilgisayarlı görü projesinin tüm iş akışını tek bir arayüzde toplayan genel amaçlı bir araçtır. PyQt5 ile geliştirilmiş masaüstü bir uygulamadır ve herhangi bir YOLO projesinde (nesne tespiti veya insan poz analizi) kullanılabilecek şekilde tasarlanmıştır.

Platform, gerçek bir senaryoda test edilmiştir: **İş Güvenliği (İSG) — Kişisel Koruyucu Donanım (KKD/PPE) tespiti**. Ancak bu yalnızca bir örnek uygulamadır; Vision Studio herhangi bir nesne tespiti problemine sınıf ve keypoint şablonları değiştirilerek uyarlanabilir.

> **Önemli ayrım:** Vision Studio bir **araçtır**. Aşağıda anlatılan İSG/KKD tespiti ise bu araçla **üretilmiş bir uygulamadır**. Araç, hiçbir sınıf veya senaryoya bağlı değildir.

---

## Özellikler

Uygulama, sol navigasyon panelinden erişilen 7 modülden oluşur:

| Modül | Açıklama |
|-------|----------|
| **Detect Labeling** | Bounding box (nesne kutusu) etiketleme modülü. Zoom, pan, sınıf bazlı renklendirme ve klavye kısayolları ile hızlı etiketleme sağlar; çıktıları YOLO `.txt` formatında kaydeder. |
| **Keypoint Labeling** | Nokta (keypoint) etiketleme modülü. İnsan pozu (COCO 17), el, dörtgen gibi hazır şablonların yanında kullanıcı tanımlı özelleştirilebilir nokta şablonlarını da destekler; YOLO-Pose formatında kaydeder. |
| **Data Augmentation** | `albumentations` kütüphanesi ile veri artırma. Flip, rotate, brightness/contrast, blur, gürültü gibi dönüşümleri toplu olarak uygular ve etiketleri dönüşümle birlikte günceller. |
| **Dataset Split** | Veri setini `train` / `val` / `test` olarak, ayarlanabilir oranlarla böler; görsel ve etiket dosyalarını eşleştirerek taşır. |
| **Training** | Ultralytics YOLOv8 ile **Detect** ve **Pose** model eğitimi. Tüm eğitim parametreleri arayüzden ayarlanır; canlı log akışı ve eğitim grafikleri (loss, mAP vb.) gösterilir. |
| **Testing** | Görsel, video ve webcam üzerinde canlı inference. Confidence eşiği ayarı, gerçek zamanlı FPS gösterimi ve anlık görüntü (snapshot) alma imkânı sunar. |
| **Dashboard (Home / Settings)** | Merkezi proje yönetimi, sistem/proje durumu özeti ve Light/Dark tema kontrolü. Aktif `data.yaml` buradan seçilir. |

---

## Genel Amaçlı Tasarım

Vision Studio'yu tek bir senaryoya değil, herhangi bir bilgisayarlı görü projesine uygun kılan temel tasarım kararları şunlardır:

- **Merkezi proje yönetimi.** Bir kez `data.yaml` seçilir (Settings modülü), ardından tüm modüller bu proje durumunu paylaşır. Sınıf listesi, keypoint şablonu ve veri seti kökü uygulama genelinde ortak bir durumdan (`config/project_state.py`) okunur. Bir modülde yapılan değişiklik dinleyici (listener) mekanizmasıyla diğer modüllere yansır.
- **Sınıflar koda gömülü değildir.** Nesne sınıfları ve keypoint şablonları uygulamanın içine sabitlenmemiştir; kullanıcının seçtiği `data.yaml` dosyasından gelir. `names` hem liste hem de sözlük formatında okunabilir.
- **Özelleştirilebilir keypoint şablonları.** İnsan pozu için varsayılan COCO 17 nokta şablonu hazır gelir, ancak `data.yaml` içindeki `kpt_names` / `kpt_shape` alanlarıyla ya da arayüzden farklı nokta sayısı ve iskelet tanımlanabilir.
- **Light / Dark tema sistemi.** Tüm arayüz iki tema arasında anlık geçişi destekler (`config/theme.py`).

Bu yaklaşım sayesinde platform yalnızca İSG projelerinde değil; trafik, tarım, perakende, tıbbi görüntüleme gibi **herhangi bir nesne tespiti veya poz analizi projesinde** yeniden kullanılabilir.

---

## Kurulum

**Gereksinim:** Python 3.8 veya üzeri.

```bash
# 1) Sanal ortam oluştur (Windows)
python -m venv venv

# 2) Ortamı aktive et (Windows)
venv\Scripts\activate

# 3) Bağımlılıkları yükle
pip install -r requirements.txt
```

> Not: Eğitim (Training) modülünde GPU kullanmak isterseniz, sisteminize uygun CUDA destekli PyTorch sürümünü ayrıca kurmanız önerilir.

---

## Kullanım

Uygulamayı başlatmak için:

```bash
python main.py
```

### Tipik iş akışı

1. **Proje seç** — `Settings` modülünden projenizin `data.yaml` dosyasını seçin. Tüm modüller bu andan itibaren aynı sınıf ve keypoint tanımlarını kullanır.
2. **Etiketle** — `Detect Labeling` veya `Keypoint Labeling` ile görselleri etiketleyin.
3. **Augment** — `Data Augmentation` ile veri setini çeşitlendirin.
4. **Split** — `Dataset Split` ile veriyi `train` / `val` / `test` olarak bölün.
5. **Train** — `Training` modülünde modeli eğitin, canlı log ve grafikleri izleyin.
6. **Test** — `Testing` modülünde eğitilen modeli görsel, video veya webcam üzerinde deneyin.

### GPU eğitimi (Colab)

Büyük veri setlerinde ve uzun süren eğitimlerde, yerel makine yerine **Google Colab** gibi ücretsiz GPU sağlayan bir ortamda eğitim yapmanız önerilir. Bu durumda: veri setini Vision Studio ile hazırlayın (etiketle → augment → split), ardından Colab'da GPU ile eğiterek ürettiğiniz `.pt` model dosyasını `Testing` modülüyle yerelde test edin.

---

## Örnek Uygulama — İSG / KKD Tespiti

Vision Studio ile geliştirilen örnek bir uygulama, iş sahalarında **Kişisel Koruyucu Donanım (KKD/PPE)** kullanımını denetlemeye yöneliktir. Model; baret, yelek, eldiven, koruyucu gözlük ve iş botu gibi ekipmanların takılı olup olmadığını tespit eder.

- **Sınıf sayısı:** 10 sınıf
- **Model:** YOLOv8s (Detect)
- **Başarım:** mAP50 ≈ 0.60

### Veri seti dengeleme deneyi

Geliştirme sırasında, veri setindeki **sınıf dengesizliğinin** yalnızca eğitim kalitesini değil, aynı zamanda **başarı ölçümünü** de bozduğu gözlemlendi. Az örnekli sınıflar hem model tarafından yeterince öğrenilemiyor hem de doğrulama setinde çok az örnekle temsil edildiğinden, ölçülen mAP değerleri yanıltıcı biçimde dalgalanıyordu. Sınıf başına örnek sayısını dengeleme yönünde yapılan düzenlemeler, hem eğitimin kararlılığını hem de metriklerin güvenilirliğini artırdı. Bu bulgu, Vision Studio'nun `Dataset Split` ve `Data Augmentation` modüllerinin dengeleme amacıyla birlikte kullanılmasının önemini ortaya koydu.

---

## Proje Yapısı

```
vision-studio/
├── main.py                       # Uygulama giriş noktası (navigasyon + modül yönlendirme)
├── requirements.txt              # Python bağımlılıkları
├── config/
│   ├── theme.py                  # Light / Dark tema tanımları ve stil üretimi
│   └── project_state.py          # Merkezi proje durumu (aktif data.yaml, sınıflar, keypoint şablonu)
└── gui/
    ├── dashboard/
    │   ├── home.py               # Ana ekran / genel durum
    │   └── settings.py           # Proje seçimi ve tema ayarları
    ├── detect_labeling/          # Bounding box etiketleme (ui.py, utils.py)
    ├── keypoint_labeling/        # Keypoint etiketleme (ui.py, utils.py)
    ├── augmentation/             # Veri artırma (ui.py, worker.py)
    ├── dataset_split/            # Veri bölme (ui.py, worker.py)
    ├── training/                 # Model eğitimi (ui.py, worker.py)
    └── testing/                  # Model testi / inference (ui.py, worker.py)
```

Her modül, arayüz (`ui.py`) ile ağır işlemleri arka planda yürüten iş parçacığını (`worker.py`) ayırarak arayüzün eğitim/işlem sırasında donmasını önler.

---

## Kullanılan Teknolojiler

- **Python** — Uygulama dili
- **PyQt5** — Masaüstü arayüz (GUI) çatısı
- **OpenCV** — Görüntü/video işleme, webcam yakalama
- **Ultralytics YOLOv8** — Nesne tespiti ve poz tahmini için model eğitimi ve inference
- **Albumentations** — Veri artırma dönüşümleri
- **PyYAML** — `data.yaml` proje yapılandırma dosyalarının okunması

---

## Geliştirici

**Esma Ece Yılmaz**
Staj Projesi — 2026

---

*Vision Studio genel amaçlı bir bilgisayarlı görü platformudur; İSG/KKD tespiti ise bu platformla üretilmiş örnek bir uygulamadır.*
