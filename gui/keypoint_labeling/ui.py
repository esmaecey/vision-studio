"""
Keypoint Etiketleme Arayüzü — insan iskeleti (17 nokta) annotation.
Nokta yerleştirme/taşıma/silme, boyut ayarı, görünürlük, iskelet, images+labels+vis çıktısı.
"""
import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem,
    QListWidget, QPushButton, QLabel, QFileDialog, QSlider, QMessageBox,
    QInputDialog, QDialog, QTextEdit, QDialogButtonBox, QComboBox, QLineEdit
)

from PyQt5.QtGui import QPixmap, QPen, QColor, QBrush, QPainter
from PyQt5.QtCore import Qt, QRectF

from .utils import save_pose_label, load_pose_label, draw_pose_on_image, save_image_unicode
from config.project_state import project_state


class KeypointItem(QGraphicsEllipseItem):
    """Taşınabilir keypoint noktası."""
    def __init__(self, x, y, radius, index, canvas):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setPos(x, y)
        self.kp_index = index
        self.canvas = canvas
        self.visibility = 2
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.update_color()

    def update_color(self):
        if self.visibility == 2:
            self.setBrush(QBrush(QColor(255, 0, 0)))       # görünür - kırmızı
        elif self.visibility == 1:
            self.setBrush(QBrush(QColor(255, 165, 0)))     # örtük - turuncu
        else:
            self.setBrush(QBrush(QColor(120, 120, 120)))   # yok - gri
        self.setPen(QPen(QColor(255, 255, 255), 1))

    def set_radius(self, radius):
        self.setRect(-radius, -radius, 2 * radius, 2 * radius)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            self.canvas.sync_keypoint(self.kp_index, self.scenePos().x(), self.scenePos().y())
            self.canvas._redraw_skeleton()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        # Çift tık: görünürlük döngüsü 2 -> 1 -> 0 -> 2
        self.visibility = {2: 1, 1: 0, 0: 2}[self.visibility]
        self.canvas.sync_visibility(self.kp_index, self.visibility)
        self.update_color()
        self.canvas._redraw_skeleton()
        super().mouseDoubleClickEvent(event)


class KeypointCanvas(QGraphicsView):
    def __init__(self, status_callback=None):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.pixmap_item = None
        self.keypoints = []       # [(x, y, v), ...]
        self.point_items = []
        self.line_items = []
        self.current_kp_index = 0
        self.point_radius = 5
        self.status_callback = status_callback

        self._panning = False
        self._pan_start = None

    def load_image(self, image_path):
        self.scene.clear()
        self.keypoints = []
        self.point_items = []
        self.line_items = []
        self.current_kp_index = 0

        pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._update_status()
        return pixmap.width(), pixmap.height()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton and self.pixmap_item is not None:
            # Var olan bir noktanın üstüne mi tıkladık? (taşıma için)
            item_at = self.itemAt(event.pos())
            if isinstance(item_at, KeypointItem):
                super().mousePressEvent(event)  # taşımayı Qt'ye bırak
                return
            # Yeni nokta ekle (henüz 17 tamamlanmadıysa)
            if self.current_kp_index < project_state.num_keypoints():
                scene_pos = self.mapToScene(event.pos())
                self._add_keypoint(scene_pos.x(), scene_pos.y())
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        super().mouseReleaseEvent(event)

    def _add_keypoint(self, x, y):
        self.keypoints.append((x, y, 2))
        item = KeypointItem(x, y, self.point_radius, self.current_kp_index, self)
        self.scene.addItem(item)
        self.point_items.append(item)
        self.current_kp_index += 1
        self._redraw_skeleton()
        self._update_status()

    def sync_keypoint(self, index, x, y):
        if index < len(self.keypoints):
            _, _, v = self.keypoints[index]
            self.keypoints[index] = (x, y, v)

    def sync_visibility(self, index, v):
        if index < len(self.keypoints):
            x, y, _ = self.keypoints[index]
            self.keypoints[index] = (x, y, v)

    def _redraw_skeleton(self):
        for line in self.line_items:
            self.scene.removeItem(line)
        self.line_items = []
        for (a, b) in project_state.skeleton:
            if a < len(self.keypoints) and b < len(self.keypoints):
                xa, ya, va = self.keypoints[a]
                xb, yb, vb = self.keypoints[b]
                if va > 0 and vb > 0:
                    line = QGraphicsLineItem(xa, ya, xb, yb)
                    line.setPen(QPen(QColor(0, 200, 255), 2))
                    line.setZValue(-1)
                    self.scene.addItem(line)
                    self.line_items.append(line)

    def set_point_radius(self, radius):
        self.point_radius = radius
        for item in self.point_items:
            item.set_radius(radius)

    def undo_last_point(self):
        if self.keypoints:
            self.keypoints.pop()
            item = self.point_items.pop()
            self.scene.removeItem(item)
            self.current_kp_index -= 1
            self._redraw_skeleton()
            self._update_status()

    def clear_points(self):
        for item in self.point_items:
            self.scene.removeItem(item)
        for line in self.line_items:
            self.scene.removeItem(line)
        self.keypoints = []
        self.point_items = []
        self.line_items = []
        self.current_kp_index = 0
        self._update_status()

    def load_keypoints(self, keypoints):
        """Kayıtlı keypoint'leri geri yükler."""
        self.clear_points()
        for (x, y, v) in keypoints:
            if v > 0:
                self.keypoints.append((x, y, v))
                item = KeypointItem(x, y, self.point_radius, self.current_kp_index, self)
                item.visibility = v
                item.update_color()
                self.scene.addItem(item)
                self.point_items.append(item)
                self.current_kp_index += 1
        self._redraw_skeleton()
        self._update_status()

    def _update_status(self):
        if self.status_callback:
            n = project_state.num_keypoints()
            if self.current_kp_index < n:
                next_name = project_state.keypoint_names[self.current_kp_index]
                self.status_callback(f"Sıradaki: {self.current_kp_index}: {next_name}", self.current_kp_index)
            else:
                self.status_callback(f"Tüm noktalar yerleştirildi ({n}/{n})", n)
    def get_keypoints(self):
        return self.keypoints


class KeypointLabelingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_folder = None
        self.output_folder = None
        self.image_files = []
        self.current_index = -1
        self.processed_files = set()
        self._build_ui()
        project_state.add_listener(self._on_project_changed)
        self._populate_kp_list()

    def _build_ui(self):
        main_layout = QHBoxLayout()

        # Sol panel
        left_panel = QVBoxLayout()
        self.btn_open = QPushButton("Resim Klasörü Aç")
        self.btn_open.clicked.connect(self.open_folder)
        left_panel.addWidget(self.btn_open)
        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        left_panel.addWidget(self.file_list)
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(250)

        # Orta panel
        center_layout = QVBoxLayout()
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Önceki")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Sonraki >")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_undo = QPushButton("Geri Al")
        self.btn_undo.clicked.connect(lambda: self.canvas.undo_last_point())
        self.btn_clear = QPushButton("Temizle")
        self.btn_clear.clicked.connect(lambda: self.canvas.clear_points())
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_undo)
        nav_layout.addWidget(self.btn_clear)
        center_layout.addLayout(nav_layout)

        # Nokta boyutu ayarı
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Nokta Boyutu:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(2, 20)
        self.size_slider.setValue(5)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        size_layout.addWidget(self.size_slider)
        self.size_value_label = QLabel("5")
        size_layout.addWidget(self.size_value_label)
        center_layout.addLayout(size_layout)

        self.canvas = KeypointCanvas(status_callback=self._set_status)
        center_layout.addWidget(self.canvas)

        self.status_label = QLabel("Bir klasör açarak başlayın.")
        center_layout.addWidget(self.status_label)
        center_widget = QWidget()
        center_widget.setLayout(center_layout)

        # Sağ panel
        right_panel = QVBoxLayout()

        self.project_label = QLabel(project_state.summary())
        self.project_label.setWordWrap(True)
        self.project_label.setStyleSheet("font-size: 11px; padding: 4px;")
        right_panel.addWidget(self.project_label)

        btn_load_yaml = QPushButton("data.yaml Yükle")
        btn_load_yaml.clicked.connect(self.load_data_yaml)
        right_panel.addWidget(btn_load_yaml)

        btn_edit_template = QPushButton("Şablon Düzenle")
        btn_edit_template.clicked.connect(self.edit_keypoint_template)
        right_panel.addWidget(btn_edit_template)

        right_panel.addWidget(QLabel("Keypoint Sırası"))
        self.kp_list = QListWidget()
        right_panel.addWidget(self.kp_list)

        info = QLabel("Çift tık: görünürlük\n(kırmızı→turuncu→gri)\nSürükle: taşı")
        info.setStyleSheet("font-size: 11px;")
        right_panel.addWidget(info)
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(200)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(center_widget, stretch=1)
        main_layout.addWidget(right_widget)
        self.setLayout(main_layout)

    def on_size_changed(self, value):
        self.size_value_label.setText(str(value))
        self.canvas.set_point_radius(value)

    def _set_status(self, text, kp_index):
        self.status_label.setText(text)
        if kp_index < self.kp_list.count():
            self.kp_list.setCurrentRow(kp_index)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Resim Klasörünü Seç")
        if not folder:
            return
        self.image_folder = folder

        parent_dir = os.path.dirname(folder)
        self.output_folder = os.path.join(parent_dir, "keypoint_çıktı")
        os.makedirs(os.path.join(self.output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "vis"), exist_ok=True)

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(valid_ext))
        self.processed_files = set()
        self.file_list.clear()
        self.file_list.addItems(self.image_files)
        self.current_index = -1
        if self.image_files:
            self.file_list.setCurrentRow(0)

    def on_file_selected(self, row):
        if row < 0 or row >= len(self.image_files):
            return
        if self.current_index != -1 and self.current_index != row:
            self.save_current()

        self.current_index = row
        image_path = os.path.join(self.image_folder, self.image_files[row])
        img_w, img_h = self.canvas.load_image(image_path)

        # Kayıtlı etiket varsa geri yükle
        base_name = os.path.splitext(self.image_files[row])[0]
        out_label = os.path.join(self.output_folder, "labels", base_name + ".txt")
        if os.path.exists(out_label):
            persons = load_pose_label(out_label, img_w, img_h)
            if persons:
                self.canvas.load_keypoints(persons[0]['keypoints'])

    def save_current(self):
        if self.current_index == -1 or self.canvas.pixmap_item is None:
            return
        if self.output_folder is None:
            return

        keypoints = list(self.canvas.get_keypoints())
        if not keypoints:
            return  # hiç nokta yoksa kaydetme

        while len(keypoints) < project_state.num_keypoints():
            keypoints.append((0, 0, 0))

        img_w = self.canvas.pixmap_item.pixmap().width()
        img_h = self.canvas.pixmap_item.pixmap().height()

        valid_pts = [(x, y) for (x, y, v) in keypoints if v > 0]
        if valid_pts:
            xs = [p[0] for p in valid_pts]
            ys = [p[1] for p in valid_pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            bbox = (0, 0, img_w, img_h)

        person = {'bbox': bbox, 'keypoints': keypoints}
        image_filename = self.image_files[self.current_index]
        base_name = os.path.splitext(image_filename)[0]
        source_image_path = os.path.join(self.image_folder, image_filename)

        # 1) labels
        label_out = os.path.join(self.output_folder, "labels", base_name + ".txt")
        save_pose_label(label_out, [person], img_w, img_h)

        # 2) images (orijinal kopya)
        image_out = os.path.join(self.output_folder, "images", image_filename)
        shutil.copy(source_image_path, image_out)

        # 3) vis (iskelet çizili)
        vis_img = draw_pose_on_image(source_image_path, [person], project_state.skeleton)
        if vis_img is not None:
            vis_out = os.path.join(self.output_folder, "vis", base_name + ".jpg")
            save_image_unicode(vis_out, vis_img)

        self.processed_files.add(image_filename)
        item = self.file_list.item(self.current_index)
        if item:
            item.setForeground(QBrush(QColor(0, 170, 0)))
        self.status_label.setText(f"Kaydedildi: {base_name}")

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.file_list.setCurrentRow(self.current_index + 1)

    def prev_image(self):
        if self.current_index > 0:
            self.file_list.setCurrentRow(self.current_index - 1)

    def _populate_kp_list(self):
        self.kp_list.clear()
        for i, name in enumerate(project_state.keypoint_names):
            self.kp_list.addItem(f"{i}: {name}")

    def _on_project_changed(self):
        self._populate_kp_list()
        if hasattr(self, "project_label"):
            self.project_label.setText(project_state.summary())

    def load_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "data.yaml Seç", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            project_state.load_data_yaml(path)
            QMessageBox.information(
                self, "Başarılı",
                f"Proje yüklendi.\nKeypoint sayısı: {project_state.num_keypoints()}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"data.yaml okunamadı:\n{str(e)}")

    def edit_keypoint_template(self):
        """Kullanıcının kendi keypoint şablonunu tanımlamasını sağlar."""
        dlg = KeypointTemplateDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            names, skeleton = dlg.get_template()
            if names:
                project_state.keypoint_names = list(names)
                project_state.skeleton = list(skeleton)
                
                # Listeyi doğrudan yenile
                self._populate_kp_list()
                
                # Canvas'taki mevcut noktaları temizle (şablon değişti)
                self.canvas.clear_points()
                
                QMessageBox.information(
                    self, "Şablon Güncellendi",
                    f"{len(names)} nokta, {len(skeleton)} bağlantı tanımlandı."
                )

# ---------- Hazır Şablonlar ----------
PRESETS = {
    "İnsan Pozu (COCO 17)": {
        "names": [
            "burun", "sol göz", "sağ göz", "sol kulak", "sağ kulak",
            "sol omuz", "sağ omuz", "sol dirsek", "sağ dirsek",
            "sol bilek", "sağ bilek", "sol kalça", "sağ kalça",
            "sol diz", "sağ diz", "sol ayak bileği", "sağ ayak bileği"
        ],
        "skeleton": [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)
        ]
    },
    "El (21 nokta)": {
        "names": [
            "bilek",
            "başparmak_1", "başparmak_2", "başparmak_3", "başparmak_ucu",
            "işaret_1", "işaret_2", "işaret_3", "işaret_ucu",
            "orta_1", "orta_2", "orta_3", "orta_ucu",
            "yüzük_1", "yüzük_2", "yüzük_3", "yüzük_ucu",
            "serçe_1", "serçe_2", "serçe_3", "serçe_ucu"
        ],
        "skeleton": [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
    },
    "Dörtgen (4 köşe)": {
        "names": ["sol üst", "sağ üst", "sağ alt", "sol alt"],
        "skeleton": [(0, 1), (1, 2), (2, 3), (3, 0)]
    },
    "Boş (sıfırdan tanımla)": {
        "names": [],
        "skeleton": []
    },
}


class KeypointTemplateDialog(QDialog):
    """Keypoint şablonu tanımlama — hazır preset veya özel şablon."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keypoint Şablonu")
        self.resize(700, 600)

        self.names = list(project_state.keypoint_names)
        self.skeleton = list(project_state.skeleton)

        layout = QVBoxLayout()
        layout.setSpacing(10)

        # --- Açıklama ---
        info = QLabel(
            "Keypoint şablonu, etiketlerken hangi noktaları hangi sırayla koyacağınızı "
            "ve hangi noktaların birbirine çizgiyle bağlanacağını belirler."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 8px; font-size: 12px;")
        layout.addWidget(info)

        # --- Hazır şablon seçimi ---
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Hazır şablon:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        preset_row.addWidget(self.preset_combo, stretch=1)
        btn_apply_preset = QPushButton("Uygula")
        btn_apply_preset.clicked.connect(self.apply_preset)
        preset_row.addWidget(btn_apply_preset)
        layout.addLayout(preset_row)

        # --- İki panel: noktalar | bağlantılar ---
        panels = QHBoxLayout()

        # Sol: nokta listesi
        left = QVBoxLayout()
        left.addWidget(QLabel("Noktalar (sırayla yerleştirilecek)"))
        self.point_list = QListWidget()
        left.addWidget(self.point_list)

        add_row = QHBoxLayout()
        self.new_point_input = QLineEdit()
        self.new_point_input.setPlaceholderText("Yeni nokta adı...")
        self.new_point_input.returnPressed.connect(self.add_point)
        add_row.addWidget(self.new_point_input)
        btn_add_point = QPushButton("+")
        btn_add_point.setMaximumWidth(40)
        btn_add_point.clicked.connect(self.add_point)
        add_row.addWidget(btn_add_point)
        left.addLayout(add_row)

        btn_remove_point = QPushButton("Seçili Noktayı Sil")
        btn_remove_point.clicked.connect(self.remove_point)
        left.addWidget(btn_remove_point)

        panels.addLayout(left)

        # Sağ: bağlantı listesi
        right = QVBoxLayout()
        right.addWidget(QLabel("Bağlantılar (çizgiyle bağlanacak nokta çiftleri)"))
        self.link_list = QListWidget()
        right.addWidget(self.link_list)

        link_row = QHBoxLayout()
        self.from_combo = QComboBox()
        self.to_combo = QComboBox()
        link_row.addWidget(self.from_combo)
        link_row.addWidget(QLabel("→"))
        link_row.addWidget(self.to_combo)
        btn_add_link = QPushButton("Bağla")
        btn_add_link.clicked.connect(self.add_link)
        link_row.addWidget(btn_add_link)
        right.addLayout(link_row)

        btn_remove_link = QPushButton("Seçili Bağlantıyı Sil")
        btn_remove_link.clicked.connect(self.remove_link)
        right.addWidget(btn_remove_link)

        panels.addLayout(right)
        layout.addLayout(panels)

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)
        self._refresh_all()

    # ---------- Yardımcılar ----------
    def _refresh_all(self):
        self._refresh_points()
        self._refresh_links()
        self._refresh_combos()

    def _refresh_points(self):
        self.point_list.clear()
        for i, name in enumerate(self.names):
            self.point_list.addItem(f"{i}: {name}")

    def _refresh_links(self):
        self.link_list.clear()
        for (a, b) in self.skeleton:
            name_a = self.names[a] if a < len(self.names) else f"?{a}"
            name_b = self.names[b] if b < len(self.names) else f"?{b}"
            self.link_list.addItem(f"{name_a} → {name_b}")

    def _refresh_combos(self):
        self.from_combo.clear()
        self.to_combo.clear()
        for i, name in enumerate(self.names):
            self.from_combo.addItem(f"{i}: {name}")
            self.to_combo.addItem(f"{i}: {name}")

    # ---------- Eylemler ----------
    def apply_preset(self):
        preset_name = self.preset_combo.currentText()
        preset = PRESETS.get(preset_name)
        if preset:
            self.names = list(preset["names"])
            self.skeleton = list(preset["skeleton"])
            self._refresh_all()

    def add_point(self):
        name = self.new_point_input.text().strip()
        if name:
            self.names.append(name)
            self.new_point_input.clear()
            self._refresh_all()

    def remove_point(self):
        row = self.point_list.currentRow()
        if row < 0 or row >= len(self.names):
            return
        self.names.pop(row)
        # Bu noktaya bağlı tüm bağlantıları sil, sonrakilerin index'ini kaydır
        new_skeleton = []
        for (a, b) in self.skeleton:
            if a == row or b == row:
                continue
            new_a = a - 1 if a > row else a
            new_b = b - 1 if b > row else b
            new_skeleton.append((new_a, new_b))
        self.skeleton = new_skeleton
        self._refresh_all()

    def add_link(self):
        a = self.from_combo.currentIndex()
        b = self.to_combo.currentIndex()
        if a < 0 or b < 0 or a == b:
            return
        if (a, b) in self.skeleton or (b, a) in self.skeleton:
            return
        self.skeleton.append((a, b))
        self._refresh_links()

    def remove_link(self):
        row = self.link_list.currentRow()
        if 0 <= row < len(self.skeleton):
            self.skeleton.pop(row)
            self._refresh_links()

    def get_template(self):
        return self.names, self.skeleton