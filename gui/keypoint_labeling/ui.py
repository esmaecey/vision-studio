"""
Keypoint Etiketleme Arayüzü — insan iskeleti (17 nokta) annotation.
Nokta yerleştirme/taşıma/silme, boyut ayarı, görünürlük, iskelet, images+labels+vis çıktısı.
"""
import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem,
    QGraphicsSimpleTextItem, QListWidget, QPushButton, QLabel, QFileDialog,
    QSlider, QMessageBox, QInputDialog, QDialog, QTextEdit, QDialogButtonBox,
    QComboBox, QLineEdit, QShortcut
)

from PyQt5.QtGui import QPixmap, QPen, QColor, QBrush, QPainter, QKeySequence, QFont
from PyQt5.QtCore import Qt, QRectF

from .utils import (
    save_pose_label, load_pose_label, draw_pose_on_image, save_image_unicode,
    keypoint_color, skeleton_edge_color,
)
from config.project_state import project_state
from utils.dataset_paths import default_output_dir, derive_labels_dir
from gui.common.help import show_help


class KeypointItem(QGraphicsEllipseItem):
    """Seçilebilir, taşınabilir keypoint noktası (belirli bir keypoint index'ine ait)."""

    def __init__(self, x, y, radius, index, canvas, visibility=2):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setPos(x, y)
        self.kp_index = index
        self.canvas = canvas
        self.visibility = visibility
        self._radius = radius
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        # Index/isim etiketi (açılır-kapanır). Zoom'dan etkilenmesin diye sabit boyda.
        self.label_item = QGraphicsSimpleTextItem(str(index), self)
        self.label_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        _font = QFont()
        _font.setPointSize(8)
        _font.setBold(True)
        self.label_item.setFont(_font)
        self.label_item.setBrush(QBrush(QColor(255, 255, 255)))
        self.label_item.setPen(QPen(QColor(0, 0, 0), 0.7))  # okunurluk için ince dış çizgi
        self.label_item.setZValue(2)
        self._position_label()
        self.label_item.setVisible(bool(getattr(canvas, "show_labels", False)))

        self.update_color()

    def _base_color(self):
        total = self.canvas.num_kpts() if self.canvas is not None else 0
        r, g, b = keypoint_color(self.kp_index, total)
        return QColor(r, g, b)

    def update_color(self):
        color = self._base_color()
        if self.visibility == 2:
            self.setBrush(QBrush(color))                       # görünür: dolu, kendi rengi
        elif self.visibility == 1:
            faded = QColor(color)
            faded.setAlpha(110)
            self.setBrush(QBrush(faded))                       # örtük: yarı saydam
        else:
            self.setBrush(QBrush(QColor(110, 110, 110)))       # yok: gri
        if self.isSelected():
            # Seçili nokta: kalın beyaz halka ile öne çıksın (rengini korur)
            self.setPen(QPen(QColor(255, 255, 255), 3))
        else:
            # Kendi rengiyle uyumlu koyu ince kenar (arka planda kontrast)
            self.setPen(QPen(QColor(20, 20, 20), 1))

    def _position_label(self):
        # Etiketi noktanın sağ-üstüne yerleştir
        self.label_item.setPos(self._radius + 2, -self._radius - 12)

    def set_label_text(self, text):
        self.label_item.setText(text)

    def set_label_visible(self, visible):
        self.label_item.setVisible(visible)

    def set_radius(self, radius):
        self._radius = radius
        self.setRect(-radius, -radius, 2 * radius, 2 * radius)
        self._position_label()

    def set_visibility(self, v):
        self.visibility = v
        self.update_color()
        if self.canvas is not None:
            self.canvas._redraw_skeleton()

    def cycle_visibility(self):
        # 2 (görünür) -> 1 (örtük) -> 0 (yok) -> 2
        self.set_visibility({2: 1, 1: 0, 0: 2}[self.visibility])

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            if self.canvas is not None:
                self.canvas._redraw_skeleton()
        elif change == QGraphicsItem.ItemSelectedHasChanged:
            self.update_color()
            if self.canvas is not None and bool(value):
                self.canvas._on_item_selected(self)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        # Taşıma öncesi durumu kaydet (undo için)
        if self.canvas is not None:
            self.canvas.begin_interaction()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.canvas is not None:
            self.canvas.end_interaction()

    def mouseDoubleClickEvent(self, event):
        # Çift tık: görünürlük döngüsü (undo'lu)
        if self.canvas is not None:
            self.canvas.push_undo()
        self.cycle_visibility()
        if self.canvas is not None:
            self.canvas.mark_dirty()
        super().mouseDoubleClickEvent(event)


class KeypointCanvas(QGraphicsView):
    """
    Keypoint gösterimi/düzenleme canvas'ı. Noktalar keypoint INDEX'ine göre
    saklanır (point_items: {index: KeypointItem}); v=0 (yerleştirilmemiş) slotlar
    boş kalır. Detect ile aynı olgunluk: seçme, taşıma, silme, görünürlük,
    undo/redo, klavye kısayolları.
    """

    def __init__(self, status_callback=None):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFocusPolicy(Qt.StrongFocus)

        self.pixmap_item = None
        self.point_items = {}     # {index: KeypointItem}
        self.line_items = []
        self.active_index = 0     # sıradaki/aktif keypoint index'i
        self.point_radius = 5
        self.show_labels = False  # nokta üzerindeki index/isim etiketleri (L ile aç/kapa)
        self.status_callback = status_callback

        # Widget'in bağladığı geri çağrılar (klavye kısayolları için)
        self.on_prev = None
        self.on_next = None
        self.on_save = None
        self.on_labels_toggled = None  # L kısayolu butonu senkronlasın

        # Geri alma/ileri alma geçmişi ve değişiklik bayrağı
        self._undo = []
        self._redo = []
        self._pending = None
        self.dirty = False

        self._panning = False
        self._pan_start = None

    # ---------- Temel ----------
    def num_kpts(self):
        return project_state.num_keypoints()

    def has_points(self):
        return len(self.point_items) > 0

    def load_image(self, image_path):
        self.scene.clear()
        self.point_items = {}
        self.line_items = []
        self.active_index = 0

        pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        # Görsel en arkada kalsın: iskelet çizgileri (z=0) ve noktalar (z=1) üstte.
        self.pixmap_item.setZValue(-10)
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.reset_history()
        self._update_status()
        return pixmap.width(), pixmap.height()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def reset_zoom(self):
        if self.pixmap_item is not None:
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_labels_visible(self, visible):
        self.show_labels = bool(visible)
        for it in self.point_items.values():
            it.set_label_visible(self.show_labels)
        if callable(self.on_labels_toggled):
            self.on_labels_toggled(self.show_labels)

    def toggle_labels(self):
        self.set_labels_visible(not self.show_labels)
        self._status_msg("Etiketler açık (L)" if self.show_labels else "Etiketler kapalı (L)")
        return self.show_labels

    # ---------- Fare ----------
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton and self.pixmap_item is not None:
            item_at = self.itemAt(event.pos())
            if isinstance(item_at, KeypointItem):
                super().mousePressEvent(event)  # seçme/taşımayı Qt'ye bırak
                return
            # Boş alan: aktif keypoint'i buraya yerleştir
            scene_pos = self.mapToScene(event.pos())
            self.place_point(scene_pos.x(), scene_pos.y())
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

    # ---------- Klavye ----------
    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)

        if ctrl and key == Qt.Key_Z and not shift:
            self.undo(); return
        if (ctrl and key == Qt.Key_Y) or (ctrl and shift and key == Qt.Key_Z):
            self.redo(); return
        if ctrl and key == Qt.Key_S:
            if callable(self.on_save):
                self.on_save()
            return
        if key in (Qt.Key_Delete, Qt.Key_X):
            self.delete_selected(); return
        if key == Qt.Key_V:
            self.toggle_visibility_selected(); return
        if key == Qt.Key_L:
            self.toggle_labels(); return
        if key == Qt.Key_R:
            self.reset_zoom(); return
        if key == Qt.Key_Left:
            if callable(self.on_prev):
                self.on_prev()
            return
        if key == Qt.Key_Right:
            if callable(self.on_next):
                self.on_next()
            return
        if Qt.Key_0 <= key <= Qt.Key_9:
            self.set_active_index(key - Qt.Key_0)
            return
        super().keyPressEvent(event)

    # ---------- Nokta yerleştirme / index yönetimi ----------
    def _first_free_index(self):
        for i in range(self.num_kpts()):
            if i not in self.point_items:
                return i
        return None

    def _advance_active(self):
        free = self._first_free_index()
        self.active_index = free if free is not None else self.num_kpts()

    def set_active_index(self, i):
        if 0 <= i < self.num_kpts():
            self.active_index = i
            self._update_status()

    def _create_item(self, index, x, y, v):
        item = KeypointItem(x, y, self.point_radius, index, self, v)
        self.scene.addItem(item)
        self.point_items[index] = item
        return item

    def place_point(self, x, y):
        num = self.num_kpts()
        if num <= 0:
            self._status_msg("Önce bir keypoint şablonu / data.yaml yükleyin.")
            return
        idx = self.active_index
        if idx >= num or idx in self.point_items:
            idx = self._first_free_index()
        if idx is None:
            self._status_msg("Tüm noktalar yerleştirildi. Taşımak için sürükleyin.")
            return
        self.push_undo()
        self._create_item(idx, x, y, 2)
        self.mark_dirty()
        self._advance_active()
        self._redraw_skeleton()
        self._update_status()

    def sync_keypoint(self, index, x, y):
        # Geriye dönük uyumluluk için korunur (artık item'lar kaynak durumdur)
        pass

    def sync_visibility(self, index, v):
        if index in self.point_items:
            self.point_items[index].set_visibility(v)

    # ---------- Seçim / silme / görünürlük ----------
    def selected_items(self):
        return [it for it in self.scene.selectedItems() if isinstance(it, KeypointItem)]

    def delete_selected(self):
        sel = self.selected_items()
        if not sel:
            return False
        self.push_undo()
        for it in sel:
            self.point_items.pop(it.kp_index, None)
            self.scene.removeItem(it)
        self.mark_dirty()
        self._advance_active()
        self._redraw_skeleton()
        self._update_status()
        self._status_msg(f"{len(sel)} nokta silindi.")
        return True

    def toggle_visibility_selected(self):
        sel = self.selected_items()
        if not sel:
            self._status_msg("Görünürlük için önce bir nokta seçin (sol tık).")
            return
        self.push_undo()
        for it in sel:
            it.cycle_visibility()
        self.mark_dirty()

    def _on_item_selected(self, item):
        # Bir noktaya tıklanınca o keypoint aktif index olsun (sağ liste vurgulanır)
        self.active_index = item.kp_index
        self._update_status()

    # ---------- İskelet ----------
    def _redraw_skeleton(self):
        for line in self.line_items:
            self.scene.removeItem(line)
        self.line_items = []
        total = self.num_kpts()
        for (a, b) in project_state.skeleton:
            ia, ib = self.point_items.get(a), self.point_items.get(b)
            if ia is not None and ib is not None and ia.visibility > 0 and ib.visibility > 0:
                pa, pb = ia.scenePos(), ib.scenePos()
                line = QGraphicsLineItem(pa.x(), pa.y(), pb.x(), pb.y())
                r, g, bl = skeleton_edge_color(a, b, total)
                line.setPen(QPen(QColor(r, g, bl), 2))
                line.setZValue(0)  # görselin (z=-10) üstünde, noktaların (z=1) altında
                self.scene.addItem(line)
                self.line_items.append(line)

    def set_point_radius(self, radius):
        self.point_radius = radius
        for item in self.point_items.values():
            item.set_radius(radius)

    # ---------- Undo / Redo ----------
    def snapshot(self):
        return sorted(
            (idx, round(it.scenePos().x(), 3), round(it.scenePos().y(), 3), it.visibility)
            for idx, it in self.point_items.items()
        )

    def restore(self, snap):
        for it in list(self.point_items.values()):
            self.scene.removeItem(it)
        self.point_items = {}
        for (idx, x, y, v) in snap:
            self._create_item(idx, x, y, v)
        self._advance_active()
        self._redraw_skeleton()
        self._update_status()

    def push_undo(self):
        self._undo.append(self.snapshot())
        self._redo.clear()

    def begin_interaction(self):
        self._pending = self.snapshot()

    def end_interaction(self):
        if self._pending is None:
            return
        if self._pending != self.snapshot():
            self._undo.append(self._pending)
            self._redo.clear()
            self.mark_dirty()
        self._pending = None

    def undo(self):
        if not self._undo:
            self._status_msg("Geri alınacak işlem yok.")
            return
        self._redo.append(self.snapshot())
        self.restore(self._undo.pop())
        self.mark_dirty()
        self._status_msg("Geri alındı (Ctrl+Z).")

    def redo(self):
        if not self._redo:
            self._status_msg("İleri alınacak işlem yok.")
            return
        self._undo.append(self.snapshot())
        self.restore(self._redo.pop())
        self.mark_dirty()
        self._status_msg("İleri alındı.")

    def reset_history(self):
        self._undo = []
        self._redo = []
        self._pending = None
        self.dirty = False

    def mark_dirty(self):
        self.dirty = True

    # ---------- Buton eylemleri ----------
    def undo_last_point(self):
        """'Geri Al' butonu — genel undo ile aynı."""
        self.undo()

    def clear_points(self):
        if self.point_items:
            self.push_undo()
        for item in list(self.point_items.values()):
            self.scene.removeItem(item)
        for line in self.line_items:
            self.scene.removeItem(line)
        self.point_items = {}
        self.line_items = []
        self.active_index = 0
        self.mark_dirty()
        self._update_status()

    def load_keypoints(self, keypoints):
        """
        Kayıtlı keypoint'leri INDEX'i koruyarak geri yükler.
        v>0 ya da (0,0) olmayan koordinatlı noktalar yerleştirilmiş kabul edilir;
        böylece iskelet index eşlemesi bozulmaz.
        """
        for item in list(self.point_items.values()):
            self.scene.removeItem(item)
        for line in self.line_items:
            self.scene.removeItem(line)
        self.point_items = {}
        self.line_items = []

        for i, (x, y, v) in enumerate(keypoints):
            if i >= self.num_kpts():
                break
            if v > 0 or x != 0 or y != 0:
                self._create_item(i, x, y, v)
        self._advance_active()
        self._redraw_skeleton()
        self.reset_history()
        self._update_status()

    # ---------- Durum ----------
    def _status_msg(self, text):
        if self.status_callback:
            self.status_callback(text, self.active_index if self.active_index < self.num_kpts() else max(self.num_kpts() - 1, 0))

    def _update_status(self):
        if not self.status_callback:
            return
        n = self.num_kpts()
        placed = len(self.point_items)
        ai = self.active_index
        if ai < n and n > 0:
            name = project_state.keypoint_names[ai] if ai < len(project_state.keypoint_names) else str(ai)
            self.status_callback(f"Aktif: {ai}: {name}  |  Yerleşen: {placed}/{n}", ai)
        else:
            self.status_callback(f"Tüm noktalar yerleşti ({placed}/{n})", max(n - 1, 0))

    def get_keypoints(self):
        """Sıralı (index'e göre) keypoint listesi; boş slotlar (0,0,0)."""
        result = []
        for i in range(self.num_kpts()):
            it = self.point_items.get(i)
            if it is not None:
                p = it.scenePos()
                result.append((p.x(), p.y(), it.visibility))
            else:
                result.append((0, 0, 0))
        return result


class KeypointLabelingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_folder = None
        self.label_folder = None      # kaynak etiket klasörü (images -> labels)
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
        # Klavye kısayollarının canvas'tan gezinme/kaydı tetiklemesi için geri çağrılar
        self.canvas.on_prev = self.prev_image
        self.canvas.on_next = self.next_image
        self.canvas.on_save = self.save_current
        self.canvas.on_labels_toggled = self._on_labels_toggled
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

        right_panel.addWidget(QLabel("Keypoint Sırası (tıkla = aktif nokta)"))
        self.kp_list = QListWidget()
        self.kp_list.itemClicked.connect(self._on_kp_list_clicked)
        right_panel.addWidget(self.kp_list)

        self.btn_delete_point = QPushButton("Seçili Noktayı Sil (Del)")
        self.btn_delete_point.clicked.connect(lambda: self.canvas.delete_selected())
        right_panel.addWidget(self.btn_delete_point)

        self.btn_toggle_labels = QPushButton("Etiketleri Göster (L)")
        self.btn_toggle_labels.setCheckable(True)
        self.btn_toggle_labels.clicked.connect(self.toggle_labels)
        right_panel.addWidget(self.btn_toggle_labels)

        info = QLabel(
            "Sol tık: nokta ekle/seç\nSürükle: taşı\n"
            "Çift tık / V: görünürlük\nDelete/X: sil\nCtrl+Z: geri al"
        )
        info.setStyleSheet("font-size: 11px;")
        right_panel.addWidget(info)

        btn_help = QPushButton("Yardım (F1)")
        btn_help.clicked.connect(self.show_help)
        right_panel.addWidget(btn_help)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(200)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(center_widget, stretch=1)
        main_layout.addWidget(right_widget)
        self.setLayout(main_layout)

        # Klavye kısayolları — Detect ile tutarlı gezinme/kayıt.
        # (Ok tuşları, 0-9, Delete/X, V, R, Ctrl+Z/Y canvas.keyPressEvent'te; canvas
        #  odaktayken çalışır. A/D/S/Ctrl+S pencere genelinde çalışsın.)
        QShortcut(QKeySequence("A"), self, self.prev_image)
        QShortcut(QKeySequence("D"), self, self.next_image)
        QShortcut(QKeySequence("S"), self, self.save_current)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current)

    def on_size_changed(self, value):
        self.size_value_label.setText(str(value))
        self.canvas.set_point_radius(value)

    def _set_status(self, text, kp_index):
        self.status_label.setText(text)
        if 0 <= kp_index < self.kp_list.count():
            self.kp_list.blockSignals(True)
            self.kp_list.setCurrentRow(kp_index)
            self.kp_list.blockSignals(False)

    def _on_kp_list_clicked(self, item):
        """Sağ listeden bir keypoint'e tıklanınca onu aktif index yapar."""
        row = self.kp_list.row(item)
        if row >= 0:
            self.canvas.set_active_index(row)

    def toggle_labels(self):
        """'Etiketleri Göster' butonu — nokta etiketlerini aç/kapat."""
        self.canvas.toggle_labels()

    def _on_labels_toggled(self, state):
        """Etiket görünürlüğü değişince (buton ya da L tuşu) butonu senkronla."""
        self.btn_toggle_labels.setChecked(state)
        self.btn_toggle_labels.setText("Etiketleri Gizle (L)" if state else "Etiketleri Göster (L)")

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Resim Klasörünü Seç")
        if not folder:
            return
        self.image_folder = folder
        # Kaynak etiketler: standart YOLO kuralıyla türet (images/train -> labels/train),
        # yoksa görsellerin yanında ara. Detect Labeling ile aynı çözümleme.
        self.label_folder = derive_labels_dir(folder) or folder

        # Çıktı, seçilen klasörün İÇİNDE oluşturulur (üst dizinde değil)
        self.output_folder = default_output_dir(folder, "keypoint_çıktı")
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

        # Önce çıktı klasöründe kaydedilmiş etiket, yoksa KAYNAK etiket (labels/train)
        base_name = os.path.splitext(self.image_files[row])[0]
        out_label = os.path.join(self.output_folder, "labels", base_name + ".txt") if self.output_folder else None
        src_label = os.path.join(self.label_folder, base_name + ".txt") if self.label_folder else None

        label_to_load = None
        if out_label and os.path.exists(out_label):
            label_to_load = out_label
        elif src_label and os.path.exists(src_label):
            label_to_load = src_label

        n_loaded = 0
        if label_to_load:
            persons = load_pose_label(label_to_load, img_w, img_h)
            if persons:
                self.canvas.load_keypoints(persons[0]['keypoints'])
                n_loaded = sum(1 for _, _, v in persons[0]['keypoints'] if v > 0)

        status = f"{row + 1}/{len(self.image_files)} — {self.image_files[row]}"
        if label_to_load:
            status += f" — {n_loaded} nokta yüklendi"
        else:
            status += " — etiket yok (yeni)"
        self.status_label.setText(status)

    def save_current(self):
        if self.current_index == -1 or self.canvas.pixmap_item is None:
            return
        if self.output_folder is None:
            return

        if not self.canvas.has_points():
            return  # hiç yerleştirilmiş nokta yoksa kaydetme

        keypoints = list(self.canvas.get_keypoints())
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
        self.canvas.dirty = False
        self.status_label.setText(f"Kaydedildi: {base_name}")

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.file_list.setCurrentRow(self.current_index + 1)

    def prev_image(self):
        if self.current_index > 0:
            self.file_list.setCurrentRow(self.current_index - 1)

    def show_help(self):
        """F1 — Keypoint Labeling kısayol ve fare rehberi."""
        show_help(
            self,
            "Keypoint Labeling (Anahtar Nokta Etiketleme) — Yardım",
            "İnsan iskeleti / poz noktalarını (keypoint) yerleştirme ve düzenleme "
            "modülü. Noktaları tıklayarak ekle, seç, sürükle, görünürlüğünü değiştir. "
            "Kaynak etiketler (labels/) otomatik yüklenir; düzeltmeler keypoint_çıktı "
            "klasörüne kaydedilir. (Ok/0-9/Delete/V/R/Ctrl+Z için canvas odakta olmalı.)",
            [
                ("Dosya Yönetimi", [
                    ("Resim Klasörü Aç", "Görsel klasörünü seçer (labels/ otomatik bulunur)"),
                    ("data.yaml Yükle", "Keypoint şablonunu (nokta sayısı) yükler"),
                    ("Şablon Düzenle", "Nokta ve iskelet bağlantılarını özelleştirir"),
                ]),
                ("Görsel Gezinme", [
                    ("A / Sol Ok", "Önceki görsel"),
                    ("D / Sağ Ok", "Sonraki görsel"),
                    ("Fare Tekerleği", "Yakınlaştır / uzaklaştır (zoom)"),
                    ("R", "Zoom'u sıfırla (görsele sığdır)"),
                    ("Orta Tuş (basılı)", "Görseli kaydır (pan)"),
                ]),
                ("Etiketleme", [
                    ("Sol tık (boş alan)", "Aktif keypoint'i buraya yerleştirir"),
                    ("Sol tık (nokta)", "Noktayı seçer (yeşil halka)"),
                    ("Sürükle", "Seçili/var olan noktayı taşır"),
                    ("0-9", "Aktif keypoint index'ini seçer"),
                    ("Sağ listeye tık", "Aktif keypoint'i seçer"),
                    ("V / Çift tık", "Görünürlük: görünür→örtük(yarı saydam)→yok(gri)"),
                    ("L", "Nokta index/isim etiketlerini aç/kapat"),
                    ("Delete / X", "Seçili noktayı sil"),
                ]),
                ("Renkler", [
                    ("Her nokta", "Kendi sabit renginde (elde parmak bazlı)"),
                    ("İskelet çizgileri", "Parmak grubuna göre renkli"),
                    ("Seçili nokta", "Kalın beyaz halka (rengini korur)"),
                ]),
                ("Geri Al / Yinele", [
                    ("Ctrl+Z", "Son işlemi geri al"),
                    ("Ctrl+Y / Ctrl+Shift+Z", "Yinele (ileri al)"),
                    ("Geri Al / Temizle", "Buton: son işlemi geri al / tümünü temizle"),
                ]),
                ("Kayıt", [
                    ("S / Ctrl+S", "Mevcut görseli kaydet"),
                    ("A/D ile geçiş", "Görsel değişince otomatik kaydedilir"),
                ]),
            ]
        )

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
    "El (MediaPipe 21)": {
        "names": [
            "wrist",
            "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip",
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ],
        "skeleton": [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
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