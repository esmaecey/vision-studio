"""
Detect Etiketleme Arayüzü — Bounding Box annotation tool.
Zoom, pan, kutu çizme/taşıma/boyutlandırma/silme, sınıf yönetimi, YOLO kayıt.
"""
import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QListWidget,
    QPushButton, QLabel, QFileDialog, QInputDialog, QShortcut, QMessageBox
)
from PyQt5.QtGui import (
    QPixmap, QPen, QColor, QBrush, QKeySequence, QPainter, QFont, QFontMetrics
)
from PyQt5.QtCore import Qt, QRectF
from config.project_state import project_state
from utils.dataset_paths import derive_labels_dir, default_output_dir
from gui.common.help import show_help

from .utils import (
    load_yolo_labels, save_yolo_labels, pixel_to_yolo, yolo_to_pixel,
    get_class_color, draw_boxes_on_image, save_image_unicode
)


class BoxLabel(QGraphicsItem):
    """Kutunun sol üst köşesinde duran, zoom'dan etkilenmeyen sınıf adı etiketi.

    Arka plan kutunun rengiyle doldurulur; yazı rengi (siyah/beyaz) arka planın
    parlaklığına göre otomatik seçilir, böylece her sınıf renginde okunur kalır.
    """

    def __init__(self, parent):
        super().__init__(parent)
        # Zoom yapılınca yazı küçülüp büyümesin; her zaman aynı boyda kalsın.
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._text = ""
        self._bg = QColor(0, 0, 0)
        self._fg = QColor(255, 255, 255)
        self._font = QFont()
        self._font.setPointSize(9)
        self._font.setBold(True)

    def _size(self):
        fm = QFontMetrics(self._font)
        w = fm.horizontalAdvance(self._text) + 8
        h = fm.height() + 4
        return w, h

    def set_label(self, text, rgb):
        self.prepareGeometryChange()
        self._text = text
        r, g, b = rgb
        self._bg = QColor(r, g, b)
        # Parlaklığa göre kontrastlı yazı rengi (koyu zeminde açık, açıkta koyu).
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        self._fg = QColor(0, 0, 0) if luminance > 140 else QColor(255, 255, 255)
        self.update()

    def boundingRect(self):
        w, h = self._size()
        # Etiket, kutunun üst kenarının hemen üstünde dursun (negatif y).
        return QRectF(0, -h, w, h)

    def paint(self, painter, option, widget=None):
        if not self._text:
            return
        painter.setFont(self._font)
        w, h = self._size()
        rect = QRectF(0, -h, w, h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._bg)
        painter.drawRect(rect)
        painter.setPen(QPen(self._fg))
        painter.drawText(rect, Qt.AlignCenter, self._text)


class BoxItem(QGraphicsRectItem):
    """Taşınabilir ve yeniden boyutlandırılabilir bounding box öğesi."""

    HANDLE_SIZE = 8

    def __init__(self, rect, class_id, canvas=None, kpt_tail=""):
        super().__init__(rect)
        self.class_id = class_id
        # Sınıf adını çözebilmek için canvas referansı (canvas.class_names tutar).
        self.canvas = canvas
        # Pose etiketlerinde bu kutuya ait keypoint token'ları (düzenlemede
        # kullanılmaz ama kayıtta korunarak geri yazılır). Yeni kutularda boş.
        self.kpt_tail = kpt_tail
        self.setFlags(
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._resizing = False
        self._resize_dir = (False, False, False, False)
        # Kutunun üstündeki sınıf adı etiketi (bu kutunun çocuğu).
        self.label = BoxLabel(self)
        self.update_appearance()

    def _class_name(self):
        if self.canvas is not None:
            return self.canvas.class_name_for(self.class_id)
        return str(self.class_id)

    def update_appearance(self):
        """Rengi, seçim vurgusunu ve üstteki sınıf adı etiketini günceller."""
        r, g, b = get_class_color(self.class_id)
        if self.isSelected():
            # Seçili kutu daha kalın kenar ve daha belirgin dolgu ile öne çıksın.
            self.setPen(QPen(QColor(r, g, b), 4))
            self.setBrush(QBrush(QColor(r, g, b, 80)))
        else:
            self.setPen(QPen(QColor(r, g, b), 2))
            self.setBrush(QBrush(QColor(r, g, b, 40)))
        self.label.set_label(self._class_name(), (r, g, b))
        self._position_label()

    def _position_label(self):
        rect = self.rect()
        # Etiketi kutunun sol üst köşesine yerleştir (çocuk, kutuyla taşınır).
        self.label.setPos(rect.left(), rect.top())

    def set_class(self, class_id):
        """Kutunun sınıfını değiştirir; renk ve etiket anında güncellenir."""
        self.class_id = class_id
        self.update_appearance()

    def setRect(self, *args):
        super().setRect(*args)
        # Boyut/konum değişince etiket köşede kalsın.
        if getattr(self, "label", None) is not None:
            self._position_label()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self.update_appearance()
        return super().itemChange(change, value)

    def _edge_flags(self, pos):
        rect = self.rect()
        m = self.HANDLE_SIZE
        near_left = abs(pos.x() - rect.left()) < m
        near_right = abs(pos.x() - rect.right()) < m
        near_top = abs(pos.y() - rect.top()) < m
        near_bottom = abs(pos.y() - rect.bottom()) < m
        return near_left, near_right, near_top, near_bottom

    def hoverMoveEvent(self, event):
        left, right, top, bottom = self._edge_flags(event.pos())
        if (right and bottom) or (left and top):
            self.setCursor(Qt.SizeFDiagCursor)
        elif (right and top) or (left and bottom):
            self.setCursor(Qt.SizeBDiagCursor)
        elif right or left:
            self.setCursor(Qt.SizeHorCursor)
        elif bottom or top:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        # Taşıma/boyutlandırma öncesi durumu kaydet (undo için).
        if self.canvas is not None:
            self.canvas.begin_interaction()
        flags = self._edge_flags(event.pos())
        self._resizing = any(flags)
        self._resize_dir = flags
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            rect = QRectF(self.rect())
            pos = event.pos()
            left, right, top, bottom = self._resize_dir
            if left:
                rect.setLeft(pos.x())
            if right:
                rect.setRight(pos.x())
            if top:
                rect.setTop(pos.y())
            if bottom:
                rect.setBottom(pos.y())
            if rect.width() > 5 and rect.height() > 5:
                self.setRect(rect.normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._resizing = False
        super().mouseReleaseEvent(event)
        # Taşıma/boyutlandırma gerçekten değiştiyse undo geçmişine işle.
        if self.canvas is not None:
            self.canvas.end_interaction()


class AnnotationCanvas(QGraphicsView):
    """Görsel gösterimi, zoom, pan ve bbox çizimini yöneten canvas."""

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFocusPolicy(Qt.StrongFocus)

        self.pixmap_item = None
        self.box_items = []
        # Sağ listede seçili sınıf; None ise seçim yok (yeni kutu çizilemez).
        self.current_class_id = None
        # Aktif projenin sınıf adları (widget tarafından senkron tutulur).
        self.class_names = []
        self.draw_mode = True

        # Widget'in bağladığı geri çağrılar.
        self.on_no_class = None            # sınıf seçili değilken çizim denemesi
        self.on_selected_class_changed = None  # klavye ile sınıf değişince
        self.on_status = None              # durum çubuğuna kısa mesaj yazmak için

        # Geri alma/ileri alma (undo/redo) geçmişi ve değişiklik bayrağı.
        self._undo_stack = []
        self._redo_stack = []
        self._pending_snapshot = None      # taşıma/boyutlandırma öncesi durum
        self.dirty = False                 # kaydedilmemiş değişiklik var mı

        self._drawing = False
        self._start_point = None
        self._temp_rect_item = None

        self._panning = False
        self._pan_start = None

    def class_name_for(self, class_id):
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def refresh_box_labels(self):
        """Sınıf adları değişince mevcut kutuların etiketlerini yeniler."""
        for item in self.box_items:
            item.update_appearance()

    # ---------- Değişiklik bayrağı & durum ----------
    def mark_dirty(self):
        """Kaydedilmemiş değişiklik olduğunu işaretler (otomatik kayıt için)."""
        self.dirty = True

    def _status(self, msg):
        if callable(self.on_status):
            self.on_status(msg)

    # ---------- Geri alma / ileri alma (undo/redo) ----------
    def _snapshot(self):
        """Mevcut tüm kutuların durumunu (sahne koordinatlarında) döndürür."""
        snap = []
        for item in self.box_items:
            rect = item.rect()
            pos = item.pos()
            x1 = rect.x() + pos.x()
            y1 = rect.y() + pos.y()
            snap.append((
                item.class_id,
                round(x1, 3), round(y1, 3),
                round(rect.width(), 3), round(rect.height(), 3),
                getattr(item, "kpt_tail", ""),
            ))
        return snap

    def _restore(self, snapshot):
        """Verilen anlık görüntüye göre tüm kutuları yeniden kurar."""
        for item in list(self.box_items):
            self.scene.removeItem(item)
        self.box_items = []
        for class_id, x1, y1, w, h, tail in snapshot:
            box_item = BoxItem(QRectF(x1, y1, w, h), class_id, canvas=self, kpt_tail=tail)
            self.scene.addItem(box_item)
            self.box_items.append(box_item)

    def push_undo_state(self):
        """Değiştirici bir işlemden ÖNCE mevcut durumu geçmişe kaydeder."""
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def begin_interaction(self):
        """Taşıma/boyutlandırma başlarken önceki durumu geçici olarak tutar."""
        self._pending_snapshot = self._snapshot()

    def end_interaction(self):
        """Etkileşim bitince, gerçekten değişiklik olduysa undo'ya işler."""
        if self._pending_snapshot is None:
            return
        if self._pending_snapshot != self._snapshot():
            self._undo_stack.append(self._pending_snapshot)
            self._redo_stack.clear()
            self.mark_dirty()
        self._pending_snapshot = None

    def undo(self):
        if not self._undo_stack:
            self._status("Geri alınacak işlem yok.")
            return
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self.mark_dirty()
        self._status("Geri alındı (Ctrl+Z).")

    def redo(self):
        if not self._redo_stack:
            self._status("İleri alınacak işlem yok.")
            return
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self.mark_dirty()
        self._status("İleri alındı.")

    def reset_history(self):
        """Yeni görsele geçince geçmişi ve değişiklik bayrağını sıfırlar."""
        self._undo_stack = []
        self._redo_stack = []
        self._pending_snapshot = None
        self.dirty = False

    def delete_selected(self):
        """Seçili kutu(ları) siler. Hiç seçili yoksa hiçbir şey yapmaz."""
        selected = [i for i in self.scene.selectedItems() if isinstance(i, BoxItem)]
        if not selected:
            return False
        self.push_undo_state()
        for item in selected:
            if item in self.box_items:
                self.box_items.remove(item)
            self.scene.removeItem(item)
        self.mark_dirty()
        self._status(f"{len(selected)} kutu silindi.")
        return True

    def load_image(self, image_path):
        self.scene.clear()
        self.box_items = []
        pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
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

        if self.draw_mode and event.button() == Qt.LeftButton:
            item_at_pos = self.itemAt(event.pos())
            if item_at_pos is None or item_at_pos == self.pixmap_item:
                # Boş alana tıklandı → yeni kutu çizimi. Sınıf seçili değilse uyar.
                if self.current_class_id is None:
                    if callable(self.on_no_class):
                        self.on_no_class()
                    return
                self._drawing = True
                self._start_point = self.mapToScene(event.pos())
                rect = QRectF(self._start_point, self._start_point)
                self._temp_rect_item = QGraphicsRectItem(rect)
                self._temp_rect_item.setPen(QPen(QColor(255, 255, 255), 2, Qt.DashLine))
                self.scene.addItem(self._temp_rect_item)
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self._drawing and self._temp_rect_item is not None:
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self._start_point, current_point).normalized()
            self._temp_rect_item.setRect(rect)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return

        if self._drawing and self._temp_rect_item is not None:
            rect = self._temp_rect_item.rect()
            self.scene.removeItem(self._temp_rect_item)
            self._temp_rect_item = None
            self._drawing = False

            if rect.width() > 5 and rect.height() > 5:
                # Yeni kutu, o an seçili sınıfı alır (None ise yukarıda çizim
                # zaten engellenmişti; güvenlik için 0'a düşülür).
                self.push_undo_state()
                class_id = self.current_class_id if self.current_class_id is not None else 0
                box_item = BoxItem(rect, class_id, canvas=self)
                self.scene.addItem(box_item)
                self.box_items.append(box_item)
                self.mark_dirty()
                # Yeni kutuyu seç ki kullanıcı hemen sınıf/konum düzenleyebilsin.
                self.scene.clearSelection()
                box_item.setSelected(True)
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)

        # Ctrl+Z: geri al  |  Ctrl+Y ya da Ctrl+Shift+Z: ileri al
        if ctrl and key == Qt.Key_Z and not shift:
            self.undo()
            return
        if (ctrl and key == Qt.Key_Y) or (ctrl and shift and key == Qt.Key_Z):
            self.redo()
            return

        # Delete / Backspace: seçili kutuyu sil (seçili yoksa bir şey olmaz).
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
            return

        # 0-9 tuşları: seçili kutunun sınıfını doğrudan değiştirir.
        if Qt.Key_0 <= key <= Qt.Key_9:
            digit = key - Qt.Key_0
            selected = [i for i in self.scene.selectedItems() if isinstance(i, BoxItem)]
            if selected and digit < len(self.class_names):
                self.push_undo_state()
                for item in selected:
                    item.set_class(digit)
                self.mark_dirty()
                if callable(self.on_selected_class_changed):
                    self.on_selected_class_changed(digit)
            return

        super().keyPressEvent(event)

    def get_boxes_yolo(self, img_width, img_height):
        boxes = []
        for item in self.box_items:
            rect = item.rect()
            pos = item.pos()
            x1 = rect.x() + pos.x()
            y1 = rect.y() + pos.y()
            xc, yc, w, h = pixel_to_yolo(x1, y1, rect.width(), rect.height(), img_width, img_height)
            boxes.append((item.class_id, xc, yc, w, h, getattr(item, "kpt_tail", "")))
        return boxes

    def load_boxes_yolo(self, boxes, img_width, img_height):
        for box in boxes:
            class_id, xc, yc, w, h = box[:5]
            tail = box[5] if len(box) > 5 else ""
            x1, y1, w_px, h_px = yolo_to_pixel(xc, yc, w, h, img_width, img_height)
            box_item = BoxItem(QRectF(x1, y1, w_px, h_px), class_id, canvas=self, kpt_tail=tail)
            self.scene.addItem(box_item)
            self.box_items.append(box_item)


class DetectLabelingWidget(QWidget):
    """Detect (bounding box) etiketleme ana arayüzü."""

    def __init__(self):
        super().__init__()
        self.image_folder = None
        self.label_folder = None
        self.image_files = []
        self.current_index = -1
        self.output_folder = None          # çıktı_data yolu
        self.processed_files = set()        # kaydedilmiş (yeşil) dosya adları
        self._build_ui()
        # Merkezi proje durumundaki değişiklikleri dinle
        project_state.add_listener(self._on_project_changed)
        self._populate_class_list()

    @property
    def class_names(self):
        return project_state.class_names

    def class_name(self, class_id):
        names = self.class_names
        if 0 <= class_id < len(names):
            return names[class_id]
        return str(class_id)

    def _build_ui(self):
        main_layout = QHBoxLayout()

        # Sol panel: dosya listesi
        left_panel = QVBoxLayout()
        self.btn_open_folder = QPushButton("Resim Klasörü Aç")
        self.btn_open_folder.clicked.connect(self.open_folder)
        left_panel.addWidget(self.btn_open_folder)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        left_panel.addWidget(self.file_list)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(250)

        # Orta: canvas
        center_layout = QVBoxLayout()
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("< Önceki (A)")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("Sonraki > (D)")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_save = QPushButton("Kaydet (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_current_labels)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.btn_save)
        center_layout.addLayout(nav_layout)

        self.canvas = AnnotationCanvas()
        self.canvas.on_no_class = self._warn_no_class
        self.canvas.on_selected_class_changed = self._on_canvas_class_changed
        self.canvas.on_status = self.status_label_setter
        center_layout.addWidget(self.canvas)

        self.status_label = QLabel("Bir klasör açarak başlayın.")
        center_layout.addWidget(self.status_label)

        center_widget = QWidget()
        center_widget.setLayout(center_layout)

        # Sağ panel: sınıflar
        right_panel = QVBoxLayout()

        self.project_label = QLabel(project_state.summary())
        self.project_label.setWordWrap(True)
        self.project_label.setStyleSheet("font-size: 11px; padding: 4px;")
        right_panel.addWidget(self.project_label)

        btn_load_yaml = QPushButton("data.yaml Yükle")
        btn_load_yaml.clicked.connect(self.load_data_yaml)
        right_panel.addWidget(btn_load_yaml)

        right_panel.addWidget(QLabel("Sınıflar (tıkla veya 0-9)"))
        self.class_list = QListWidget()
        # currentRowChanged: yeni kutular için aktif sınıfı belirler.
        self.class_list.currentRowChanged.connect(self.on_class_selected)
        # itemClicked: aynı satıra tekrar tıklanınca da seçili kutuya uygulanır.
        self.class_list.itemClicked.connect(self.on_class_clicked)
        right_panel.addWidget(self.class_list)

        self.btn_add_class = QPushButton("Yeni Sınıf Ekle")
        self.btn_add_class.clicked.connect(self.add_class)
        right_panel.addWidget(self.btn_add_class)

        self.btn_delete_box = QPushButton("Seçili Kutuyu Sil (Del)")
        self.btn_delete_box.clicked.connect(self.delete_selected_boxes)
        right_panel.addWidget(self.btn_delete_box)

        # Varsayılan kayıt güvenli çıktı_data'ya yapılır (Kaydet / Ctrl+S).
        # Bu buton ise düzeltmeleri DOĞRUDAN orijinal etiket dosyasına yazar.
        self.btn_update_original = QPushButton("Orijinali Güncelle")
        self.btn_update_original.setToolTip(
            "Seçili görselin düzeltmelerini doğrudan ORİJİNAL etiket dosyasına yazar "
            "(çıktı_data yerine). Dikkatli kullanın — orijinalin üzerine yazılır."
        )
        self.btn_update_original.clicked.connect(self.update_original_labels)
        right_panel.addWidget(self.btn_update_original)

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

        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current_labels)
        QShortcut(QKeySequence("D"), self, self.next_image)
        QShortcut(QKeySequence("A"), self, self.prev_image)

    def _populate_class_list(self):
        # Canvas'ın sınıf adlarını senkron tut (etiket metinleri buradan gelir).
        self.canvas.class_names = list(self.class_names)

        # Repopülasyon sırasında setCurrentRow'un sinyali seçili kutuyu
        # yanlışlıkla değiştirmesin diye sinyalleri geçici olarak durdur.
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for i, name in enumerate(self.class_names):
            self.class_list.addItem(f"{i}: {name}")
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.canvas.current_class_id = 0
        else:
            self.canvas.current_class_id = None
        self.class_list.blockSignals(False)

        # Açık bir görsel varsa kutu etiketleri güncel adlarla yenilensin.
        self.canvas.refresh_box_labels()

    def status_label_setter(self, msg):
        """Canvas'ın durum çubuğuna yazması için geri çağrı."""
        self.status_label.setText(msg)

    def delete_selected_boxes(self):
        """Sağ paneldeki 'Seçili Kutuyu Sil' butonu için."""
        if not self.canvas.delete_selected():
            self.status_label.setText("Silinecek seçili kutu yok.")

    def _apply_class_to_selection(self, class_id):
        """Sağ listeden seçilen sınıfı, o an seçili kutulara uygular (undo'lu)."""
        selected = [i for i in self.canvas.scene.selectedItems() if isinstance(i, BoxItem)]
        if not selected:
            return
        self.canvas.push_undo_state()
        for item in selected:
            item.set_class(class_id)
        self.canvas.mark_dirty()
        self.status_label.setText(
            f"Seçili kutu sınıfı: {class_id} ({self.class_name(class_id)})"
        )

    def on_class_selected(self, row):
        """Sağ listede satır değişince: yalnızca yeni kutular için aktif sınıfı ayarlar.

        (Seçili kutuya uygulamak tıklama ile olur — bkz. on_class_clicked. Böylece
        tek tıklamada hem currentRowChanged hem itemClicked tetiklenip çift undo
        kaydı oluşması engellenir.)
        """
        if row >= 0:
            self.canvas.current_class_id = row
        else:
            self.canvas.current_class_id = None

    def on_class_clicked(self, item):
        """Listedeki bir sınıfa tıklanınca (aynı satır dahil) seçili kutuya uygula."""
        row = self.class_list.row(item)
        if row >= 0:
            self.canvas.current_class_id = row
            self._apply_class_to_selection(row)

    def _on_canvas_class_changed(self, class_id):
        """Klavyeyle (0-9) sınıf değişince sağ listeyi ve aktif sınıfı senkronla."""
        self.class_list.blockSignals(True)
        self.class_list.setCurrentRow(class_id)
        self.class_list.blockSignals(False)
        self.canvas.current_class_id = class_id
        self.status_label.setText(
            f"Seçili kutu sınıfı: {class_id} ({self.class_name(class_id)})"
        )

    def _warn_no_class(self):
        QMessageBox.warning(
            self, "Sınıf seçilmedi",
            "Kutu çizmeden önce sağdaki listeden bir sınıf seçin."
        )

    def add_class(self):
        name, ok = QInputDialog.getText(self, "Yeni Sınıf", "Sınıf adı:")
        if ok and name.strip():
            project_state.add_class(name.strip())

    def load_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "data.yaml Seç", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            classes = project_state.load_data_yaml(path)
            QMessageBox.information(
                self, "Başarılı",
                f"{len(classes)} sınıf yüklendi:\n" + ", ".join(classes[:10]) +
                ("..." if len(classes) > 10 else "")
            )
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"data.yaml okunamadı:\n{str(e)}")

    def _on_project_changed(self):
        """Merkezi proje durumu değişince tetiklenir."""
        self._populate_class_list()
        if hasattr(self, "project_label"):
            self.project_label.setText(project_state.summary())

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Resim Klasörünü Seç")
        if not folder:
            return
        self.image_folder = folder
        # Kaynak etiketler görsellerin yanında ya da standart YOLO yapısında
        # (images/ -> labels/) olabilir; ortak kuralla türet.
        self.label_folder = derive_labels_dir(folder) or folder

        # Çıktı klasörünü oluştur: seçilen klasörün İÇİNDE "çıktı_data"
        self.output_folder = default_output_dir(folder, "çıktı_data")
        os.makedirs(os.path.join(self.output_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "vis"), exist_ok=True)

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(valid_ext))

        self.processed_files = set()
        self.file_list.clear()
        self.file_list.addItems(self.image_files)

        self.current_index = -1  # ilk seçimde yanlışlıkla kayıt tetiklenmesin

        if self.image_files:
            self.file_list.setCurrentRow(0)

        self.status_label.setText(f"Çıktı klasörü: {self.output_folder}")

    def on_file_selected(self, row):
        if row < 0 or row >= len(self.image_files):
            return

        # Yeni görsele geçmeden önce mevcut görseli HER DURUMDA otomatik kaydet
        # (değişiklik yapılmış olsun olmasın). En güvenlisi her geçişte kaydetmek;
        # kullanıcı bir düzeltmeden emin olamayabilir ya da sistem değişikliği
        # algılamayabilir.
        if self.current_index != -1 and self.current_index != row:
            self.save_current_labels()

        self.current_index = row
        # Kutu etiketlerinin doğru adı gösterebilmesi için canvas'ı senkronla.
        self.canvas.class_names = list(self.class_names)
        image_path = os.path.join(self.image_folder, self.image_files[row])
        img_w, img_h = self.canvas.load_image(image_path)

        # Önce çıktı klasöründe daha önce kaydedilmiş etiket var mı, yoksa orijinalde mi bak
        base_name = os.path.splitext(self.image_files[row])[0]
        out_label = os.path.join(self.output_folder, "labels", base_name + ".txt") if self.output_folder else None
        src_label = self._get_label_path(self.image_files[row])

        if out_label and os.path.exists(out_label):
            boxes = load_yolo_labels(out_label)
        else:
            boxes = load_yolo_labels(src_label)

        self.canvas.load_boxes_yolo(boxes, img_w, img_h)
        # Yeni görsel yüklendi: undo geçmişini ve değişiklik bayrağını sıfırla.
        self.canvas.reset_history()

        status = f"{row + 1}/{len(self.image_files)} — {self.image_files[row]} — {len(boxes)} kutu"
        if any(len(b) > 5 and b[5] for b in boxes):
            status += "  (pose etiketi: keypoint verisi korunacak)"
        self.status_label.setText(status)

    def _get_label_path(self, image_filename):
        base_name = os.path.splitext(image_filename)[0]
        return os.path.join(self.label_folder, base_name + ".txt")

    def save_current_labels(self):
        if self.current_index == -1 or self.canvas.pixmap_item is None:
            return
        if self.output_folder is None:
            return

        img_w = self.canvas.pixmap_item.pixmap().width()
        img_h = self.canvas.pixmap_item.pixmap().height()
        boxes = self.canvas.get_boxes_yolo(img_w, img_h)

        image_filename = self.image_files[self.current_index]
        base_name = os.path.splitext(image_filename)[0]
        source_image_path = os.path.join(self.image_folder, image_filename)

        # 1) labels/ -> YOLO txt
        label_out = os.path.join(self.output_folder, "labels", base_name + ".txt")
        save_yolo_labels(label_out, boxes)

        # 2) images/ -> orijinal görselin kopyası
        image_out = os.path.join(self.output_folder, "images", image_filename)
        shutil.copy(source_image_path, image_out)

        # 3) vis/ -> kutular çizili görselleştirilmiş kopya
        vis_img = draw_boxes_on_image(source_image_path, boxes, self.class_names)
        if vis_img is not None:
            vis_out = os.path.join(self.output_folder, "vis", base_name + ".jpg")
            save_image_unicode(vis_out, vis_img)

        # İşlenmiş olarak işaretle (yeşil)
        self.processed_files.add(image_filename)
        self._mark_file_processed(self.current_index)

        # Kaydedildi: artık kaydedilmemiş değişiklik yok.
        self.canvas.dirty = False

        self.status_label.setText(f"Kaydedildi: {image_filename} ({len(boxes)} kutu)")

    def update_original_labels(self):
        """Düzeltmeleri doğrudan ORİJİNAL etiket dosyasına yazar (onay ister)."""
        if self.current_index == -1 or self.canvas.pixmap_item is None:
            QMessageBox.information(self, "Görsel yok", "Önce bir görsel açın.")
            return

        image_filename = self.image_files[self.current_index]
        label_path = self._get_label_path(image_filename)

        confirm = QMessageBox.question(
            self, "Orijinali Güncelle",
            "Düzeltmeler GÜVENLİ çıktı_data yerine doğrudan orijinal etiket "
            f"dosyasının ÜZERİNE yazılacak:\n\n{label_path}\n\n"
            "Bu işlem orijinal dosyayı değiştirir. Devam edilsin mi?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        img_w = self.canvas.pixmap_item.pixmap().width()
        img_h = self.canvas.pixmap_item.pixmap().height()
        boxes = self.canvas.get_boxes_yolo(img_w, img_h)

        try:
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            save_yolo_labels(label_path, boxes)
        except OSError as e:
            QMessageBox.critical(self, "Hata", f"Orijinal dosya yazılamadı:\n{e}")
            return

        self.status_label.setText(
            f"Orijinal güncellendi: {os.path.basename(label_path)} ({len(boxes)} kutu)"
        )

    def show_help(self):
        """F1 — Detect Labeling kısayol ve fare rehberi."""
        show_help(
            self,
            "Detect Labeling (Nesne Etiketleme) — Yardım",
            "Bounding box (kutu) etiketleme modülü. Kutuları çiz, sınıfını düzelt, "
            "sil ve kaydet. Düzeltmeler varsayılan olarak güvenli 'çıktı_data' "
            "klasörüne yazılır.",
            [
                ("Dosya Yönetimi", [
                    ("Resim Klasörü Aç", "Etiketlenecek görsellerin klasörünü seçer"),
                    ("data.yaml Yükle", "Aktif projenin sınıf isimlerini yükler"),
                ]),
                ("Görsel Gezinme", [
                    ("A / Sol Ok", "Önceki görsel"),
                    ("D / Sağ Ok", "Sonraki görsel"),
                    ("Fare Tekerleği", "Yakınlaştır / uzaklaştır (zoom)"),
                    ("Orta Tuş (basılı)", "Görseli kaydır (pan)"),
                ]),
                ("Etiketleme", [
                    ("Sol tık + sürükle", "Boş alanda yeni kutu çiz"),
                    ("Sol tık (kutuya)", "Kutuyu seç (vurgulanır)"),
                    ("Kenardan sürükle", "Seçili kutuyu yeniden boyutlandır"),
                    ("0-9", "Seçili kutunun sınıfını değiştir / yeni kutu sınıfını seç"),
                    ("Sınıf listesine tık", "Seçili kutunun sınıfını o sınıf yapar"),
                    ("Delete / Backspace", "Seçili kutuyu sil"),
                ]),
                ("Geri Al / Yinele", [
                    ("Ctrl+Z", "Son işlemi geri al"),
                    ("Ctrl+Y / Ctrl+Shift+Z", "Yinele (ileri al)"),
                ]),
                ("Kayıt", [
                    ("Ctrl+S", "Güvenli çıktı_data'ya kaydet"),
                    ("A/D ile geçiş", "Görsel değişince otomatik kaydedilir"),
                    ("Orijinali Güncelle", "Düzeltmeleri doğrudan orijinal dosyaya yazar"),
                ]),
            ]
        )

    def _mark_file_processed(self, index):
        """Listedeki ilgili satırı yeşile boyar."""
        item = self.file_list.item(index)
        if item is not None:
            item.setForeground(QBrush(QColor(0, 170, 0)))

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.file_list.setCurrentRow(self.current_index + 1)

    def prev_image(self):
        if self.current_index > 0:
            self.file_list.setCurrentRow(self.current_index - 1)
