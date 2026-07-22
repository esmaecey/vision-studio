"""
Detect Etiketleme Arayüzü — Bounding Box annotation tool.
Zoom, pan, kutu çizme/taşıma/boyutlandırma/silme, sınıf yönetimi, YOLO kayıt.
"""
import os 
import cv2
import shutil
from PyQt5.QtGui import QColor, QBrush  # zaten bazıları var, eksikse ekle
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QListWidget,
    QPushButton, QLabel, QFileDialog, QInputDialog, QShortcut, QMessageBox
)
from PyQt5.QtGui import QPixmap, QPen, QColor, QBrush, QKeySequence, QPainter
from PyQt5.QtCore import Qt, QRectF
from config.project_state import project_state
from utils.dataset_paths import derive_labels_dir, default_output_dir

from .utils import (
    load_yolo_labels, save_yolo_labels, pixel_to_yolo, yolo_to_pixel,
    get_class_color, draw_boxes_on_image, save_image_unicode
)
from .utils import (
    load_yolo_labels, save_yolo_labels, pixel_to_yolo, yolo_to_pixel, get_class_color
)


class BoxItem(QGraphicsRectItem):
    """Taşınabilir ve yeniden boyutlandırılabilir bounding box öğesi."""

    HANDLE_SIZE = 8

    def __init__(self, rect, class_id, kpt_tail=""):
        super().__init__(rect)
        self.class_id = class_id
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
        self.update_color()

    def update_color(self):
        r, g, b = get_class_color(self.class_id)
        self.setPen(QPen(QColor(r, g, b), 2))
        self.setBrush(QBrush(QColor(r, g, b, 40)))

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


class AnnotationCanvas(QGraphicsView):
    """Görsel gösterimi, zoom, pan ve bbox çizimini yöneten canvas."""

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.pixmap_item = None
        self.box_items = []
        self.current_class_id = 0
        self.draw_mode = True

        self._drawing = False
        self._start_point = None
        self._temp_rect_item = None

        self._panning = False
        self._pan_start = None

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
                box_item = BoxItem(rect, self.current_class_id)
                self.scene.addItem(box_item)
                self.box_items.append(box_item)
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            for item in self.scene.selectedItems():
                if item in self.box_items:
                    self.box_items.remove(item)
                self.scene.removeItem(item)
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
            box_item = BoxItem(QRectF(x1, y1, w_px, h_px), class_id, kpt_tail=tail)
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

        right_panel.addWidget(QLabel("Sınıflar"))
        self.class_list = QListWidget()

        self.class_list.currentRowChanged.connect(self.on_class_selected)
        right_panel.addWidget(self.class_list)

        self.btn_add_class = QPushButton("Yeni Sınıf Ekle")
        self.btn_add_class.clicked.connect(self.add_class)
        right_panel.addWidget(self.btn_add_class)

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
        self.class_list.clear()
        for i, name in enumerate(self.class_names):
            self.class_list.addItem(f"{i}: {name}")
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)

    def on_class_selected(self, row):
        if row >= 0:
            self.canvas.current_class_id = row

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

        # Yeni görsele geçmeden önce mevcut görseli otomatik kaydet
        if self.current_index != -1 and self.current_index != row:
            self.save_current_labels()

        self.current_index = row
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

        self.status_label.setText(f"Kaydedildi: {image_filename} ({len(boxes)} kutu)")

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
