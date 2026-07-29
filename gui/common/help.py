"""
Ortak yardım (F1) penceresi.

Modüller kendi kısayol/fare rehberlerini kategorilere ayrılmış olarak gösterir.
Kısayolu olmayan modüller (Augmentation, Split, Training, Testing) için ise
sadece kısa bir bilgilendirme metni gösterilebilir.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QWidget,
    QDialogButtonBox, QFrame
)
from PyQt5.QtCore import Qt


class HelpDialog(QDialog):
    """
    title: pencere başlığı
    intro: üstte gösterilecek kısa açıklama (modülün ne işe yaradığı)
    sections: [(kategori_adı, [(kısayol, açıklama), ...]), ...] — boş bırakılabilir
    """

    def __init__(self, parent, title, intro="", sections=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 620)

        outer = QVBoxLayout(self)

        header = QLabel(title)
        header.setStyleSheet("font-size: 18px; font-weight: 700; padding: 2px 0;")
        outer.addWidget(header)

        if intro:
            intro_label = QLabel(intro)
            intro_label.setWordWrap(True)
            intro_label.setStyleSheet("font-size: 12px; color: gray; padding-bottom: 6px;")
            outer.addWidget(intro_label)

        # Kaydırılabilir içerik alanı (uzun listeler için)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(10)

        for category, rows in (sections or []):
            cat_label = QLabel(category)
            cat_label.setStyleSheet(
                "font-size: 13px; font-weight: 700; padding: 6px 0 2px 0;"
            )
            content_layout.addWidget(cat_label)

            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            content_layout.addWidget(line)

            for keys, desc in rows:
                row = QHBoxLayout()
                key_label = QLabel(keys)
                key_label.setStyleSheet(
                    "font-family: Consolas, monospace; font-weight: 600; "
                    "padding: 2px 6px; "
                )
                key_label.setMinimumWidth(150)
                key_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
                key_label.setWordWrap(True)
                desc_label = QLabel(desc)
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("font-size: 12px;")
                row.addWidget(key_label)
                row.addWidget(desc_label, stretch=1)
                content_layout.addLayout(row)

        content_layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        # Close düğmesi rejected verir; her ikisini de kapatmaya bağla
        buttons.clicked.connect(lambda _btn: self.accept())
        outer.addWidget(buttons)


def show_help(parent, title, intro="", sections=None):
    """Yardım penceresini modal olarak açar."""
    HelpDialog(parent, title, intro, sections).exec_()
