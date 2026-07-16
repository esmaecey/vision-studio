"""
Vision Studio — merkezi tema sistemi.
Renk paleti, tipografi ve tüm Qt bileşenleri için stil tanımları.
"""

# ---------- RENK PALETLERİ ----------

LIGHT = {
    "bg":            "#f5f7fa",   # ana arka plan
    "surface":       "#ffffff",   # kart/panel yüzeyi
    "surface_alt":   "#eef1f6",   # alternatif yüzey
    "border":        "#d8dee9",   # kenarlık
    "text":          "#1f2933",   # ana metin
    "text_muted":    "#6b7785",   # ikincil metin
    "primary":       "#2563eb",   # ana vurgu (mavi)
    "primary_hover": "#1d4ed8",
    "primary_text":  "#ffffff",
    "success":       "#16a34a",
    "warning":       "#f59e0b",
    "danger":        "#dc2626",
    "sidebar_bg":    "#ffffff",
    "sidebar_hover": "#eef2ff",
}

DARK = {
    "bg":            "#161a20",
    "surface":       "#1e242c",
    "surface_alt":   "#252c36",
    "border":        "#333b47",
    "text":          "#e6eaf0",
    "text_muted":    "#95a1b2",
    "primary":       "#3b82f6",
    "primary_hover": "#60a5fa",
    "primary_text":  "#ffffff",
    "success":       "#22c55e",
    "warning":       "#fbbf24",
    "danger":        "#ef4444",
    "sidebar_bg":    "#1a1f26",
    "sidebar_hover": "#252c36",
}


def build_stylesheet(palette):
    """Verilen renk paletinden tam bir Qt stylesheet üretir."""
    c = palette
    return f"""
    /* ---------- GENEL ---------- */
    QWidget {{
        background-color: {c['bg']};
        color: {c['text']};
        font-family: 'Segoe UI', 'Inter', sans-serif;
        font-size: 13px;
    }}

    /* ---------- GRUP KUTULARI (KART GÖRÜNÜMÜ) ---------- */
    QGroupBox {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 8px;
        margin-top: 14px;
        padding: 16px 12px 12px 12px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 6px;
        color: {c['text_muted']};
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* ---------- BUTONLAR ---------- */
    QPushButton {{
        background-color: {c['surface']};
        color: {c['text']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {c['surface_alt']};
        border-color: {c['primary']};
    }}
    QPushButton:pressed {{
        background-color: {c['border']};
    }}
    QPushButton:disabled {{
        color: {c['text_muted']};
        background-color: {c['surface_alt']};
        border-color: {c['border']};
    }}
    QPushButton:checked {{
        background-color: {c['primary']};
        color: {c['primary_text']};
        border-color: {c['primary']};
    }}

    /* ---------- GİRİŞ ALANLARI ---------- */
    QLineEdit, QTextEdit, QPlainTextEdit {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 7px 10px;
        selection-background-color: {c['primary']};
    }}
    QLineEdit:focus, QTextEdit:focus {{
        border-color: {c['primary']};
    }}
    QLineEdit:read-only {{
        background-color: {c['surface_alt']};
        color: {c['text_muted']};
    }}

    /* ---------- COMBO VE SPIN ---------- */
    QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 6px 10px;
        min-height: 20px;
    }}
    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {c['primary']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 22px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        selection-background-color: {c['primary']};
        selection-color: {c['primary_text']};
        outline: none;
    }}

    /* ---------- LİSTELER ---------- */
    QListWidget {{
        background-color: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        outline: none;
        padding: 4px;
    }}
    QListWidget::item {{
        padding: 7px 10px;
        border-radius: 4px;
    }}
    QListWidget::item:hover {{
        background-color: {c['surface_alt']};
    }}
    QListWidget::item:selected {{
        background-color: {c['primary']};
        color: {c['primary_text']};
    }}

    /* ---------- PROGRESS BAR ---------- */
    QProgressBar {{
        background-color: {c['surface_alt']};
        border: none;
        border-radius: 6px;
        height: 10px;
        text-align: center;
        color: {c['text']};
    }}
    QProgressBar::chunk {{
        background-color: {c['primary']};
        border-radius: 6px;
    }}

    /* ---------- SLIDER ---------- */
    QSlider::groove:horizontal {{
        background: {c['surface_alt']};
        height: 5px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {c['primary']};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {c['primary_hover']};
    }}
    QSlider::sub-page:horizontal {{
        background: {c['primary']};
        border-radius: 3px;
    }}

    /* ---------- CHECKBOX ---------- */
    QCheckBox {{
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 17px;
        height: 17px;
        border: 1px solid {c['border']};
        border-radius: 4px;
        background-color: {c['surface']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c['primary']};
        border-color: {c['primary']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {c['primary']};
    }}

    /* ---------- SCROLLBAR ---------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c['border']};
        border-radius: 5px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c['text_muted']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 10px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c['border']};
        border-radius: 5px;
        min-width: 30px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ---------- GRAPHICS VIEW (CANVAS) ---------- */
    QGraphicsView {{
        background-color: {c['surface_alt']};
        border: 1px solid {c['border']};
        border-radius: 6px;
    }}

    /* ---------- ETİKETLER ---------- */
    QLabel {{
        background: transparent;
    }}
    """


def sidebar_stylesheet(palette):
    """Sol navigation panel için özel stil."""
    c = palette
    return f"""
    QListWidget {{
        background-color: {c['sidebar_bg']};
        border: none;
        border-right: 1px solid {c['border']};
        outline: none;
        padding: 8px 6px;
        font-size: 14px;
    }}
    QListWidget::item {{
        padding: 12px 14px;
        border-radius: 8px;
        margin: 2px 0;
        color: {c['text_muted']};
    }}
    QListWidget::item:hover {{
        background-color: {c['sidebar_hover']};
        color: {c['text']};
    }}
    QListWidget::item:selected {{
        background-color: {c['primary']};
        color: {c['primary_text']};
        font-weight: 600;
    }}
    """


def logo_stylesheet(palette):
    c = palette
    return f"""
        background-color: {c['sidebar_bg']};
        color: {c['primary']};
        font-size: 19px;
        font-weight: 700;
        padding: 22px 16px;
        border-right: 1px solid {c['border']};
        border-bottom: 1px solid {c['border']};
        letter-spacing: 0.5px;
    """