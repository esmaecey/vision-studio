"""
Veri Doğrulama Arayüzü — etiket dosyalarındaki bozuklukları tarar ve düzeltir.
"""
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QFileDialog, QDoubleSpinBox, QProgressBar, QTextEdit, QGroupBox, QMessageBox
)
from PyQt5.QtCore import Qt

from .worker import ScanWorker, FixWorker
from config.project_state import project_state
from utils.dataset_paths import resolve_dataset_dirs, dir_has_images


ISSUE_LABELS = {
    "overflow": "Koordinat taşması (<0 veya >1)",
    "missing_label": "Eksik etiket (görsel var, .txt yok)",
    "empty_file": "Boş etiket dosyası",
    "inconsistent_kpt": "Tutarsız keypoint sayısı",
    "invalid_class": "Geçersiz sınıf kimliği",
    "malformed_line": "Bozuk satır (token sayısı hatalı)",
}


class DataValidationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.source_dir = None
        self.images_dir = None
        self.labels_dir = None
        self.worker = None
        self.scanned = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Veri Doğrulama")
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)

        desc = QLabel(
            "Etiket dosyalarındaki koordinat taşması, eksik/boş/bozuk etiket ve "
            "tutarsız keypoint gibi sorunları tespit eder; koordinat taşmalarını "
            "isteğe bağlı olarak düzeltir. 'Tara' yalnızca analiz eder, hiçbir şey "
            "değiştirmez."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: gray;")
        layout.addWidget(desc)

        # --- Kaynak seçimi ---
        src_group = QGroupBox("Kaynak (görsel klasörü ya da images/ + labels/ kökü)")
        src_layout = QHBoxLayout()
        self.src_line = QLineEdit()
        self.src_line.setReadOnly(True)
        btn_src = QPushButton("Kaynak Seç")
        btn_src.clicked.connect(self.select_source)
        src_layout.addWidget(self.src_line)
        src_layout.addWidget(btn_src)
        src_group.setLayout(src_layout)
        layout.addWidget(src_group)

        # --- Ayar + kontrol satırı ---
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Keypoint taşma eşiği:"))
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(1.0, 2.0)
        self.thr_spin.setSingleStep(0.01)
        self.thr_spin.setDecimals(2)
        self.thr_spin.setValue(1.05)
        self.thr_spin.setToolTip(
            "Keypoint koordinat taşması bu değerin altındaysa kırpılır; "
            "üstündeyse kırpılır ve görünürlük (v) 0 yapılır."
        )
        ctrl_layout.addWidget(self.thr_spin)
        ctrl_layout.addStretch()

        self.btn_scan = QPushButton("Tara")
        self.btn_scan.clicked.connect(self.start_scan)
        self.btn_fix = QPushButton("Düzelt")
        self.btn_fix.clicked.connect(self.start_fix)
        self.btn_fix.setEnabled(False)
        self.btn_cancel = QPushButton("İptal")
        self.btn_cancel.clicked.connect(self.cancel_worker)
        self.btn_cancel.setEnabled(False)
        ctrl_layout.addWidget(self.btn_scan)
        ctrl_layout.addWidget(self.btn_fix)
        ctrl_layout.addWidget(self.btn_cancel)
        layout.addLayout(ctrl_layout)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # --- Sonuç alanı ---
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setFontFamily("Consolas")
        layout.addWidget(self.result_area, stretch=1)

        self.setLayout(layout)

    # ---------- Kaynak ----------
    def select_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Kaynak Klasörü Seç")
        if not folder:
            return
        self.source_dir = folder
        self.src_line.setText(folder)
        self.images_dir, self.labels_dir = resolve_dataset_dirs(folder)
        self.scanned = False
        self.btn_fix.setEnabled(False)
        self.result_area.clear()
        self.result_area.append(f"Görsel klasörü: {self.images_dir}")
        self.result_area.append(f"Etiket klasörü: {self.labels_dir}")
        if not dir_has_images(self.images_dir):
            self.result_area.append(
                "UYARI: Seçilen klasörde doğrudan görsel yok; alt klasörler "
                "(train/val) taranacaktır."
            )

    # ---------- Tara ----------
    def start_scan(self):
        if not self.source_dir:
            self.result_area.append("HATA: Önce kaynak klasör seçin.")
            return
        nc = len(project_state.class_names) if project_state.has_project() else None

        self._set_running(True)
        self.result_area.clear()
        self.result_area.append("Tarama başladı...")
        if nc is None:
            self.result_area.append(
                "Not: Aktif proje yok — geçersiz sınıf kimliği kontrolü atlanacak "
                "(Settings'ten data.yaml seçin)."
            )

        self.worker = ScanWorker(self.source_dir, nc=nc)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.result_area.append)
        self.worker.finished_signal.connect(self.on_scan_done)
        self.worker.start()

    def on_scan_done(self, report):
        self._set_running(False)
        if not report:
            self.result_area.append("\nTarama tamamlanamadı.")
            return
        self.scanned = True
        self.result_area.append(self._format_scan_report(report))

        overflow = report["counts"]["overflow"]
        if report.get("cancelled"):
            self.result_area.append("\n(Tarama iptal edildi — kısmi sonuç.)")
        if overflow > 0:
            self.btn_fix.setEnabled(True)
            self.result_area.append(
                f"\n'Düzelt' ile {overflow} dosyadaki koordinat taşmaları "
                "düzeltilebilir (önce otomatik yedek alınır)."
            )
        else:
            self.btn_fix.setEnabled(False)
            self.result_area.append("\nOtomatik düzeltilecek koordinat taşması yok.")

    # ---------- Düzelt ----------
    def start_fix(self):
        if not self.source_dir or not self.scanned:
            return
        confirm = QMessageBox.question(
            self, "Düzeltmeyi Onayla",
            "Koordinat taşmaları düzeltilecek. Etiket klasörünün otomatik yedeği "
            "alınacak (labels_orig). Devam edilsin mi?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        self._set_running(True)
        self.result_area.append("\n--- Düzeltme başladı ---")
        self.worker = FixWorker(self.source_dir, threshold=self.thr_spin.value())
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.result_area.append)
        self.worker.finished_signal.connect(self.on_fix_done)
        self.worker.start()

    def on_fix_done(self, report):
        self._set_running(False)
        self.btn_fix.setEnabled(False)  # düzeltme yapıldı; yeniden taramak gerekir
        self.scanned = False
        if not report:
            self.result_area.append("\nDüzeltme tamamlanamadı.")
            return
        self.result_area.append(self._format_fix_report(report))
        self.result_area.append("\nDoğrulamak için tekrar 'Tara' çalıştırın.")

    # ---------- İptal / durum ----------
    def cancel_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.result_area.append("İptal isteniyor...")

    def _set_running(self, running):
        self.btn_scan.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        if running:
            self.btn_fix.setEnabled(False)
            self.progress.setValue(0)

    # ---------- Rapor biçimlendirme ----------
    def _format_scan_report(self, r):
        c = r["counts"]
        lines = ["", "=" * 48, "TARAMA ÖZETİ", "=" * 48]
        lines.append(f"Taranan görsel      : {r['images_total']}")
        lines.append(f"Okunan etiket dosyası: {r['labels_scanned']}")
        if r.get("used_subdirs"):
            lines.append(f"Alt klasörler       : {', '.join(r['used_subdirs'])}")
        lines.append(f"Sorunlu dosya sayısı : {r['files_with_issues']}")
        lines.append("")
        lines.append("Sorun türüne göre dağılım:")
        for key in ("overflow", "missing_label", "empty_file",
                    "inconsistent_kpt", "invalid_class", "malformed_line"):
            n = c[key]
            mark = "•" if n == 0 else "!"
            lines.append(f"  {mark} {ISSUE_LABELS[key]:<40}: {n}")

        if c["overflow"] > 0:
            lines.append("")
            lines.append(f"Taşan koordinat değeri sayısı: {r['overflow_value_count']}")
            lines.append(f"En büyük taşma: {r['max_overflow']:.4f} "
                         f"(koordinat değeri {r['max_overflow_value']:.4f})")

        if r["nc"] is None:
            lines.append("")
            lines.append("Not: Geçersiz sınıf kontrolü atlandı (aktif proje yok).")

        # Örnek dosyalar
        for key in ("overflow", "missing_label", "empty_file",
                    "inconsistent_kpt", "invalid_class", "malformed_line"):
            samples = r["samples"][key]
            if samples:
                lines.append("")
                lines.append(f"{ISSUE_LABELS[key]} — örnekler ({len(samples)}):")
                for s in samples:
                    lines.append(f"    {s}")

        # İstatistikler
        s = r["stats"]
        area = s["box_area"]
        vc = s["v_counts"]
        vp = s["v_pct"]
        lines.append("")
        lines.append("=" * 48)
        lines.append("VERİ SETİ İSTATİSTİKLERİ")
        lines.append("=" * 48)
        lines.append(f"Beklenen keypoint sayısı (K): {r['expected_K']}")
        lines.append(f"Toplam nesne (kutu)         : {s['total_objects']}")
        lines.append(f"Toplam keypoint             : {s['total_keypoints']}")
        lines.append("")
        lines.append("Görünürlük dağılımı:")
        lines.append(f"  v=0 (görünmez): {vc.get(0,0)} ({vp.get(0,0):.1f}%)")
        lines.append(f"  v=1 (örtük)   : {vc.get(1,0)} ({vp.get(1,0):.1f}%)")
        lines.append(f"  v=2 (görünür) : {vc.get(2,0)} ({vp.get(2,0):.1f}%)")
        lines.append("")
        lines.append("Kutu alanı (normalize, w×h):")
        lines.append(f"  min={area['min']:.5f}  ortanca={area['median']:.5f}  "
                     f"ort={area['mean']:.5f}  max={area['max']:.5f}")
        lines.append(f"Nesne başına görünür keypoint ort.: {s['avg_visible_kpts']:.2f}")
        return "\n".join(lines)

    def _format_fix_report(self, r):
        lines = ["", "=" * 48, "DÜZELTME ÖZETİ", "=" * 48]
        if r.get("cancelled"):
            lines.append("(Düzeltme iptal edildi — kısmi sonuç.)")
        lines.append(f"Yedek klasörü          : {r['backup']}")
        lines.append(f"Kullanılan eşik        : {r['threshold']:.2f}")
        lines.append(f"Düzeltilen dosya       : {r['files_fixed']}")
        lines.append(f"Kırpılan koordinat     : {r['values_clipped']}")
        lines.append(f"Görünmez yapılan nokta : {r['points_hidden']}")
        return "\n".join(lines)
