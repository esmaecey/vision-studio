"""
Veri Doğrulama worker'ları — etiket dosyalarını tarar (bozuklukları tespit eder)
ve isteğe bağlı olarak düzeltir. Uzun işlemler QThread'de; ilerleme + iptal destekli.

Taranan sorunlar:
  1. Koordinat taşması: kutu/keypoint koordinatı <0 veya >1
  2. Eksik etiket: görsel var ama .txt yok
  3. Boş etiket dosyası (satır yok)
  4. Tutarsız keypoint sayısı: beklenenden farklı K içeren satırlar
  5. Geçersiz sınıf kimliği: data.yaml sınıf sayısından büyük
  6. Bozuk satır: geçerli token desenine uymayan satırlar

Düzeltme (yalnızca koordinat taşması):
  - Kutu koordinatları 0-1'e kırpılır.
  - Keypoint koordinatı taşması eşiğin (varsayılan 1.05) altındaysa kırpılır;
    üstündeyse kırpılır ve görünürlük (v) 0 yapılır.
Diğer sorunlar yalnızca raporlanır.
"""
import os
import time
import shutil
import statistics
from collections import Counter
from PyQt5.QtCore import QThread, pyqtSignal

from utils.dataset_paths import resolve_dataset_dirs, collect_dataset_items


SAMPLE_LIMIT = 40  # rapora eklenecek örnek dosya sayısı (tür başına)


def classify_shape(n_tokens):
    """Token sayısına göre (kind, K): 'box'(5), 'pose'(5+3K) veya 'bad'."""
    if n_tokens == 5:
        return "box", 0
    if n_tokens > 5 and (n_tokens - 5) % 3 == 0:
        return "pose", (n_tokens - 5) // 3
    return "bad", None


def coord_overflow(v):
    """v [0,1] dışındaysa taşma miktarını (pozitif) döner, aksi halde 0."""
    if v < 0:
        return -v
    if v > 1:
        return v - 1
    return 0.0


def _clip01(v):
    return min(1.0, max(0.0, v))


def _read_nonempty(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]


class ScanWorker(QThread):
    """Etiketleri tarar; hiçbir şey değiştirmez, rapor üretir."""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, source_dir, nc=None):
        super().__init__()
        self.source_dir = source_dir
        self.nc = nc                    # data.yaml sınıf sayısı (None ise sınıf kontrolü atlanır)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            images_dir, labels_dir = resolve_dataset_dirs(self.source_dir)
            self.log.emit(f"Görsel klasörü: {images_dir}")
            self.log.emit(f"Etiket klasörü: {labels_dir}")

            if not images_dir or not os.path.isdir(images_dir):
                self.log.emit(f"HATA: Görsel klasörü bulunamadı: {images_dir}")
                self.finished_signal.emit({})
                return

            items, used_subdirs = collect_dataset_items(images_dir, labels_dir)
            if not items:
                self.log.emit("HATA: Görsel bulunamadı.")
                self.finished_signal.emit({})
                return

            total = len(items)
            missing, empty = [], []
            overflow_files, malformed_files = set(), set()
            invalid_class_files = set()
            line_ks = []            # (rel, K) — tutarlılık için
            v_counts = {0: 0, 1: 0, 2: 0}
            total_objects = 0
            total_keypoints = 0
            visible_kpt_total = 0
            box_areas = []
            max_over = 0.0
            max_over_val = 0.0
            overflow_value_count = 0
            labels_scanned = 0
            cancelled = False
            last_pct = -1

            for i, (img, lbl, name) in enumerate(items):
                if self._cancel:
                    cancelled = True
                    break

                if not lbl or not os.path.exists(lbl):
                    missing.append(name)
                else:
                    rel = os.path.relpath(lbl, labels_dir)
                    labels_scanned += 1
                    lines = _read_nonempty(lbl)
                    if not lines:
                        empty.append(rel)
                    for line in lines:
                        parts = line.split()
                        n = len(parts)
                        kind, K = classify_shape(n)
                        if kind == "bad":
                            malformed_files.add(rel)
                            continue
                        try:
                            cid = int(float(parts[0]))
                            cx, cy, bw, bh = map(float, parts[1:5])
                        except ValueError:
                            malformed_files.add(rel)
                            continue

                        if self.nc is not None and (cid < 0 or cid >= self.nc):
                            invalid_class_files.add(rel)

                        total_objects += 1
                        box_areas.append(max(0.0, bw) * max(0.0, bh))
                        for v in (cx, cy, bw, bh):
                            o = coord_overflow(v)
                            if o > 0:
                                overflow_files.add(rel)
                                overflow_value_count += 1
                                if o > max_over:
                                    max_over, max_over_val = o, v
                        line_ks.append((rel, K))

                        if kind == "pose":
                            total_keypoints += K
                            vals = parts[5:]
                            vis = 0
                            bad_kp = False
                            for j in range(K):
                                try:
                                    x = float(vals[3 * j])
                                    y = float(vals[3 * j + 1])
                                    vv = int(float(vals[3 * j + 2]))
                                except ValueError:
                                    bad_kp = True
                                    break
                                v_counts[vv] = v_counts.get(vv, 0) + 1
                                if vv > 0:
                                    vis += 1
                                for c in (x, y):
                                    o = coord_overflow(c)
                                    if o > 0:
                                        overflow_files.add(rel)
                                        overflow_value_count += 1
                                        if o > max_over:
                                            max_over, max_over_val = o, c
                            if bad_kp:
                                malformed_files.add(rel)
                            else:
                                visible_kpt_total += vis

                pct = int((i + 1) / total * 100)
                if pct != last_pct:
                    self.progress.emit(pct)
                    last_pct = pct

            # Beklenen keypoint sayısı: geçerli satırların en sık K değeri
            kcount = Counter(K for _, K in line_ks)
            expected_K = kcount.most_common(1)[0][0] if kcount else 0
            inconsistent_files = set(rel for rel, K in line_ks if K != expected_K)

            issue_files = set(missing) | set(empty) | overflow_files | \
                inconsistent_files | invalid_class_files | malformed_files

            area_stats = self._area_stats(box_areas)
            total_v = sum(v_counts.values())
            v_pct = {k: (100.0 * n / total_v if total_v else 0.0) for k, n in v_counts.items()}

            report = {
                "mode": "scan",
                "cancelled": cancelled,
                "images_total": total,
                "labels_scanned": labels_scanned,
                "used_subdirs": used_subdirs,
                "nc": self.nc,
                "expected_K": expected_K,
                "labels_dir": labels_dir,
                "counts": {
                    "overflow": len(overflow_files),
                    "missing_label": len(missing),
                    "empty_file": len(empty),
                    "inconsistent_kpt": len(inconsistent_files),
                    "invalid_class": len(invalid_class_files),
                    "malformed_line": len(malformed_files),
                },
                "files_with_issues": len(issue_files),
                "samples": {
                    "overflow": sorted(overflow_files)[:SAMPLE_LIMIT],
                    "missing_label": sorted(missing)[:SAMPLE_LIMIT],
                    "empty_file": sorted(empty)[:SAMPLE_LIMIT],
                    "inconsistent_kpt": sorted(inconsistent_files)[:SAMPLE_LIMIT],
                    "invalid_class": sorted(invalid_class_files)[:SAMPLE_LIMIT],
                    "malformed_line": sorted(malformed_files)[:SAMPLE_LIMIT],
                },
                "overflow_value_count": overflow_value_count,
                "max_overflow": max_over,
                "max_overflow_value": max_over_val,
                "stats": {
                    "total_objects": total_objects,
                    "total_keypoints": total_keypoints,
                    "v_counts": v_counts,
                    "v_pct": v_pct,
                    "box_area": area_stats,
                    "avg_visible_kpts": (visible_kpt_total / total_objects) if total_objects else 0.0,
                },
            }
            self.finished_signal.emit(report)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit({})

    @staticmethod
    def _area_stats(areas):
        if not areas:
            return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "min": min(areas),
            "median": statistics.median(areas),
            "mean": statistics.mean(areas),
            "max": max(areas),
        }


class FixWorker(QThread):
    """Koordinat taşmalarını düzeltir. Düzeltmeden önce etiket klasörünü yedekler."""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)

    def __init__(self, source_dir, threshold=1.05):
        super().__init__()
        self.source_dir = source_dir
        self.threshold = threshold
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            images_dir, labels_dir = resolve_dataset_dirs(self.source_dir)
            if not labels_dir or not os.path.isdir(labels_dir):
                self.log.emit(f"HATA: Etiket klasörü bulunamadı: {labels_dir}")
                self.finished_signal.emit({})
                return

            # --- Otomatik yedek ---
            backup = labels_dir.rstrip("/\\") + "_orig"
            if os.path.exists(backup):
                backup = labels_dir.rstrip("/\\") + "_orig_" + time.strftime("%Y%m%d_%H%M%S")
            shutil.copytree(labels_dir, backup)
            self.log.emit(f"Yedek alındı: {backup}")

            items, _ = collect_dataset_items(images_dir, labels_dir)
            total = len(items) or 1
            files_fixed = 0
            values_clipped = 0
            points_hidden = 0
            cancelled = False
            last_pct = -1
            thr = self.threshold
            low = 1.0 - thr   # simetrik alt sınır (örn. 1.05 -> -0.05)

            for i, (img, lbl, name) in enumerate(items):
                if self._cancel:
                    cancelled = True
                    break
                if lbl and os.path.exists(lbl):
                    changed, nclip, nhide = self._fix_file(lbl, thr, low)
                    if changed:
                        files_fixed += 1
                        values_clipped += nclip
                        points_hidden += nhide

                pct = int((i + 1) / total * 100)
                if pct != last_pct:
                    self.progress.emit(pct)
                    last_pct = pct

            report = {
                "mode": "fix",
                "cancelled": cancelled,
                "backup": backup,
                "files_fixed": files_fixed,
                "values_clipped": values_clipped,
                "points_hidden": points_hidden,
                "threshold": thr,
            }
            self.finished_signal.emit(report)

        except Exception as e:
            self.log.emit(f"HATA: {str(e)}")
            self.finished_signal.emit({})

    @staticmethod
    def _fix_file(path, thr, low):
        """Tek dosyayı düzeltir. Dönüş: (changed, clipped_count, hidden_count)."""
        lines = _read_nonempty(path)
        new_lines = []
        changed = False
        clipped = 0
        hidden = 0

        for line in lines:
            parts = line.split()
            n = len(parts)
            kind, K = classify_shape(n)
            if kind == "bad":
                new_lines.append(line)   # bozuk satıra dokunma (yalnızca raporlanır)
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                new_lines.append(line)
                continue

            # Kutu koordinatları: 0-1'e kırp
            box_vals = [cx, cy, bw, bh]
            fixed_box = []
            for v in box_vals:
                nv = _clip01(v)
                if nv != v:
                    clipped += 1
                    changed = True
                fixed_box.append(nv)

            out = [str(cid)] + [f"{v:.6f}" for v in fixed_box]

            if kind == "pose":
                vals = parts[5:]
                ok = True
                kp_out = []
                for j in range(K):
                    try:
                        x = float(vals[3 * j])
                        y = float(vals[3 * j + 1])
                        v = int(float(vals[3 * j + 2]))
                    except ValueError:
                        ok = False
                        break
                    # Eşik üstü taşma -> nokta gerçekten kadraj dışı: kırp + v=0
                    far = (x > thr or x < low or y > thr or y < low)
                    nx, ny = _clip01(x), _clip01(y)
                    if nx != x:
                        clipped += 1
                        changed = True
                    if ny != y:
                        clipped += 1
                        changed = True
                    if far and v > 0:
                        v = 0
                        hidden += 1
                        changed = True
                    kp_out += [f"{nx:.6f}", f"{ny:.6f}", str(v)]
                if not ok:
                    new_lines.append(line)   # bozuk keypoint -> dokunma
                    continue
                out += kp_out

            new_lines.append(" ".join(out))

        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines) + "\n")
        return changed, clipped, hidden
