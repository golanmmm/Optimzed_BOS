#!/usr/bin/env python3
"""
Ultrasound GUI — Calibration → Line FFT → THEN Filtering (1D along line or 2D wedge)

Flow (as requested):
  1) User loads image (UNFILTERED view so clicks are clear)
  2) User selects 2 points for px/mm calibration
  3) User selects a line (2 points)
  4) We compute 1D FFT along that line (c = 1497 m/s for water @25°C by default)
  5) ONLY AFTER the FFT: apply a frequency‑range filter to the image.
     You can choose:
        • "1D line‑by‑line" → rotates image so the selected line is horizontal,
          band‑passes each row (1‑D FFT along the row), rotates back.
        • "2D FFT wedge" → prior directional wedge mask in the 2‑D spectrum.

Notes:
  • Default speed is 1497 m/s (water @25°C). You can change it in Calibration.
  • Default filter method = 1D line‑by‑line; filtering is DISABLED until the line FFT is done.
  • Min/Max f [MHz] + Taper% control the passband in both methods.

Requirements:
    pip install numpy matplotlib scikit-image PyQt5
"""

from __future__ import annotations
import sys, os
import numpy as np
from dataclasses import dataclass
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QDoubleSpinBox, QGroupBox, QPushButton, QLabel, QFileDialog, QComboBox,
    QSlider, QSplitter, QMessageBox, QDialog, QCheckBox
)
from PyQt5.QtWidgets import QInputDialog
from skimage import io as skio
from skimage.color import rgb2gray
from skimage.measure import profile_line
from skimage.transform import rotate

@dataclass
class Calibration:
    mm_per_pixel: float = 0.05
    speed_ms: float = 1497.0  # water @25°C
    standing_wave: bool = True   # True → k=2/λ, False → k=1/λ

class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(constrained_layout=True)
        self.ax = fig.add_subplot(111)
        self.ax.set_axis_off()
        super().__init__(fig)
        self.ax.set_aspect('equal', adjustable='datalim')

class FFTDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('1D Line Profile & Spectrum')
        self.resize(820, 540)
        self.fig = Figure(constrained_layout=True)
        self.ax_sig = self.fig.add_subplot(211)
        self.ax_fft = self.fig.add_subplot(212)
        self.canvas = FigureCanvas(self.fig)
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas)
        self.lbl = QLabel('λ = –,  f = –')
        lay.addWidget(self.lbl)
    def update_plots(self, s_mm: np.ndarray, signal: np.ndarray, k_mm: np.ndarray, mag: np.ndarray, lam_mm: float | None, f_mhz: float | None):
        self.ax_sig.clear(); self.ax_fft.clear()
        self.ax_sig.plot(s_mm, signal, lw=1)
        self.ax_sig.set_xlabel('Distance [mm]')
        self.ax_sig.set_ylabel('Intensity [a.u.]')
        self.ax_sig.grid(True, alpha=0.3)
        self.ax_fft.plot(k_mm, mag, lw=1)
        self.ax_fft.set_xlabel('k [cycles/mm]')
        self.ax_fft.set_ylabel('|FFT|')
        self.ax_fft.set_xlim(left=0)
        self.ax_fft.grid(True, alpha=0.3)
        self.canvas.draw_idle()
        if lam_mm is not None and f_mhz is not None:
            self.lbl.setText(f"λ ≈ {lam_mm:.4f} mm    |    f ≈ {f_mhz:.3f} MHz")
        else:
            self.lbl.setText('λ = –,  f = –')

class BeamSimDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Beam Simulation (x–z slice)')
        self.resize(900, 720)
        self.fig = Figure(constrained_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(60)
        self.lbl = QLabel('Display ≥ 60% of peak')
        self.slider.valueChanged.connect(self._on_thresh)
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas)
        row = QWidget(); rh = QHBoxLayout(row); rh.setContentsMargins(0,0,0,0)
        rh.addWidget(self.lbl); rh.addWidget(self.slider); rh.addStretch(1)
        lay.addWidget(row)
        self.I = None; self.im = None; self.extent = None
    def set_data(self, I_norm: np.ndarray, x_mm: np.ndarray, z_mm: np.ndarray):
        self.I = I_norm
        self.extent = [x_mm.min(), x_mm.max(), z_mm.min(), z_mm.max()]
        self.ax.clear()
        self.im = self.ax.imshow(self.I, origin='upper', cmap='gray', extent=self.extent, aspect='auto', interpolation='nearest')
        self.ax.set_xlabel('x [mm]')
        self.ax.set_ylabel('z [mm]')
        self.canvas.draw_idle()
        self._on_thresh()
    def _on_thresh(self):
        if self.I is None: return
        t = self.slider.value()
        self.lbl.setText(f'Display ≥ {t}% of peak')
        thr = np.clip(t/100.0, 0.0, 1.0)
        M = np.where(self.I >= thr, self.I, np.nan)
        self.im.set_data(M)
        self.canvas.draw_idle()

class UltrasoundFilterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ultrasound GUI — Calibrate → Line → FFT → Filter')
        self.resize(1480, 900)
        self.orig_img: np.ndarray | None = None
        self.fft_cache: np.ndarray | None = None
        self._KX = None; self._KY = None; self._R = None; self._TH = None
        self.calib = Calibration()
        self.mode: str = 'idle'
        self.pending_pts: list[tuple[float, float]] = []
        self.line_pts: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.beam_pts: tuple[tuple[float, float], tuple[float, float]] | None = None
        self.D_mm: float | None = None
        self.lam_mm: float | None = None
        self.f_mhz: float | None = None
        self.filter_enabled: bool = False  # stays False until line FFT is done
        self._dialogs = []
        self.canvas = MplCanvas()
        self.cid_click = self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        right_panel = self._build_right_panel()
        splitter = QSplitter(Qt.Horizontal)
        left_wrap = QWidget(); lw = QVBoxLayout(left_wrap); lw.setContentsMargins(4,4,4,4); lw.addWidget(self.canvas)
        splitter.addWidget(left_wrap); splitter.addWidget(right_panel)
        splitter.setCollapsible(0, False); splitter.setCollapsible(1, False)
        splitter.setSizes([1040, 440])
        root = QHBoxLayout(self); root.addWidget(splitter)
        self._load_synthetic()
        self._update_everything()

    def _build_right_panel(self) -> QWidget:
        panel = QWidget(); v = QVBoxLayout(panel); v.setContentsMargins(8,8,8,8); v.setSpacing(10)

        # Data
        g_data = QGroupBox('Data'); h_data = QHBoxLayout(g_data)
        self.btn_load = QPushButton('Load Image…'); self.btn_load.clicked.connect(self._on_load_image)
        self.btn_reset = QPushButton('Reset View'); self.btn_reset.clicked.connect(self._reset_view)
        h_data.addWidget(self.btn_load); h_data.addWidget(self.btn_reset); h_data.addStretch(1)

        # Calibration
        g_cal = QGroupBox('Calibration'); f_cal = QFormLayout(g_cal)
        self.spin_mmpp = QDoubleSpinBox(); self.spin_mmpp.setRange(1e-6, 1e3); self.spin_mmpp.setSingleStep(0.001); self.spin_mmpp.setDecimals(6); self.spin_mmpp.setValue(self.calib.mm_per_pixel); self.spin_mmpp.valueChanged.connect(self._on_calib_changed)
        self.spin_speed = QDoubleSpinBox(); self.spin_speed.setRange(100.0, 4000.0); self.spin_speed.setSingleStep(1.0); self.spin_speed.setDecimals(1); self.spin_speed.setValue(self.calib.speed_ms); self.spin_speed.valueChanged.connect(self._on_calib_changed)
        self.combo_pattern = QComboBox(); self.combo_pattern.addItems(['Standing wave (2/λ)', 'Traveling wave (1/λ)']); self.combo_pattern.currentIndexChanged.connect(self._on_calib_changed)
        f_cal.addRow('mm / pixel', self.spin_mmpp)
        f_cal.addRow('Speed (m/s)', self.spin_speed)
        f_cal.addRow('Pattern → k(λ)', self.combo_pattern)

        # Filtering settings (gated after FFT)
        g_mhz = QGroupBox('Filtering (applied ONLY after line FFT)'); f_mhz = QFormLayout(g_mhz)
        self.spin_fmin = QDoubleSpinBox(); self.spin_fmin.setRange(0.1, 30.0); self.spin_fmin.setSingleStep(0.1); self.spin_fmin.setDecimals(2); self.spin_fmin.setValue(1.0); self.spin_fmin.valueChanged.connect(self._on_freq_changed)
        self.spin_fmax = QDoubleSpinBox(); self.spin_fmax.setRange(0.1, 30.0); self.spin_fmax.setSingleStep(0.1); self.spin_fmax.setDecimals(2); self.spin_fmax.setValue(10.0); self.spin_fmax.valueChanged.connect(self._on_freq_changed)
        self.slider_taper = QSlider(Qt.Horizontal); self.slider_taper.setRange(0, 40); self.slider_taper.setValue(15); self.slider_taper.valueChanged.connect(self._on_freq_changed)
        self.lbl_taper = QLabel('15 %')
        self.chk_enable_after_fft = QCheckBox('Enable filter only after FFT'); self.chk_enable_after_fft.setChecked(True); self.chk_enable_after_fft.stateChanged.connect(self._on_freq_changed)
        self.combo_method = QComboBox(); self.combo_method.addItems(['1D line‑by‑line (rotate + per‑row FFT)', '2D FFT wedge']); self.combo_method.currentIndexChanged.connect(self._on_freq_changed)

        # Direction controls (used by both methods)
        self.chk_use_lineang = QCheckBox('Use line angle θ'); self.chk_use_lineang.setChecked(True); self.chk_use_lineang.stateChanged.connect(self._on_freq_changed)
        self.chk_perp = QCheckBox('Perpendicular (⊥)'); self.chk_perp.setChecked(False); self.chk_perp.stateChanged.connect(self._on_freq_changed)
        self.chk_auto_theta = QCheckBox('Auto‑angle (2D spectrum)'); self.chk_auto_theta.setChecked(False); self.chk_auto_theta.stateChanged.connect(self._on_freq_changed)
        self.spin_theta = QDoubleSpinBox(); self.spin_theta.setRange(-180.0, 180.0); self.spin_theta.setDecimals(1); self.spin_theta.setSingleStep(1.0); self.spin_theta.setValue(0.0); self.spin_theta.valueChanged.connect(self._on_freq_changed)
        self.spin_halfdeg = QDoubleSpinBox(); self.spin_halfdeg.setRange(1.0, 90.0); self.spin_halfdeg.setDecimals(1); self.spin_halfdeg.setSingleStep(1.0); self.spin_halfdeg.setValue(20.0); self.spin_halfdeg.valueChanged.connect(self._on_freq_changed)
        self.lbl_theta_src = QLabel('θ: —')

        f_mhz.addRow('Min f (MHz)', self.spin_fmin)
        f_mhz.addRow('Max f (MHz)', self.spin_fmax)
        f_mhz.addRow('Edge taper', self._row(self.slider_taper, self.lbl_taper))
        f_mhz.addRow('Method', self.combo_method)
        f_mhz.addRow(self.chk_enable_after_fft)
        f_mhz.addRow(self.chk_use_lineang)
        f_mhz.addRow(self._row(self.chk_perp, QLabel(' (check if your line is along the fringes)')))
        f_mhz.addRow(self.chk_auto_theta)
        f_mhz.addRow('Angle θ (°)', self.spin_theta)
        f_mhz.addRow('Half‑angle (°) [2D only]', self.spin_halfdeg)
        f_mhz.addRow('θ source', self.lbl_theta_src)

        # Tools
        g_tools = QGroupBox('Tools'); v_tools = QVBoxLayout(g_tools)
        self.btn_calib = QPushButton('Calibrate px↔mm (2 clicks)'); self.btn_calib.clicked.connect(self._start_calibrate)
        self.btn_line = QPushButton('Select line for 1D FFT'); self.btn_line.clicked.connect(self._start_line_fft)
        self.btn_diam = QPushButton('Measure beam diameter (2 clicks)'); self.btn_diam.clicked.connect(self._start_beam_diam)
        self.btn_sim = QPushButton('Simulate beam…'); self.btn_sim.clicked.connect(self._simulate_beam)
        v_tools.addWidget(self.btn_calib); v_tools.addWidget(self.btn_line); v_tools.addWidget(self.btn_diam); v_tools.addWidget(self.btn_sim)

        # Processing
        g_proc = QGroupBox('Processing'); v_proc = QVBoxLayout(g_proc)
        self.btn_apply = QPushButton('Apply / Refresh'); self.btn_apply.clicked.connect(self._update_everything)
        v_proc.addWidget(self.btn_apply); v_proc.addStretch(1)

        # Metrics
        g_met = QGroupBox('Metrics'); f_met = QFormLayout(g_met)
        self.lbl_mmpp = QLabel('—'); self.lbl_D = QLabel('—'); self.lbl_lambda = QLabel('—'); self.lbl_freq = QLabel('—'); self.lbl_rayleigh = QLabel('—')
        f_met.addRow('mm / pixel', self.lbl_mmpp)
        f_met.addRow('Beam D [mm]', self.lbl_D)
        f_met.addRow('λ from line [mm]', self.lbl_lambda)
        f_met.addRow('f from line [MHz]', self.lbl_freq)
        f_met.addRow('Rayleigh N [mm]', self.lbl_rayleigh)

        v.addWidget(g_data); v.addWidget(g_cal); v.addWidget(g_mhz); v.addWidget(g_tools); v.addWidget(g_proc); v.addWidget(g_met); v.addStretch(1)
        return panel

    @staticmethod
    def _row(*widgets):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0)
        for x in widgets: h.addWidget(x)
        h.addStretch(1)
        return w

    # --------------------- IO / setup ---------------------
    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open image', os.getcwd(), 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not path:
            return
        try:
            img = skio.imread(path)
            if img.ndim == 3:
                img = rgb2gray(img)
            img = img.astype(np.float64)
            if img.max() > 1.0:
                img = img / 255.0
            self._set_image(img)
            # Reset gating so the user sees a clean, unfiltered image for clicks
            self.filter_enabled = False
        except Exception as e:
            QMessageBox.critical(self, 'Load error', f"Could not open image:{e}")

    def _load_synthetic(self):
        # A synthetic image with two spatial tones + Gaussian envelope
        h, w = 512, 768
        y, x = np.mgrid[0:h, 0:w]
        def mk_wave(f_MHz, calib: Calibration):
            f_hz = f_MHz * 1e6
            lam_mm = (calib.speed_ms * 1000.0) / f_hz
            factor = 2.0 if calib.standing_wave else 1.0
            k_mm = factor / lam_mm
            k_cpp = k_mm * calib.mm_per_pixel
            return 0.5 + 0.25 * np.sin(2*np.pi * (x * k_cpp))
        base = mk_wave(2.5, self.calib) + mk_wave(5.0, self.calib)
        base = np.clip(base, 0.0, 1.0)
        cy, cx = h/2, w/2
        rr2 = ((y-cy)**2 + (x-cx)**2) / (0.45*min(h,w))**2
        mask = np.exp(-rr2)
        img = (base*mask + 0.02*np.random.randn(h, w)).astype(np.float64)
        img = (img - img.min()) / (img.max() - img.min())
        self._set_image(img)

    def _set_image(self, img: np.ndarray):
        self.orig_img = img
        self._prepare_fft()
        self._update_everything()

    def _prepare_fft(self):
        if self.orig_img is None:
            self.fft_cache = None; self._KX = self._KY = self._R = self._TH = None; return
        img = self.orig_img
        self.fft_cache = np.fft.fftshift(np.fft.fft2(img))
        h, w = img.shape
        ky = np.fft.fftfreq(h)  # cycles/pixel
        kx = np.fft.fftfreq(w)
        KX, KY = np.meshgrid(kx, ky)
        self._KX = KX
        self._KY = KY
        self._R = np.sqrt(KX**2 + KY**2)
        self._TH = np.arctan2(KY, KX)

    # --------------------- parameter changes ---------------------
    def _on_calib_changed(self):
        self.calib.mm_per_pixel = float(self.spin_mmpp.value())
        self.calib.speed_ms = float(self.spin_speed.value())
        self.calib.standing_wave = (self.combo_pattern.currentIndex() == 0)
        self._update_metrics()
        self._update_everything()

    def _on_freq_changed(self, *_):
        try:
            fmin = float(self.spin_fmin.value()); fmax = float(self.spin_fmax.value())
            if fmin > fmax:
                sender = self.sender()
                if sender is self.spin_fmin: self.spin_fmax.setValue(fmin)
                else: self.spin_fmin.setValue(fmax)
            self.lbl_taper.setText(f'{self.slider_taper.value()} %')
            self._update_everything()
        except Exception:
            return

    def _reset_view(self):
        self.canvas.ax.set_xlim(auto=True); self.canvas.ax.set_ylim(auto=True)
        self.canvas.figure.canvas.draw_idle()

    # --------------------- math helpers ---------------------
    def _mhz_band_to_cycles_per_pixel(self, fmin_mhz: float, fmax_mhz: float) -> tuple[float, float]:
        mmpp = max(1e-12, float(self.calib.mm_per_pixel))
        c_ms = max(1e-12, float(self.calib.speed_ms))
        factor = 2.0 if self.calib.standing_wave else 1.0
        def single(f_mhz: float) -> float:
            f_hz = f_mhz * 1e6
            lam_mm = (c_ms * 1000.0) / f_hz
            k_mm = factor / lam_mm
            return k_mm * mmpp  # cycles/pixel
        kmin_cpp = single(fmin_mhz); kmax_cpp = single(fmax_mhz)
        if kmin_cpp > kmax_cpp: kmin_cpp, kmax_cpp = kmax_cpp, kmin_cpp
        return kmin_cpp, kmax_cpp

    @staticmethod
    def _raised_cosine_taper(x: np.ndarray, lo: float, hi: float, pct: float) -> np.ndarray:
        lo, hi = float(lo), float(hi)
        if hi <= lo: return np.zeros_like(x)
        width = hi - lo
        t = np.clip(pct/100.0, 0.0, 0.9)
        ramp = 0.5 * t * width
        a0, a1 = lo, lo + ramp; b0, b1 = hi - ramp, hi
        m = np.zeros_like(x)
        m = np.where((x >= a1) & (x <= b0), 1.0, m)
        idx = (x >= a0) & (x < a1)
        if np.any(idx):
            xi = (x[idx] - a0) / max(1e-12, (a1 - a0))
            m[idx] = 0.5 * (1 - np.cos(np.pi * xi))
        idx = (x > b0) & (x <= b1)
        if np.any(idx):
            xi = 1 - (x[idx] - b0) / max(1e-12, (b1 - b0))
            m[idx] = 0.5 * (1 - np.cos(np.pi * xi))
        return m

    @staticmethod
    def _angdiff(a: np.ndarray, b: float) -> np.ndarray:
        """Smallest signed angular difference wrap to [-π, π]."""
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def _wedge_mask(self, theta_grid: np.ndarray, theta0_deg: float, half_deg: float, pct: float) -> np.ndarray:
        """Raised‑cosine wedge mask centered at ±theta0 with half width half_deg.
        Returns values in [0,1]."""
        th0 = np.deg2rad(theta0_deg)
        half = np.deg2rad(max(1e-3, half_deg))
        t = np.clip(pct/100.0, 0.0, 0.9)
        ramp = t * half
        flat = max(0.0, half - ramp)

        def single(center):
            d = np.abs(self._angdiff(theta_grid, center))  # [0, π]
            m = np.zeros_like(d)
            # flat 1 region
            m = np.where(d <= flat, 1.0, m)
            # ramp down to 0 between flat..(flat+ramp)
            idx = (d > flat) & (d <= flat + ramp)
            if np.any(idx):
                xi = 1.0 - (d[idx] - flat) / max(1e-12, ramp)
                m[idx] = 0.5 * (1 - np.cos(np.pi * xi))
            return m

        m_pos = single(+th0)
        m_neg = single(-th0)
        return np.maximum(m_pos, m_neg)

    def _line_angle_deg(self, use_perp: bool=False) -> float | None:
        if self.line_pts is None:
            return None
        (x1, y1), (x2, y2) = self.line_pts
        dx = x2 - x1; dy = y2 - y1
        ang = np.degrees(np.arctan2(dy, dx))
        if use_perp:
            ang += 90.0
        ang = (ang + 180.0) % 360.0 - 180.0
        return float(ang)

    def _estimate_theta_deg(self, kmin_cpp: float) -> float:
        if self.fft_cache is None or self._R is None or self._TH is None:
            return 0.0
        M = np.abs(self.fft_cache)
        mask = (self._R >= max(kmin_cpp*0.8, 0.01))
        if not np.any(mask):
            return 0.0
        idx = np.argmax(M * mask)
        iy, ix = np.unravel_index(int(idx), M.shape)
        th = float(self._TH[iy, ix])
        return np.rad2deg(th)

    # --------------------- Filtering backends ---------------------
    def _filter_1d_linebyline(self, fmin, fmax, taper_pct) -> np.ndarray:
        """Rotate image so the selected line is horizontal, band‑pass each row via 1‑D FFT,
        rotate back, and center‑crop to original size."""
        if self.orig_img is None:
            return np.zeros((512,512))
        angle = 0.0
        src = 'manual'
        if self.chk_use_lineang.isChecked() and (self.line_pts is not None):
            a = self._line_angle_deg(use_perp=self.chk_perp.isChecked())
            if a is not None:
                angle = a
                src = 'line'
        elif self.chk_auto_theta.isChecked():
            # Use 2D spectrum to estimate
            kmin_cpp, _ = self._mhz_band_to_cycles_per_pixel(fmin, fmax)
            angle = self._estimate_theta_deg(kmin_cpp)
            src = 'auto'
        else:
            angle = float(self.spin_theta.value())
            src = 'manual'
        self.lbl_theta_src.setText(f"θ_{src}: {angle:+.1f}°")

        # Rotate so the selected line becomes (approximately) horizontal
        img_rot = rotate(self.orig_img, -angle, resize=True, order=1, mode='reflect', preserve_range=True)
        img_rot = img_rot.astype(np.float64)
        H, W = img_rot.shape

        # 1‑D FFT along rows
        F = np.fft.rfft(img_rot, axis=1)
        freqs_cpp = np.fft.rfftfreq(W, d=1.0)
        kmin_cpp, kmax_cpp = self._mhz_band_to_cycles_per_pixel(fmin, fmax)
        mask = self._raised_cosine_taper(freqs_cpp, kmin_cpp, kmax_cpp, taper_pct)
        F_filtered = F * mask[None, :]
        rec_rot = np.fft.irfft(F_filtered, n=W, axis=1)

        # Rotate back and center‑crop to original size
        rec = rotate(rec_rot, angle, resize=True, order=1, mode='reflect', preserve_range=True)
        rec = rec.astype(np.float64)
        rec = self._center_crop_like(rec, self.orig_img)
        return rec

    def _center_crop_like(self, arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
        h, w = ref.shape
        H, W = arr.shape
        y0 = max(0, (H - h)//2); x0 = max(0, (W - w)//2)
        out = arr[y0:y0+h, x0:x0+w]
        if out.shape != ref.shape:
            # If rotated result is smaller, pad reflectively
            pad_y = max(0, h - out.shape[0]); pad_x = max(0, w - out.shape[1])
            out = np.pad(out, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2)), mode='reflect')
            out = out[:h, :w]
        return out

    def _filter_2d_wedge(self, fmin, fmax, taper_pct) -> np.ndarray:
        if self.orig_img is None or self.fft_cache is None or self._R is None:
            return np.zeros((512, 512))
        kmin_cpp, kmax_cpp = self._mhz_band_to_cycles_per_pixel(fmin, fmax)
        Mr = self._raised_cosine_taper(self._R, kmin_cpp, kmax_cpp, taper_pct)
        # Angle
        theta0 = float(self.spin_theta.value()); src='manual'
        if self.chk_use_lineang.isChecked() and (self.line_pts is not None):
            ang = self._line_angle_deg(use_perp=self.chk_perp.isChecked())
            if ang is not None:
                theta0 = ang; src='line'
        elif self.chk_auto_theta.isChecked():
            theta0 = self._estimate_theta_deg(kmin_cpp); src='auto'
        self.lbl_theta_src.setText(f"θ_{src}: {theta0:+.1f}°")
        Mt = self._wedge_mask(self._TH, theta0, float(self.spin_halfdeg.value()), taper_pct)
        M = Mr * Mt
        F = self.fft_cache * M
        rec = np.fft.ifft2(np.fft.ifftshift(F)).real
        return rec

    # --------------------- draw / GUI ---------------------
    def _update_everything(self):
        if self.orig_img is None:
            return
        try:
            # Decide whether to filter yet
            do_filter = self.filter_enabled or (not self.chk_enable_after_fft.isChecked())
            if not do_filter:
                out = self.orig_img
            else:
                fmin = float(self.spin_fmin.value()); fmax = float(self.spin_fmax.value()); taper = float(self.slider_taper.value())
                if self.combo_method.currentIndex() == 0:
                    out = self._filter_1d_linebyline(fmin, fmax, taper)
                else:
                    out = self._filter_2d_wedge(fmin, fmax, taper)
            self._draw_image(out)
        except Exception as e:
            QMessageBox.critical(self, 'Processing error', str(e))

    def _draw_image(self, img: np.ndarray):
        ax = self.canvas.ax
        ax.clear(); ax.set_axis_off()
        ax.imshow(img, cmap='gray', interpolation='nearest', aspect='equal')
        if self.line_pts is not None:
            (x1,y1),(x2,y2) = self.line_pts
            ax.plot([x1,x2],[y1,y2],'y-',lw=2,alpha=0.9)
        if self.beam_pts is not None:
            (x1,y1),(x2,y2) = self.beam_pts
            ax.plot([x1,x2],[y1,y2],'m-',lw=2,alpha=0.9)
        self.canvas.draw_idle()

    # --------------------- user tools ---------------------
    def _start_calibrate(self):
        self.mode = 'calibrate'; self.pending_pts = []
        QMessageBox.information(self, 'Calibration', 'Click two points a known distance apart (mm).')

    def _start_line_fft(self):
        self.mode = 'line_fft'; self.pending_pts = []
        QMessageBox.information(self, 'Line FFT', 'Click two points to define the analysis line.')

    def _start_beam_diam(self):
        self.mode = 'beam_diam'; self.pending_pts = []
        QMessageBox.information(self, 'Beam diameter', 'Click two points across the beam aperture edge-to-edge.')

    def _on_canvas_click(self, event):
        if event.inaxes != self.canvas.ax: return
        if self.mode == 'idle': return
        if event.xdata is None or event.ydata is None: return
        if self.orig_img is None: return
        x = float(np.clip(event.xdata, 0, self.orig_img.shape[1]-1))
        y = float(np.clip(event.ydata, 0, self.orig_img.shape[0]-1))
        self.pending_pts.append((x, y))
        if len(self.pending_pts) < 2: return
        p1, p2 = self.pending_pts[:2]
        self.pending_pts = []
        if self.mode == 'calibrate':
            self._finish_calibration(p1, p2)
        elif self.mode == 'line_fft':
            self._finish_line_fft(p1, p2)
        elif self.mode == 'beam_diam':
            self._finish_beam_diameter(p1, p2)
        self.mode = 'idle'

    def _finish_calibration(self, p1, p2):
        dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
        pix_dist = float(np.hypot(dx, dy))
        if pix_dist < 1e-9:
            QMessageBox.warning(self, 'Calibration', 'Points too close. Try again.')
            return
        mm_guess = max(1e-12, self.calib.mm_per_pixel) * pix_dist
        mm_known, ok = QInputDialog.getDouble(self, 'Known distance', 'Distance between clicks (mm):', value=mm_guess, min=1e-12, max=1e12, decimals=6)
        if not ok: return
        mmpp_new = mm_known / pix_dist
        if not np.isfinite(mmpp_new) or mmpp_new <= 0:
            QMessageBox.warning(self, 'Calibration', 'Invalid value. Try again.')
            return
        self.calib.mm_per_pixel = mmpp_new
        self.spin_mmpp.blockSignals(True); self.spin_mmpp.setValue(self.calib.mm_per_pixel); self.spin_mmpp.blockSignals(False)
        self.line_pts = (p1, p2)
        self._update_metrics(); self.canvas.draw_idle()

    def _finish_line_fft(self, p1, p2):
        self.line_pts = (p1, p2)
        self.canvas.draw_idle()
        img = self.orig_img
        if img is None: return
        p1r = (np.clip(p1[1], 0, img.shape[0]-1), np.clip(p1[0], 0, img.shape[1]-1))
        p2r = (np.clip(p2[1], 0, img.shape[0]-1), np.clip(p2[0], 0, img.shape[1]-1))
        prof = profile_line(img, p1r, p2r, mode='reflect', order=1)
        n = len(prof)
        if n < 8:
            QMessageBox.warning(self, 'Line FFT', 'Selected line is too short. Pick longer line.')
            return
        mmpp = max(1e-12, self.calib.mm_per_pixel)
        s_mm = np.arange(n) * mmpp
        freqs_cpp = np.fft.rfftfreq(n, d=1.0)
        freqs_cmm = freqs_cpp / mmpp
        spec = np.abs(np.fft.rfft(prof))
        if len(spec) > 1: spec[0] = 0.0
        if spec.max() <= 0:
            QMessageBox.information(self, 'Line FFT', 'No dominant frequency found.')
            return
        k_idx = int(np.argmax(spec))
        k_peak = float(freqs_cmm[k_idx])
        if k_peak <= 0:
            QMessageBox.information(self, 'Line FFT', 'Peak frequency is zero.')
            return
        factor = 2.0 if self.calib.standing_wave else 1.0
        lam_mm = factor / k_peak
        f_mhz = self.calib.speed_ms / (1000.0 * lam_mm)
        self.lam_mm = float(lam_mm); self.f_mhz = float(f_mhz)
        self._update_metrics()

        # Enable filtering now that the line FFT has been done
        if self.chk_enable_after_fft.isChecked():
            self.filter_enabled = True
        # Show the line FFT dialog
        dlg = FFTDialog(self)
        dlg.update_plots(s_mm, prof, freqs_cmm, spec, self.lam_mm, self.f_mhz)
        dlg.show(); self._dialogs.append(dlg)
        self._update_everything()

    def _finish_beam_diameter(self, p1, p2):
        self.beam_pts = (p1, p2)
        self.canvas.draw_idle()
        pix_dist = float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
        self.D_mm = pix_dist * max(1e-12, self.calib.mm_per_pixel)
        self._update_metrics()

    # --------------------- metrics / sim ---------------------
    def _update_metrics(self):
        self.lbl_mmpp.setText(f"{self.calib.mm_per_pixel:.6f}")
        self.lbl_D.setText('—' if self.D_mm is None else f"{self.D_mm:.4f}")
        self.lbl_lambda.setText('—' if self.lam_mm is None else f"{self.lam_mm:.4f}")
        self.lbl_freq.setText('—' if self.f_mhz is None else f"{self.f_mhz:.3f}")
        if (self.D_mm is not None) and (self.lam_mm is not None) and (self.lam_mm > 0):
            N_mm = (self.D_mm**2) / (4.0 * self.lam_mm)
            self.lbl_rayleigh.setText(f"{N_mm:.2f}")
        else:
            self.lbl_rayleigh.setText('—')

    def _simulate_beam(self):
        if self.D_mm is None or self.lam_mm is None:
            QMessageBox.warning(self, 'Beam simulation', 'Measure beam diameter and extract λ from line FFT first.')
            return
        try:
            D = float(self.D_mm); lam_mm = float(self.lam_mm)
            Lx_mm = max(4*D, 10.0)
            Nx = 256; Ny = 256
            dx_mm = Lx_mm / Nx
            a_mm = D * 0.5
            x = (np.arange(Nx) - Nx/2) * dx_mm
            y = (np.arange(Ny) - Ny/2) * dx_mm
            X, Y = np.meshgrid(x, y)
            U0 = np.where((X**2 + Y**2) <= a_mm*a_mm, 1.0, 0.0).astype(np.complex128)
            A = np.fft.fft2(np.fft.ifftshift(U0))
            dx_m = dx_mm / 1000.0
            k = 2*np.pi / (lam_mm / 1000.0)
            fx = np.fft.fftfreq(Nx, d=dx_m); fy = np.fft.fftfreq(Ny, d=dx_m)
            KX, KY = np.meshgrid(2*np.pi*fx, 2*np.pi*fy)
            K2 = KX**2 + KY**2
            N_mm = (D*D) / (4.0*lam_mm)
            z_max_mm = max(2.0*N_mm, 10.0)
            Nz = 180
            z_mm = np.linspace(0.0, z_max_mm, Nz)
            I = np.empty((Nz, Nx), dtype=np.float64)
            for i, z in enumerate(z_mm):
                z_m = z / 1000.0
                tmp = k*k - K2
                tmp = np.where(tmp >= 0.0, np.sqrt(tmp), 0.0)
                H = np.exp(1j * z_m * tmp)
                U = np.fft.ifftshift(np.fft.ifft2(A * H))
                row = np.abs(U[Ny//2, :])**2
                I[i, :] = row.real
            I = I - I.min(); mx = I.max();
            if mx > 0: I = I / mx
            sim = BeamSimDialog(self)
            sim.set_data(I, x_mm=x, z_mm=z_mm)
            sim.show(); self._dialogs.append(sim)
        except Exception as e:
            QMessageBox.critical(self, 'Simulation error', str(e))

# --------------------- main ---------------------

def main():
    app = QApplication(sys.argv)
    w = UltrasoundFilterGUI()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
