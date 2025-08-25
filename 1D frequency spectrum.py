#!/usr/bin/env python3
"""
Interactive wave-frequency analyzer (real-time sliders)

Features
- 2-point physical calibration (click two points with known separation; enter mm)
- Live background removal (Gaussian local mean) controlled via slider [mm]
- Live annular 2D band-pass (low/high cutoffs) controlled via sliders [cycles/mm]
- Click to reselect the analysis line in the same window (button)
- 1D profile & 1D FFT (orig vs filtered), 2D FFT (log) with axes in cycles/mm
- Optional wave speed input; top x-axis shows temporal frequency in Hz/MHz
- “Intensity ÷2” toggle so standing-wave intensity gives the *true* field frequency

Requirements
    pip install numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from tkinter import Tk, filedialog, simpledialog, messagebox
from skimage import io, color
from skimage.filters import gaussian
from skimage.measure import profile_line

# -------------------- Basic UI helpers (tkinter) --------------------
def msg_info(msg, title="Info"):
    root = Tk(); root.withdraw(); root.update()
    messagebox.showinfo(title, msg)
    root.destroy()

def ask_float(title, prompt, initial=None, minvalue=None):
    """Uses tkinter.simpledialog.askfloat with 'minvalue' (fixed)."""
    root = Tk(); root.withdraw(); root.update()
    val = simpledialog.askfloat(title, prompt, initialvalue=initial, minvalue=minvalue)
    root.destroy()
    return val

def ask_string(title, prompt):
    root = Tk(); root.withdraw(); root.update()
    s = simpledialog.askstring(title, prompt)
    root.destroy()
    return s

def pick_file():
    root = Tk(); root.withdraw(); root.update()
    path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Images","*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif"),
                   ("All files","*.*")]
    )
    root.destroy()
    if not path:
        raise SystemExit("No file selected.")
    return path

# -------------------- Core helpers --------------------
def to_grayscale_float(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    # normalize if looks like 0..255 range
    if img.max() > 1.5:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img

def normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    mn, mx = np.min(a), np.max(a)
    if mx > mn:
        return (a - mn) / (mx - mn)
    return np.zeros_like(a)

def gaussian_background_subtract(image, radius_mm, mm_per_px):
    """Subtract local mean estimated by Gaussian blur with sigma = radius_px."""
    if radius_mm is None or radius_mm <= 0:
        return image
    sigma_px = max(radius_mm / mm_per_px, 0.0)
    if sigma_px < 1e-6:
        return image
    bg = gaussian(image, sigma=sigma_px, preserve_range=True)
    return image - bg

def bandpass_2d_fft(image, mm_per_px, low_cmm=None, high_cmm=None):
    """Hard annular mask in 2D frequency domain: keep low<=|f|<=high (in cycles/mm)."""
    H, W = image.shape
    # frequency grids in cycles/pixel
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=1.0))
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=1.0))
    FX, FY = np.meshgrid(fx, fy)
    R_cpx = np.sqrt(FX**2 + FY**2)  # radial cycles/pixel

    def cmm_to_cpx(cmm):
        return (cmm or 0.0) * mm_per_px

    low_cpx  = cmm_to_cpx(low_cmm)
    high_cpx = cmm_to_cpx(high_cmm) if (high_cmm is not None and high_cmm > 0) else None

    F = np.fft.fft2(image)
    Fsh = np.fft.fftshift(F)
    mask = np.ones_like(R_cpx, dtype=bool)
    if low_cpx > 0:
        mask &= (R_cpx >= low_cpx)
    if high_cpx is not None:
        mask &= (R_cpx <= high_cpx)
    Ff = Fsh * mask
    out = np.fft.ifft2(np.fft.ifftshift(Ff)).real
    return out

def compute_2d_fft_display(gray):
    F = np.fft.fft2(gray)
    Fsh = np.fft.fftshift(F)
    mag = np.abs(Fsh)
    mag_log = np.log1p(mag)
    return normalize01(mag_log)

def line_fft(profile):
    n = len(profile)
    if n < 2:
        return np.array([0.0]), np.array([0.0])
    prof = profile - np.mean(profile)
    prof = prof * np.hanning(n)  # light taper
    spec = np.fft.fft(prof)
    freqs_px = np.fft.fftfreq(n, d=1.0)     # cycles/pixel
    mag = np.abs(spec) / n
    return freqs_px, mag

def bandpass_1d_fft(profile, mm_per_px, low_cmm=None, high_cmm=None):
    n = len(profile)
    spec = np.fft.fft(profile - np.mean(profile))
    freqs_px = np.fft.fftfreq(n, d=1.0)

    def cmm_to_cpx(cmm):
        return (cmm or 0.0) * mm_per_px

    low_cpx  = cmm_to_cpx(low_cmm)
    high_cpx = cmm_to_cpx(high_cmm) if (high_cmm is not None and high_cmm > 0) else None

    mask = np.ones_like(freqs_px, dtype=bool)
    if low_cpx > 0:
        mask &= (np.abs(freqs_px) >= low_cpx)
    if high_cpx is not None:
        mask &= (np.abs(freqs_px) <= high_cpx)

    spec_f = spec * mask
    return np.fft.ifft(spec_f).real

# -------------------- Main (interactive) --------------------
def main():
    # 1) Load + grayscale
    path = pick_file()
    img = io.imread(path)
    gray0 = to_grayscale_float(img)

    # 2) CALIBRATION: click two points, enter known distance (mm)
    fig_cal, ax_cal = plt.subplots(figsize=(7, 7))
    ax_cal.imshow(gray0, cmap="gray", interpolation="nearest")
    ax_cal.set_title("CALIBRATION: click TWO points with known real distance\nClose window when done.")
    ax_cal.set_axis_off()
    cal_pts = plt.ginput(2, timeout=0)
    plt.close(fig_cal)
    if len(cal_pts) != 2:
        raise SystemExit("Calibration requires two points.")
    (cx0, cy0), (cx1, cy1) = cal_pts
    pixel_distance = float(np.hypot(cx1 - cx0, cy1 - cy0))
    known_mm = ask_float("Calibration distance",
                         "Enter known distance between the two points (mm):",
                         initial=1.0, minvalue=1e-12)
    if known_mm is None:
        raise SystemExit("Calibration distance is required.")
    um_per_px = (known_mm * 1000.0) / pixel_distance
    mm_per_px = um_per_px / 1000.0
    px_m = mm_per_px * 1e-3
    nyq_cmm = 1.0 / (2.0 * mm_per_px)

    # 3) Initial parameters
    init_bg_mm = 0.2
    init_low_cmm = 0.0
    init_high_cmm = nyq_cmm
    init_speed = "1540"  # typical soft tissue (m/s)
    intensity_case = True  # default to intensity ÷2

    # 4) Build interactive figure (4 plots + controls)
    plt.close("all")
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.18], left=0.05, right=0.98, top=0.92, bottom=0.08, hspace=0.32, wspace=0.25)

    ax_img  = fig.add_subplot(gs[0, 0])
    ax_fft2 = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[1, 1])

    # (Use the two extra columns for wide plots / future expansion)
    # Put controls in the bottom row spanning columns
    ax_bg_sl   = fig.add_subplot(gs[2, 0])
    ax_low_sl  = fig.add_subplot(gs[2, 1])
    ax_high_sl = fig.add_subplot(gs[2, 2])
    ax_btns    = fig.add_subplot(gs[2, 3])

    # Prepare state
    state = {
        "mm_per_px": mm_per_px,
        "px_m": px_m,
        "nyq_cmm": nyq_cmm,
        "gray0": gray0,
        "bg_mm": init_bg_mm,
        "low_cmm": init_low_cmm,
        "high_cmm": init_high_cmm,
        "v_mps": float(init_speed),
        "use_speed": True,
        "intensity_case": intensity_case,
        "line_pts": None,  # ((x0,y0),(x1,y1))
        "selecting_line": False,
        "secax": None,     # secondary x-axis handle
        "img_artist": None,
        "line_artist": None,
        "fft2_artist": None,
        "prof_lines": [],
        "spec_lines": [],
        "peak_line": None,
        "peak_text": None,
        "extent": None,
    }

    # ---------- Controls ----------
    sl_bg   = Slider(ax_bg_sl,   "BG radius (mm)", 0.0, 5.0, valinit=init_bg_mm)
    sl_low  = Slider(ax_low_sl,  "Low c/mm", 0.0, nyq_cmm, valinit=init_low_cmm)
    sl_high = Slider(ax_high_sl, "High c/mm", 0.0, nyq_cmm, valinit=init_high_cmm)

    # Buttons + toggles
    ax_btns.axis("off")
    # small axes inside ax_btns for widgets
    btn_reselect_ax = fig.add_axes([ax_btns.get_position().x0, ax_btns.get_position().y0+0.06, 0.10, 0.05])
    btn_reselect = Button(btn_reselect_ax, "Reselect line")

    chk_ax = fig.add_axes([ax_btns.get_position().x0+0.12, ax_btns.get_position().y0+0.06, 0.12, 0.08])
    chk = CheckButtons(chk_ax, ["Intensity ÷2", "Show Hz axis"], [state["intensity_case"], state["use_speed"]])

    txt_ax = fig.add_axes([ax_btns.get_position().x0+0.26, ax_btns.get_position().y0+0.06, 0.18, 0.05])
    txt = TextBox(txt_ax, "Speed m/s: ", initial=init_speed)

    # ---------- Helper functions that depend on state ----------
    def update_filters():
        """Apply background removal + 2D band-pass to image; update 2D FFT extent."""
        gray_bg = gaussian_background_subtract(state["gray0"], state["bg_mm"], state["mm_per_px"])
        gray_f = bandpass_2d_fft(gray_bg, state["mm_per_px"], state["low_cmm"], state["high_cmm"])
        disp = normalize01(gray_f)

        # 2D FFT (for display) + physical frequency axes extent in cycles/mm
        f2d = compute_2d_fft_display(gray_f)
        H, W = gray_f.shape
        fx_m = np.fft.fftshift(np.fft.fftfreq(W, d=state["px_m"]))  # cycles/m
        fy_m = np.fft.fftshift(np.fft.fftfreq(H, d=state["px_m"]))
        fx_mm = fx_m / 1000.0
        fy_mm = fy_m / 1000.0
        extent = [fx_mm[0], fx_mm[-1], fy_mm[0], fy_mm[-1]]

        return disp, f2d, extent

    def ensure_line():
        """If no line yet, let the user click two points on the image."""
        if state["line_pts"] is not None:
            return
        ax_img.set_title("Click TWO points to define the analysis line")
        fig.canvas.draw_idle()
        pts = plt.ginput(2, timeout=0)
        if len(pts) != 2:
            msg_info("Two points required for analysis line.", "Line selection")
            return
        state["line_pts"] = (pts[0], pts[1])
        ax_img.set_title("Filtered image with analysis line")

    def current_profiles(img_disp):
        """Return (profile_original, profile_filteredDisplay) along the current line."""
        if state["line_pts"] is None:
            return None, None
        (x0, y0), (x1, y1) = state["line_pts"]
        p0 = profile_line(state["gray0"], (y0, x0), (y1, x1), mode="reflect", order=1, linewidth=1, reduce_func=None)
        p1 = profile_line(img_disp,            (y0, x0), (y1, x1), mode="reflect", order=1, linewidth=1, reduce_func=None)
        if getattr(p0, "ndim", 1) > 1: p0 = p0.mean(axis=1)
        if getattr(p1, "ndim", 1) > 1: p1 = p1.mean(axis=1)
        return p0, p1

    def compute_spectrum(profile, low_cmm, high_cmm):
        """Return (freqs_c/mm, mag, peak_freq_c/mm, lambda_direct_mm, lambda_int_mm, f_direct_Hz, f_int_Hz)"""
        if profile is None or len(profile) < 2:
            return (np.array([0.0]), np.array([0.0]), 0.0, np.inf, np.inf, None, None)

        prof_bp = bandpass_1d_fft(profile, state["mm_per_px"], low_cmm, high_cmm)
        freqs_px, mag = line_fft(prof_bp)
        freqs_cmm = freqs_px / state["mm_per_px"]

        if len(mag) > 1:
            idx_peak = 1 + np.argmax(mag[1:])  # skip DC
        else:
            idx_peak = 0
        fpk_cmm = max(freqs_cmm[idx_peak], 0.0)

        lam_direct_mm = (1.0 / fpk_cmm) if fpk_cmm > 0 else np.inf
        lam_int_mm = (2.0 / fpk_cmm) if fpk_cmm > 0 else np.inf

        f_direct = f_int = None
        v = state["v_mps"] if state["use_speed"] else None
        if v and np.isfinite(lam_direct_mm):
            lam_d_m = lam_direct_mm * 1e-3
            lam_i_m = lam_int_mm * 1e-3
            f_direct = v / lam_d_m if lam_d_m > 0 else None
            f_int = v / lam_i_m if lam_i_m > 0 else None

        return freqs_cmm, mag, fpk_cmm, lam_direct_mm, lam_int_mm, f_direct, f_int

    def set_top_axis(ax, use_speed, intensity_case):
        """Create/update the secondary x-axis mapping cycles/mm -> Hz (or hide)."""
        if state["secax"] is not None:
            # Reset/remove previous
            try:
                state["secax"].remove()
            except Exception:
                pass
            state["secax"] = None

        if not use_speed:
            fig.canvas.draw_idle()
            return

        v = state["v_mps"]
        if v is None or v <= 0:
            fig.canvas.draw_idle()
            return

        if intensity_case:
            # f = v * (nu*1000)/2  (nu in cycles/mm)
            def cmm_to_Hz(x): return v * (x * 1000.0) / 2.0
            def Hz_to_cmm(x): return (x * 2.0) / (v * 1000.0)
            label = "Temporal frequency (Hz) — INTENSITY (÷2)"
        else:
            # field case: f = v * (nu*1000)
            def cmm_to_Hz(x): return v * (x * 1000.0)
            def Hz_to_cmm(x): return x / (v * 1000.0)
            label = "Temporal frequency (Hz) — FIELD"

        secax = ax.secondary_xaxis('top', functions=(cmm_to_Hz, Hz_to_cmm))
        secax.set_xlabel(label)
        secax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _pos: f"{val/1e6:.3g} MHz" if abs(val)>=1e6 else f"{val:.3g} Hz"))
        state["secax"] = secax
        fig.canvas.draw_idle()

    # ---------- Main update pipeline ----------
    def update_all(_=None):
        # Clamp cutoffs: low <= high
        low = sl_low.val
        high = sl_high.val
        if high < low + 1e-9:
            high = min(low + 1e-6, state["nyq_cmm"])
            sl_high.set_val(high)

        state["bg_mm"] = sl_bg.val
        state["low_cmm"] = low
        state["high_cmm"] = high

        disp, f2d, extent = update_filters()
        state["extent"] = extent

        # --- Image panel
        ax_img.clear()
        ax_img.imshow(disp, cmap="gray", interpolation="nearest")
        ax_img.set_title("Filtered image (bg removed + band-pass)")
        ax_img.set_axis_off()

        # If a line exists, draw it
        if state["line_pts"] is not None:
            (x0, y0), (x1, y1) = state["line_pts"]
            ax_img.plot([x0, x1], [y0, y1], "-r", lw=2)
            ax_img.scatter([x0, x1], [y0, y1], c="yellow", s=40)

        # --- 2D FFT panel
        ax_fft2.clear()
        im = ax_fft2.imshow(f2d, cmap="gray", extent=extent, origin="lower", aspect="auto")
        ax_fft2.set_title("2D FFT magnitude (log) of filtered image")
        ax_fft2.set_xlabel("fx (cycles/mm)")
        ax_fft2.set_ylabel("fy (cycles/mm)")
        cbar = plt.colorbar(im, ax=ax_fft2, fraction=0.046, pad=0.04)
        cbar.set_label("Norm. log magnitude")

        # --- Profiles + Spectrum
        p0, p1disp = current_profiles(disp)
        ax_prof.clear()
        ax_prof.set_title("Line intensity profile")
        ax_prof.set_xlabel("Distance along line (pixels)")
        ax_prof.set_ylabel("Intensity")
        if p0 is not None:
            x_pix = np.linspace(0, 1, len(p0)) * len(p0)
            ax_prof.plot(x_pix, p0, lw=1.0, label="Original")
        if p1disp is not None:
            x_pix = np.linspace(0, 1, len(p1disp)) * len(p1disp)
            ax_prof.plot(x_pix, p1disp, lw=1.4, label="Filtered (img)")
        ax_prof.grid(True, alpha=0.25)
        ax_prof.legend(loc="best")

        # Spectrum (from *filtered* profile for stability)
        ax_spec.clear()
        ax_spec.set_title("1D FFT along line")
        ax_spec.set_xlabel("Spatial frequency (cycles/mm)")
        ax_spec.set_ylabel("Magnitude")
        ax_spec.grid(True, alpha=0.3)
        ax_spec.set_xlim(0, state["nyq_cmm"])

        if p0 is not None:
            # Original spectrum (faint)
            fp_o, mag_o, *_ = compute_spectrum(p0, 0.0, state["nyq_cmm"])
            ax_spec.plot(fp_o, mag_o, lw=0.9, label="Original (BP 0..Nyq)")

        if p1disp is not None:
            fp_f, mag_f, fpk_cmm, lam_d, lam_i, f_d, f_i = compute_spectrum(p1disp, state["low_cmm"], state["high_cmm"])
            ax_spec.plot(fp_f, mag_f, lw=1.6, label="Filtered (profile BP)")
            if np.isfinite(fpk_cmm) and fpk_cmm > 0:
                ax_spec.axvline(fpk_cmm, ls="--", lw=1)
                txt = f"peak ≈ {fpk_cmm:.3g} c/mm | λ={lam_d:.3g} mm (direct), {lam_i:.3g} mm (intensity)"
                ax_spec.text(fpk_cmm, max(mag_f)*0.9, txt, rotation=90, va="top", ha="right", fontsize=9)

            # Figure title summary
            if state["use_speed"] and state["v_mps"]:
                if state["intensity_case"] and f_i:
                    ftxt = f"{f_i/1e6:.3g} MHz (intensity)"
                elif (not state["intensity_case"]) and f_d:
                    ftxt = f"{f_d/1e6:.3g} MHz (field)"
                else:
                    ftxt = "n/a"
            else:
                ftxt = "—"
            fig.suptitle(
                f"Calibration: {um_per_px:.4g} µm/px | Filters: BG={state['bg_mm']:.3g} mm, "
                f"BP=[{state['low_cmm']:.3g}, {state['high_cmm']:.3g}] c/mm | "
                f"Peak: {fpk_cmm:.3g} c/mm | "
                f"λ_field≈{lam_i:.3g} mm | f≈{ftxt}",
                fontsize=11
            )

        # Secondary (top) axis in Hz
        set_top_axis(ax_spec, state["use_speed"], state["intensity_case"])

        ax_img.figure.canvas.draw_idle()

    # ---------- Event wiring ----------
    def on_slider_change(val):
        update_all()

    sl_bg.on_changed(on_slider_change)
    sl_low.on_changed(on_slider_change)
    sl_high.on_changed(on_slider_change)

    def on_reselect_clicked(_event):
        # Enable click-to-select on the image axes
        state["selecting_line"] = True
        state["line_pts"] = None
        ax_img.set_title("Click TWO points to define the analysis line (same area)")
        fig.canvas.draw_idle()

    btn_reselect.on_clicked(on_reselect_clicked)

    def on_check_clicked(label):
        if label == "Intensity ÷2":
            state["intensity_case"] = not state["intensity_case"]
        elif label == "Show Hz axis":
            state["use_speed"] = not state["use_speed"]
        update_all()

    chk.on_clicked(on_check_clicked)

    def on_speed_submit(text):
        try:
            v = float(text)
            state["v_mps"] = v if v > 0 else None
        except ValueError:
            state["v_mps"] = None
        update_all()

    txt.on_submit(on_speed_submit)

    # Mouse clicks for line reselection
    click_buffer = []
    def on_mouse_click(event):
        if not state["selecting_line"]:
            return
        if event.inaxes != ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        click_buffer.append((event.xdata, event.ydata))
        # Draw temporary markers
        ax_img.scatter([event.xdata], [event.ydata], c="cyan", s=40)
        fig.canvas.draw_idle()
        if len(click_buffer) >= 2:
            state["line_pts"] = (click_buffer[0], click_buffer[1])
            click_buffer.clear()
            state["selecting_line"] = False
            # fall through to update_all which will draw the line and recompute
            update_all()

    cid_click = fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # First run: compute filters and prompt for initial line
    update_all()
    ensure_line()
    update_all()

    plt.show()

if __name__ == "__main__":
    main()
