#!/usr/bin/env python3
"""
Interactive line-spectrum analyzer with SAME-SIZE panels (image & graph)
- Left:  filtered image (live preview, original aspect ratio preserved)
- Right: 1D FFT along the selected line (panel sized to match the image)

Defaults:
- Wave speed = 1497 m/s (water @ ~25 °C)
- Standing-wave INTENSITY interpretation enabled ("Intensity ÷2")

Controls (right panel):
  • BG radius (mm)
  • Band-pass Low/High (cycles/mm)
  • Speed m/s (editable)
  • Toggles: "Intensity ÷2", "Show Hz axis"
  • Button: "Reselect line" (click two points on the image)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from tkinter import Tk, filedialog, simpledialog
from skimage import io, color
from skimage.filters import gaussian
from skimage.measure import profile_line

# -------------------- Tiny tkinter helpers --------------------
def ask_float(title, prompt, initial=None, minvalue=None):
    root = Tk(); root.withdraw(); root.update()
    val = simpledialog.askfloat(title, prompt, initialvalue=initial, minvalue=minvalue)
    root.destroy()
    return val

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

# -------------------- Image & DSP helpers --------------------
def to_grayscale_float(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img

def normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    mn, mx = float(np.min(a)), float(np.max(a))
    if mx > mn:
        return (a - mn) / (mx - mn)
    return np.zeros_like(a)

def gaussian_background_subtract(image, radius_mm, mm_per_px):
    if radius_mm is None or radius_mm <= 0:
        return image
    sigma_px = max(radius_mm / mm_per_px, 0.0)
    if sigma_px < 1e-6:
        return image
    bg = gaussian(image, sigma=sigma_px, preserve_range=True)
    return image - bg

def bandpass_2d_fft(image, mm_per_px, low_cmm=None, high_cmm=None):
    """Hard annular mask in 2D frequency domain. Cutoffs in cycles/mm."""
    H, W = image.shape
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=1.0))  # cycles/pixel
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=1.0))
    FX, FY = np.meshgrid(fx, fy)
    R_cpx = np.sqrt(FX**2 + FY**2)

    def cmm_to_cpx(cmm):  # cycles/mm -> cycles/pixel
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
    return np.fft.ifft2(np.fft.ifftshift(Ff)).real

def line_fft(profile):
    n = len(profile)
    if n < 2:
        return np.array([0.0]), np.array([0.0])
    prof = profile - np.mean(profile)
    prof = prof * np.hanning(n)
    spec = np.fft.fft(prof)
    freqs_px = np.fft.fftfreq(n, d=1.0)  # cycles/pixel
    mag = np.abs(spec) / n
    return freqs_px, mag

def bandpass_1d_fft(profile, mm_per_px, low_cmm=None, high_cmm=None):
    n = len(profile)
    spec = np.fft.fft(profile - np.mean(profile))
    freqs_px = np.fft.fftfreq(n, d=1.0)

    def cmm_to_cpx(cmm):  # cycles/mm -> cycles/pixel
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

# -------------------- App --------------------
def main():
    # ---- Load & grayscale ----
    path = pick_file()
    img = io.imread(path)
    gray0 = to_grayscale_float(img)

    # ---- CALIBRATION popup ----
    fig_cal, ax_cal = plt.subplots(figsize=(7, 7))
    ax_cal.imshow(gray0, cmap="gray", interpolation="nearest")
    ax_cal.set_title("CALIBRATION: click TWO points with known distance (mm)\nClose window when done.")
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

    # Image aspect ratio (height / width) for box sizing
    img_box_aspect = gray0.shape[0] / gray0.shape[1]

    # ---- Initial state ----
    state = {
        "gray0": gray0,
        "um_per_px": um_per_px,
        "mm_per_px": mm_per_px,
        "px_m": px_m,
        "nyq_cmm": nyq_cmm,
        "bg_mm": 0.2,
        "low_cmm": 0.0,
        "high_cmm": nyq_cmm,
        "v_mps": 1497.0,     # default: speed of sound in water @ ~25°C
        "use_speed": True,
        "intensity_case": True,  # standing-wave intensity by default
        "line_pts": None,        # ( (x0,y0), (x1,y1) )
        "secax": None,
        "metrics_text": None,
        "selecting_line": True,  # start by selecting a line
        "hint_text": None,
        "img_box_aspect": img_box_aspect,
    }

    # ---- Build layout: [image | spectrum | right control panel] ----
    plt.close("all")
    fig = plt.figure(figsize=(15, 8.8))
    gs = fig.add_gridspec(
        nrows=1, ncols=3, width_ratios=[1, 1, 0.9],   # <-- equal widths for left & middle
        left=0.05, right=0.98, top=0.93, bottom=0.09, wspace=0.25
    )
    ax_img   = fig.add_subplot(gs[0, 0])
    ax_spec  = fig.add_subplot(gs[0, 1])
    ax_panel = fig.add_subplot(gs[0, 2]); ax_panel.axis("off")

    # Make BOTH left & middle panels the SAME box size & match the image aspect
    def apply_box_aspect():
        try:
            ax_img.set_box_aspect(state["img_box_aspect"])
            ax_spec.set_box_aspect(state["img_box_aspect"])
        except Exception:
            # set_box_aspect requires Matplotlib >= 3.3; if unavailable, we still preserve image aspect via imshow
            pass

    apply_box_aspect()

    # Small hint overlay in the image panel
    state["hint_text"] = ax_img.text(0.02, 0.98, "Click TWO points to set the line",
                                     transform=ax_img.transAxes, va="top", ha="left",
                                     fontsize=9, color="tab:blue",
                                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Control panel stacking helper
    panel_pos = ax_panel.get_position()
    px, py = panel_pos.x0, panel_pos.y0
    pw, ph = panel_pos.width, panel_pos.height
    y = panel_pos.y1 - 0.04
    def add_box(rel_w=0.95, h=0.055, gap=0.014):
        nonlocal y
        w = pw * rel_w
        x = px + (pw - w) / 2
        ax = fig.add_axes([x, y, w, h])
        y -= (h + gap)
        return ax

    # Metrics (monospace text)
    metrics_ax = fig.add_axes([px + 0.03*pw, panel_pos.y1 - 0.13, 0.94*pw, 0.11])
    metrics_ax.axis("off")
    state["metrics_text"] = metrics_ax.text(0, 1, "", va="top", ha="left", fontsize=9, family="monospace")

    # Sliders
    sl_bg   = Slider(add_box(), "BG radius (mm)", 0.0, 5.0, valinit=state["bg_mm"])
    sl_low  = Slider(add_box(), "Low (c/mm)",     0.0, nyq_cmm, valinit=state["low_cmm"])
    sl_high = Slider(add_box(), "High (c/mm)",    0.0, nyq_cmm, valinit=state["high_cmm"])

    # Speed textbox
    txt_ax = add_box(h=0.06)
    txt = TextBox(txt_ax, "Speed m/s:", initial=f"{state['v_mps']:.0f}")

    # Toggles
    chk_ax = add_box(h=0.08)
    chk = CheckButtons(chk_ax, ["Intensity ÷2", "Show Hz axis"],
                       [state["intensity_case"], state["use_speed"]])

    # Buttons
    btn_line_ax = add_box(h=0.06)
    btn_line = Button(btn_line_ax, "Reselect line")

    # ---- Processing helpers ----
    def compute_filtered_image():
        g0 = state["gray0"]
        g_bg = gaussian_background_subtract(g0, state["bg_mm"], state["mm_per_px"])
        g_f = bandpass_2d_fft(g_bg, state["mm_per_px"], state["low_cmm"], state["high_cmm"])
        return normalize01(g_f), g_f  # (disp, filtered)

    def spectrum_from_line(g_filtered_disp):
        if state["line_pts"] is None:
            return None
        (x0, y0), (x1, y1) = state["line_pts"]
        # profile from filtered image display; then extra 1D BP
        prof = profile_line(g_filtered_disp, (y0, x0), (y1, x1),
                            mode="reflect", order=1, linewidth=1, reduce_func=None)
        if getattr(prof, "ndim", 1) > 1:
            prof = prof.mean(axis=1)
        prof_bp = bandpass_1d_fft(prof, state["mm_per_px"], state["low_cmm"], state["high_cmm"])
        fpx, mag = line_fft(prof_bp)
        fcmm = fpx / state["mm_per_px"]
        if len(mag) > 1:
            idx = 1 + np.argmax(mag[1:])
        else:
            idx = 0
        fpk = max(fcmm[idx], 0.0)
        lam_d = (1.0 / fpk) if fpk > 0 else np.inf
        lam_i = (2.0 / fpk) if fpk > 0 else np.inf

        f_field = f_int = None
        v = state["v_mps"] if state["use_speed"] else None
        if v and np.isfinite(lam_d):
            lam_d_m = lam_d * 1e-3; lam_i_m = lam_i * 1e-3
            f_field = v / lam_d_m if lam_d_m > 0 else None
            f_int   = v / lam_i_m if lam_i_m > 0 else None
        return fcmm, mag, fpk, lam_d, lam_i, f_field, f_int

    def set_top_axis():
        # Remove existing secondary axis if any
        if state["secax"] is not None:
            try: state["secax"].remove()
            except Exception: pass
            state["secax"] = None

        if not state["use_speed"] or (state["v_mps"] is None) or state["v_mps"] <= 0:
            fig.canvas.draw_idle(); return

        v = state["v_mps"]
        if state["intensity_case"]:
            # intensity: f = v*(ν*1000)/2
            def cmm_to_Hz(x): return v * (x * 1000.0) / 2.0
            def Hz_to_cmm(x): return (x * 2.0) / (v * 1000.0)
            label = "Temporal frequency (Hz) — INTENSITY (÷2)"
        else:
            # field: f = v*(ν*1000)
            def cmm_to_Hz(x): return v * (x * 1000.0)
            def Hz_to_cmm(x): return x / (v * 1000.0)
            label = "Temporal frequency (Hz) — FIELD"

        sec = ax_spec.secondary_xaxis('top', functions=(cmm_to_Hz, Hz_to_cmm))
        sec.set_xlabel(label)
        sec.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda val, _pos: f"{val/1e6:.3g} MHz" if abs(val) >= 1e6 else f"{val:.3g} Hz"
        ))
        state["secax"] = sec
        fig.canvas.draw_idle()

    def update_metrics(fpk, lam_d, lam_i, f_field, f_int):
        lines = [
            f"Pixel size   : {state['um_per_px']:.4g} µm/px",
            f"Nyquist      : {state['nyq_cmm']:.4g} c/mm",
            f"Peak (c/mm)  : {fpk:.4g}",
            f"λ_direct     : {lam_d:.4g} mm",
            f"λ_intensity  : {lam_i:.4g} mm",
            f"Speed (m/s)  : {state['v_mps']:.4g} {'[ON]' if state['use_speed'] else '[OFF]'}",
            f"f_field      : {f_field/1e6:.4g} MHz" if f_field is not None else "f_field      : —",
            f"f_intensity  : {f_int/1e6:.4g} MHz" if f_int is not None else "f_intensity  : —",
        ]
        state["metrics_text"].set_text("\n".join(lines))

    # ---- Redraw pipeline ----
    def update_all(_=None):
        # keep low <= high
        low = float(sl_low.val)
        high = float(sl_high.val)
        if high <= low + 1e-9:
            high = min(low + 1e-6, state["nyq_cmm"])
            sl_high.set_val(high)
        state["bg_mm"] = float(sl_bg.val)
        state["low_cmm"] = low
        state["high_cmm"] = high

        # Recompute filtered image
        disp, _ = compute_filtered_image()

        # Re-apply same-size panel boxes every draw
        apply_box_aspect()

        # Image preview (real-time), preserve original aspect
        ax_img.clear()
        ax_img.imshow(disp, cmap="gray", interpolation="nearest", aspect="equal")
        ax_img.set_title("Filtered image (live)")
        ax_img.set_axis_off()
        if state["line_pts"] is not None:
            (x0, y0), (x1, y1) = state["line_pts"]
            ax_img.plot([x0, x1], [y0, y1], "-r", lw=2)
            ax_img.scatter([x0, x1], [y0, y1], c="yellow", s=40)
        state["hint_text"].set_visible(state["selecting_line"])

        # 1D spectrum panel (same box size as image)
        ax_spec.clear()
        ax_spec.set_title("1D FFT along selected line")
        ax_spec.set_xlabel("Spatial frequency (cycles/mm)")
        ax_spec.set_ylabel("Magnitude")
        ax_spec.set_xlim(0, state["nyq_cmm"])
        ax_spec.grid(True, alpha=0.3)

        fpk = 0.0; lam_d = np.inf; lam_i = np.inf; f_field = None; f_int = None
        out = spectrum_from_line(disp)
        if out is not None:
            fc, mag, fpk, lam_d, lam_i, f_field, f_int = out
            ax_spec.plot(fc, mag, lw=1.8)
            if np.isfinite(fpk) and fpk > 0:
                ax_spec.axvline(fpk, ls="--", lw=1)
                ax_spec.text(fpk, max(mag)*0.9,
                             f"peak ≈ {fpk:.3g} c/mm\nλ={lam_d:.3g} mm (direct)\nλ={lam_i:.3g} mm (int)",
                             rotation=90, va="top", ha="right", fontsize=9)

        # Secondary axis in Hz
        set_top_axis()

        # Metrics + window title
        update_metrics(fpk, lam_d, lam_i, f_field, f_int)
        fig.suptitle(
            f"BG={state['bg_mm']:.3g} mm | BP=[{state['low_cmm']:.3g}, {state['high_cmm']:.3g}] c/mm | "
            f"Speed default: 1497 m/s (water @ 25°C)",
            fontsize=11
        )
        fig.canvas.draw_idle()

    # ---- Wire controls ----
    def on_speed_submit(text):
        try:
            v = float(text)
            state["v_mps"] = v if v > 0 else None
        except ValueError:
            state["v_mps"] = None
        update_all()
    txt.on_submit(on_speed_submit)

    def on_check_clicked(label):
        if label == "Intensity ÷2":
            state["intensity_case"] = not state["intensity_case"]
        elif label == "Show Hz axis":
            state["use_speed"] = not state["use_speed"]
        update_all()
    chk.on_clicked(on_check_clicked)

    sl_bg.on_changed(update_all)
    sl_low.on_changed(update_all)
    sl_high.on_changed(update_all)

    # Line (re)selection inside the same window
    btn_line.on_clicked(lambda _e: start_line_selection())

    click_buffer = []
    def on_mouse_click(event):
        if not state["selecting_line"]:
            return
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return
        click_buffer.append((event.xdata, event.ydata))
        ax_img.scatter([event.xdata], [event.ydata], c="cyan", s=40)
        fig.canvas.draw_idle()
        if len(click_buffer) >= 2:
            state["line_pts"] = (click_buffer[0], click_buffer[1])
            click_buffer.clear()
            state["selecting_line"] = False
            update_all()
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    def start_line_selection():
        state["selecting_line"] = True
        state["line_pts"] = None
        update_all()

    # ---- First draw ----
    update_all()
    plt.show()

if __name__ == "__main__":
    main()
