#!/usr/bin/env python3
"""
Interactive line-spectrum analyzer + Near/Far field simulation —
Layout cleaned + TOP IMAGE AT NATIVE PIXEL SIZE (no scaling/stretching)

Changes in this version:
  • The right control panel and sliders are stacked in their own column — no overlaps.
  • The top-left image axes is sized in inches so that 1 image pixel maps to 1 screen pixel
    (given the figure DPI). Aspect ratio is preserved; no scaling or stretching.
  • Gridspec uses absolute-width/height ratios proportional to inches so the
    image area is exactly the image's native width/height.
  • Minor UI polish (titles, grid, colorbar placement) without covering controls.

Requirements:
    pip install numpy matplotlib scikit-image
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
    root = Tk();
    root.withdraw();
    root.update()
    val = simpledialog.askfloat(title, prompt, initialvalue=initial, minvalue=minvalue)
    root.destroy()
    return val


def pick_file():
    root = Tk();
    root.withdraw();
    root.update()
    path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.gif"),
                   ("All files", "*.*")]
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
    R_cpx = np.sqrt(FX ** 2 + FY ** 2)

    def cmm_to_cpx(cmm):  # cycles/mm -> cycles/pixel
        return (cmm or 0.0) * mm_per_px

    low_cpx = cmm_to_cpx(low_cmm)
    high_cpx = cmm_to_cpx(high_cmm) if (high_cmm is not None and high_cmm > 0) else None

    F = np.fft.fft2(image)
    Fsh = np.fft.fftshift(F)
    mask = np.ones_like(R_cpx, dtype=bool)
    if low_cpx > 0:               mask &= (R_cpx >= low_cpx)
    if high_cpx is not None:      mask &= (R_cpx <= high_cpx)
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

    low_cpx = cmm_to_cpx(low_cmm)
    high_cpx = cmm_to_cpx(high_cmm) if (high_cmm is not None and high_cmm > 0) else None

    mask = np.ones_like(freqs_px, dtype=bool)
    if low_cpx > 0:               mask &= (np.abs(freqs_px) >= low_cpx)
    if high_cpx is not None:      mask &= (np.abs(freqs_px) <= high_cpx)

    spec_f = spec * mask
    return np.fft.ifft(spec_f).real


# -------------------- Angular Spectrum Near/Far-field simulation --------------------
def simulate_piston_field(D_mm, lam_mm, z_max_mm, span_mult_D=4.0, Nx=256, Nz=160):
    """
    Simulate normalized intensity |p|^2 for a uniform circular piston (baffled) with
    diameter D_mm and wavelength lam_mm. Returns (x_mm, z_mm, I_xz[Nz,Nx], I_axis[Nz]).
    - Lateral extent = span_mult_D * D (total width)
    - Grid is square (Nx x Nx) per z; we propagate via angular spectrum.
    - We build an x–z map by slicing the center row at each z.
    """
    if D_mm is None or not np.isfinite(D_mm) or D_mm <= 0:
        return None
    if lam_mm is None or not np.isfinite(lam_mm) or lam_mm <= 0:
        return None
    a_mm = D_mm / 2.0
    k_mm = 2.0 * np.pi / lam_mm  # rad/mm
    Lx_mm = max(span_mult_D * D_mm, 4.0 * D_mm)
    dx_mm = Lx_mm / Nx
    x = (np.arange(Nx) - Nx // 2) * dx_mm
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    # Aperture at z=0
    U0 = (X ** 2 + Y ** 2) <= (a_mm ** 2)
    U0 = U0.astype(np.complex64)
    # Precompute angular spectrum of source plane
    F0 = np.fft.fft2(U0)
    fx = np.fft.fftfreq(Nx, d=dx_mm)  # cycles/mm
    kx = 2.0 * np.pi * fx  # rad/mm
    KX, KY = np.meshgrid(kx, kx)  # using same for y (square grid)
    Ksq = KX ** 2 + KY ** 2
    # Evanescent components cutoff
    mask_prop = Ksq <= (k_mm ** 2)
    KZ = np.zeros_like(KX, dtype=np.float32)
    KZ[mask_prop] = np.sqrt((k_mm ** 2) - Ksq[mask_prop]).astype(np.float32)
    # z sampling
    z = np.linspace(0.0, z_max_mm, Nz, dtype=np.float32)
    # Collect x–z map (center row) and on-axis
    I_xz = np.zeros((Nz, Nx), dtype=np.float32)
    I_axis = np.zeros(Nz, dtype=np.float32)
    for i, zi in enumerate(z):
        H = np.zeros_like(F0, dtype=np.complex64)
        H[mask_prop] = np.exp(1j * KZ[mask_prop] * zi).astype(np.complex64)
        Uz = np.fft.ifft2(F0 * H)
        Iz = (Uz * Uz.conj()).real.astype(np.float32)
        I_xz[i, :] = Iz[Nx // 2, :]  # center row (y=0) vs x
        I_axis[i] = Iz[Nx // 2, Nx // 2]  # on-axis (x=0,y=0)
    # Normalize to max = 1 for readability
    maxI = float(np.max(I_xz))
    if maxI > 0:
        I_xz /= maxI
        I_axis /= maxI
    return x, z, I_xz, I_axis


# -------------------- App --------------------
def main():
    # ---- Load & grayscale ----
    path = pick_file()
    img = io.imread(path)
    gray0 = to_grayscale_float(img)
    H, W = gray0.shape

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
    nyq_cmm = 1.0 / (2.0 * mm_per_px)

    # Image aspect ratio (height/width)
    img_box_aspect = H / W

    # ---- Initial state ----
    state = {
        "gray0": gray0,
        "um_per_px": um_per_px,
        "mm_per_px": mm_per_px,
        "nyq_cmm": nyq_cmm,
        "bg_mm": 0.0,
        "low_cmm": 0.0,
        "high_cmm": nyq_cmm,
        "v_mps": 1497.0,  # water @ ~25 °C
        "use_speed": True,
        "fmin_MHz": 1.0,
        "fmax_MHz": 10.0,
        "intensity_case": True,  # standing-wave intensity by default
        "line_pts": None,  # ((x0,y0),(x1,y1))
        "diam_pts": None,  # ((x0,y0),(x1,y1)) for transducer diameter
        "D_mm": None,  # measured diameter
        "secax": None,
        "metrics_text": None,
        "selecting_line": False,
        "selecting_diam": False,
        "hint_text": None,
        # Simulation cache
        "sim_x": None, "sim_z": None, "sim_Ixz": None, "sim_Iaxis": None,
        "sim_z_multN": 2.0,  # z max = sim_z_multN * N
        "sim_span_multD": 4.0  # lateral span = sim_span_multD * D
    }

    # -------------------- Figure layout (native pixels for top image) --------------------
    # Margins & spacing
    LEFT, RIGHT, TOP, BOTTOM = 0.055, 0.975, 0.95, 0.07
    WSPACE = 0.10
    HSPACE = 0.12

    # Choose figure DPI and absolute inch sizes:
    DPI = 100  # 1 image pixel -> 1 figure pixel at this DPI

    # Column widths in inches: left = image native width, right = control panel width
    left_w_in = W / DPI
    right_w_in = 5.2  # good width to avoid slider overlap

    # Row heights in inches: top = image native height; others fixed but roomy
    top_h_in = H / DPI
    spec_h_in = 2.4
    sim_h_in = 3.0

    # Compute overall figure size from absolute inches and relative margins
    FIG_W = (left_w_in + right_w_in) / (RIGHT - LEFT)
    FIG_H = (top_h_in + spec_h_in + sim_h_in) / (TOP - BOTTOM)

    # Ratios for GridSpec (proportional to inch sizes)
    WIDTH_RATIOS = [left_w_in, right_w_in]
    HEIGHT_RATIOS = [top_h_in, spec_h_in, sim_h_in]

    plt.close("all")
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)

    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        width_ratios=WIDTH_RATIOS, height_ratios=HEIGHT_RATIOS,
        left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
        wspace=WSPACE, hspace=HSPACE
    )
    ax_img = fig.add_subplot(gs[0, 0])  # TOP: filtered image (native size)
    ax_spec = fig.add_subplot(gs[1, 0])  # MIDDLE: 1D FFT along line
    ax_sim = fig.add_subplot(gs[2, 0])  # BOTTOM: x–z intensity map
    ax_panel = fig.add_subplot(gs[:, 1]);
    ax_panel.axis("off")

    # enforce exact aspect on the image axes (no stretching)
    try:
        ax_img.set_box_aspect(img_box_aspect)
    except Exception:
        pass

    # Hint overlay on the image (drawn inside image axes, no overlap with controls)
    state["hint_text"] = ax_img.text(
        0.02, 0.98, "",
        transform=ax_img.transAxes, va="top", ha="left",
        fontsize=9, color="tab:blue",
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )

    # --------- Control panel stacking helper (prevents overlap) ---------
    panel_pos = ax_panel.get_position()
    px, py = panel_pos.x0, panel_pos.y0
    pw, ph = panel_pos.width, panel_pos.height

    # ===== Dedicated RESULTS panel at the very top (separate, readable, no overlap) =====
    # Reserve ~22% of the right panel height for a framed results box
    metrics_h = 0.22 * ph
    metrics_ax = fig.add_axes([px + 0.03 * pw, panel_pos.y1 - metrics_h - 0.01, 0.94 * pw, metrics_h])
    metrics_ax.set_facecolor("#f7f7f7")
    for s in metrics_ax.spines.values():
        s.set_visible(True)
        s.set_color("#bcbcbc")
    metrics_ax.set_xticks([])
    metrics_ax.set_yticks([])
    metrics_ax.set_title("Results", loc="left", fontsize=10, pad=4)
    state["metrics_text"] = metrics_ax.text(
        0.02, 0.96, "",
        va="top", ha="left", fontsize=9, family="monospace",
        transform=metrics_ax.transAxes
    )

    # --------- Control panel stacking helper (controls start BELOW the results box) ---------
    y_pos = [metrics_ax.get_position().y0 - 0.02]

    def add_box(rel_w=0.95, h=0.055, gap=0.014):
        w = pw * rel_w
        x = px + (pw - w) / 2
        ax = fig.add_axes([x, y_pos[0], w, h])
        y_pos[0] -= (h + gap)
        return ax

    # Frequency (intensity) range sliders (MHz)
    sl_fmin = Slider(add_box(), "Min f_int (MHz)", 1.0, 10.0, valinit=1.0)
    sl_fmax = Slider(add_box(), "Max f_int (MHz)", 1.0, 10.0, valinit=10.0)

    # Speed textbox
    txt_ax = add_box(h=0.06)
    txt = TextBox(txt_ax, "Speed m/s:", initial=f"{state['v_mps']:.0f}")

    # Simulation range sliders (compact)
    sim_z_ax = add_box(h=0.05)
    sl_sim_z = Slider(sim_z_ax, "Sim z_max ×N", 0.5, 4.0, valinit=state["sim_z_multN"])
    sim_x_ax = add_box(h=0.05)
    sl_sim_x = Slider(sim_x_ax, "Sim width ×D", 2.0, 8.0, valinit=state["sim_span_multD"])

    # Toggles
    chk_ax = add_box(h=0.08)
    chk = CheckButtons(chk_ax, ["Show Hz axis"], [state["use_speed"]])

    # Buttons
    btn_line_ax = add_box(h=0.06)
    btn_line = Button(btn_line_ax, "Reselect line")

    btn_diam_ax = add_box(h=0.06)
    btn_diam = Button(btn_diam_ax, "Measure diameter")

    btn_sim_ax = add_box(h=0.06)
    btn_sim = Button(btn_sim_ax, "Simulate field")

    btn_eq_ax = add_box(h=0.06)
    btn_eq = Button(btn_eq_ax, "Show equations")

    # ---- Equations popup ----
    def show_equations(_event=None):
        eq = [
            "Calibration:",
            "  d_px = sqrt((x2-x1)^2 + (y2-y1)^2)",
            "  mm_per_px = d_real_mm / d_px;  um_per_px = 1000 * mm_per_px",
            "  Nyquist (cycles/mm) = 1 / (2 * mm_per_px)",
            "",
            "Background removal:",
            "  sigma_px = radius_mm / mm_per_px",
            "  I_bg = Gaussian_sigma(I);  I' = I - I_bg",
            "",
            "2D band-pass (FFT):",
            "  F = FFT2(I')",
            "  r = sqrt(u^2 + v^2)  (u,v in cycles/pixel)",
            "  low_cpx  = low_cmm  * mm_per_px",
            "  high_cpx = high_cmm * mm_per_px",
            "  F_bp = F * 1[ low_cpx <= r <= high_cpx ]",
            "  I_bp = IFFT2(F_bp)",
            "",
            "Line profile & 1D FFT:",
            "  p[n] = profile(I_bp along chosen line)",
            "  P[k] = FFT( (p - mean(p)) * hann )",
            "  nu_px[k] = k/N  (cycles/pixel);  nu_mm = nu_px / mm_per_px",
            "  nu_peak = argmax_{k>0} |P[k]|",
            "",
            "Wavelengths:",
            "  lambda_field_mm     = 1 / nu_peak",
            "  lambda_intensity_mm = 2 / nu_peak   (since I ~ cos^2)",
            "  lambda_from_f_int   = v / f_intensity * 1e3  (mm)",
            "",
            "Temporal frequency (speed v in m/s):",
            "  f_field     = v * (nu_peak * 1000)",
            "  f_intensity = v * (nu_peak * 1000) / 2",
            "",
            "Near/Far field for circular piston (unfocused):",
            "  D = diameter (mm) measured on image",
            "  Rayleigh length: N_mm ≈ D^2 / (4 * lambda_field_mm)",
        ]
        fig2 = plt.figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111);
        ax2.axis("off")
        ax2.text(0.02, 0.98, "\n".join(eq), va="top", ha="left",
                 family="monospace", fontsize=10)
        fig2.suptitle("Equations used", fontsize=12)
        fig2.tight_layout()
        fig2.show()

    btn_eq.on_clicked(show_equations)

    # ---- Processing helpers ----
    def compute_filtered_image():
        g0 = state["gray0"]
        # Band-pass only by INTENSITY frequency range mapped to spatial c/mm
        g_f = bandpass_2d_fft(g0, state["mm_per_px"], state["low_cmm"], state["high_cmm"])
        return normalize01(g_f), g_f  # (disp, filtered)

    def spectrum_from_line(g_filtered_disp):
        if state["line_pts"] is None:
            return None
        (x0, y0), (x1, y1) = state["line_pts"]
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
        lam_direct = (1.0 / fpk) if fpk > 0 else np.inf
        lam_int = (2.0 / fpk) if fpk > 0 else np.inf

        v = state["v_mps"] if state["use_speed"] else None
        f_field = f_int = None
        if v and np.isfinite(lam_direct):
            lam_d_m = lam_direct * 1e-3;
            lam_i_m = lam_int * 1e-3
            f_field = v / lam_d_m if lam_d_m > 0 else None
            f_int = v / lam_i_m if lam_i_m > 0 else None
        return fcmm, mag, fpk, lam_direct, lam_int, f_field, f_int

    def set_top_axis():
        if state["secax"] is not None:
            try:
                state["secax"].remove()
            except Exception:
                pass
            state["secax"] = None
        if not state["use_speed"] or (state["v_mps"] is None) or state["v_mps"] <= 0:
            fig.canvas.draw_idle();
            return
        v = state["v_mps"]

        # INTENSITY mapping: f_intensity(Hz) = v * (nu_cmm * 1000) / 2
        def cmm_to_Hz(x):
            return v * (x * 1000.0) / 2.0

        def Hz_to_cmm(x):
            return (x * 2.0) / (v * 1000.0)

        label = "Temporal frequency (Hz) — INTENSITY"
        sec = ax_spec.secondary_xaxis('top', functions=(cmm_to_Hz, Hz_to_cmm))
        sec.set_xlabel(label)
        sec.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda val, _pos: f"{val / 1e6:.3g} MHz" if abs(val) >= 1e6 else f"{val:.3g} Hz"
        ))
        state["secax"] = sec
        fig.canvas.draw_idle()

    def near_field_length(D_mm, lambda_field_mm):
        if (D_mm is None) or (lambda_field_mm is None) or not np.isfinite(lambda_field_mm) or lambda_field_mm <= 0:
            return None
        return (D_mm ** 2) / (4.0 * lambda_field_mm)

    def update_metrics(fpk, lam_direct, lam_int, f_field, f_int):
        lam_from_fint_mm = None
        if f_int is not None and f_int > 0:
            lam_from_fint_mm = (state["v_mps"] / f_int) * 1e3  # mm
        N_mm = near_field_length(state["D_mm"], lam_int if np.isfinite(lam_int) else None)
        lines = [
            f"Pixel size   : {state['um_per_px']:.4g} µm/px",
            f"Nyquist      : {state['nyq_cmm']:.4g} c/mm",
            f"Peak (c/mm)  : {fpk:.4g}",
            f"λ_direct     : {lam_direct:.4g} mm",
            f"λ_intensity  : {lam_int:.4g} mm",
            f"λ_from f_int : {lam_from_fint_mm:.4g} mm" if lam_from_fint_mm else "λ_from f_int : —",
            f"Speed (m/s)  : {state['v_mps']:.4g} {'[ON]' if state['use_speed'] else '[OFF]'}",
            f"f_field      : {f_field / 1e6:.4g} MHz" if f_field is not None else "f_field      : —",
            f"f_intensity  : {f_int / 1e6:.4g} MHz" if f_int is not None else "f_intensity  : —",
            f"D (diameter) : {state['D_mm']:.4g} mm" if state['D_mm'] else "D (diameter) : —",
            f"N (near fld) : {N_mm:.4g} mm" if N_mm else "N (near fld) : —",
        ]
        state["metrics_text"].set_text("\n".join(lines))

    # ---- Redraw pipeline ----
    def update_all(_=None):
        # read intensity frequency range (MHz) and convert to spatial c/mm using speed
        fmin = float(sl_fmin.val)
        fmax = float(sl_fmax.val)
        if fmax <= fmin + 1e-12:
            fmax = min(fmin + 0.01, 10.0);
            sl_fmax.set_val(fmax)
        state["fmin_MHz"], state["fmax_MHz"] = fmin, fmax

        v = state["v_mps"] if state["use_speed"] else None
        if (v is None) or (v <= 0):
            # if no valid speed, disable filtering
            low_cmm = 0.0
            high_cmm = state["nyq_cmm"]
        else:
            # f_intensity(Hz) -> spatial cycles/mm: ν = (2 f) / (1000 v)
            low_cmm = (2.0 * fmin * 1e6) / (1000.0 * v)
            high_cmm = (2.0 * fmax * 1e6) / (1000.0 * v)
            # clip to Nyquist
            low_cmm = max(0.0, min(low_cmm, state["nyq_cmm"]))
            high_cmm = max(0.0, min(high_cmm, state["nyq_cmm"]))
            if high_cmm <= low_cmm + 1e-12:
                high_cmm = min(low_cmm + 1e-6, state["nyq_cmm"])
        state["low_cmm"], state["high_cmm"] = low_cmm, high_cmm

        state["sim_z_multN"] = float(sl_sim_z.val)
        state["sim_span_multD"] = float(sl_sim_x.val)

        disp, _ = compute_filtered_image()

        # Image (top) — native pixels, no scaling
        ax_img.clear()
        try:
            ax_img.set_box_aspect(img_box_aspect)
        except Exception:
            pass
        ax_img.imshow(disp, cmap="gray", interpolation="nearest", aspect="equal")
        ax_img.set_title("Filtered image (f_int range)")
        ax_img.set_axis_off()
        # line overlay
        if state["line_pts"] is not None:
            (x0, y0), (x1, y1) = state["line_pts"]
            ax_img.plot([x0, x1], [y0, y1], "-r", lw=2)
            ax_img.scatter([x0, x1], [y0, y1], c="yellow", s=40)
        # diameter overlay
        if state["diam_pts"] is not None:
            (dx0, dy0), (dx1, dy1) = state["diam_pts"]
            ax_img.plot([dx0, dx1], [dy0, dy1], "-g", lw=2)
            ax_img.scatter([dx0, dx1], [dy0, dy1], c="lime", s=40)
        # hint
        hint_txt = "Click TWO points to set the LINE" if state["selecting_line"] else \
            "Click TWO points to measure DIAMETER" if state["selecting_diam"] else ""
        state["hint_text"].set_text(hint_txt)
        state["hint_text"].set_visible(bool(hint_txt))

        # Spectrum (middle)
        ax_spec.clear()
        ax_spec.set_title("1D FFT along selected line")
        ax_spec.set_xlabel("Spatial frequency (cycles/mm)")
        ax_spec.set_ylabel("Magnitude")
        ax_spec.set_xlim(0, state["nyq_cmm"])
        ax_spec.grid(True, alpha=0.3)

        fpk = 0.0;
        lam_direct = np.inf;
        lam_int = np.inf;
        f_field = None;
        f_int = None
        out = spectrum_from_line(disp)
        if out is not None:
            fc, mag, fpk, lam_direct, lam_int, f_field, f_int = out
            ax_spec.plot(fc, mag, lw=1.8)
            if np.isfinite(fpk) and fpk > 0:
                ax_spec.axvline(fpk, ls="--", lw=1)
                ytext = (max(mag) if len(mag) else 1.0) * 0.9
                ax_spec.text(
                    fpk, ytext,
                    f"""peak ≈ {fpk:.3g} c/mm
λ={lam_direct:.3g} mm (direct)
λ={lam_int:.3g} mm (int)""",
                    rotation=90, va="top", ha="right", fontsize=9
                )
        set_top_axis()
        update_metrics(fpk, lam_direct, lam_int, f_field, f_int)

        # Simulation (bottom)
        ax_sim.clear()
        ax_sim.set_title("Near/Far field simulation (x–z, normalized intensity)")
        ax_sim.set_xlabel("Lateral position x (mm)")
        ax_sim.set_ylabel("Axial distance z (mm)")
        if (state["sim_Ixz"] is not None) and (state["sim_x"] is not None) and (state["sim_z"] is not None):
            Ixz = np.clip(state["sim_Ixz"], 1e-6, 1.0)
            im = ax_sim.imshow(10.0 * np.log10(Ixz),
                               extent=[state["sim_x"][0], state["sim_x"][-1],
                                       state["sim_z"][-1], state["sim_z"][0]],
                               aspect="auto", cmap="viridis")
            N_mm = near_field_length(state["D_mm"], lam_int if np.isfinite(lam_int) else None)
            if N_mm and np.isfinite(N_mm):
                ax_sim.axhline(N_mm, color="w", lw=1.2, ls="--")
                ax_sim.text(state["sim_x"][0], N_mm, "  Rayleigh N", color="w", va="bottom", ha="left")
            cax = fig.add_axes([ax_sim.get_position().x1 + 0.004,
                                ax_sim.get_position().y0,
                                0.01,
                                ax_sim.get_position().height])
            cb = plt.colorbar(im, cax=cax)
            cb.set_label("Intensity (dB, 0 dB = max)")

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
        if label == "Show Hz axis":
            state["use_speed"] = not state["use_speed"]
        update_all()

    chk.on_clicked(on_check_clicked)

    sl_fmin.on_changed(update_all)
    sl_fmax.on_changed(update_all)
    sl_sim_z.on_changed(lambda _v: None)
    sl_sim_x.on_changed(lambda _v: None)

    # Selection modes
    def start_line_selection():
        state["selecting_line"] = True
        state["selecting_diam"] = False
        state["line_pts"] = None
        update_all()

    def start_diam_selection():
        state["selecting_diam"] = True
        state["selecting_line"] = False
        state["diam_pts"] = None
        state["D_mm"] = None
        update_all()

    btn_line.on_clicked(lambda _e: start_line_selection())
    btn_diam.on_clicked(lambda _e: start_diam_selection())

    # Simulate field (button): uses current D and λ_int
    def run_simulation(_e=None):
        # Need diameter and wavelength
        disp, _ = compute_filtered_image()
        lam_int = None
        out = spectrum_from_line(disp)
        if out is not None:
            _, _, _, lam_direct, lam_int_val, _, _ = out
            lam_int = lam_int_val if np.isfinite(lam_int_val) else None
        D_mm = state["D_mm"]
        if (D_mm is None) or (lam_int is None):
            ax_sim.clear()
            ax_sim.set_title("Near/Far field simulation — need DIAMETER and λ_int")
            ax_sim.set_xlabel("Lateral position x (mm)")
            ax_sim.set_ylabel("Axial distance z (mm)")
            fig.canvas.draw_idle()
            return

        # Rayleigh distance
        N_mm = near_field_length(D_mm, lam_int)
        if (N_mm is None) or (N_mm <= 0):
            return

        # Determine ranges from sliders
        z_max_mm = max(0.25 * N_mm, state["sim_z_multN"] * N_mm)
        span_mult = max(2.0, state["sim_span_multD"])

        # Run angular spectrum sim
        x, z, Ixz, Iaxis = simulate_piston_field(D_mm, lam_int, z_max_mm,
                                                 span_mult_D=span_mult, Nx=256, Nz=180)
        state["sim_x"], state["sim_z"], state["sim_Ixz"], state["sim_Iaxis"] = x, z, Ixz, Iaxis

        # Draw immediately
        update_all()

        # Overlay on-axis curve as an inset
        try:
            bbox = ax_sim.get_position()
            inset = fig.add_axes([bbox.x1 - 0.26, bbox.y1 - 0.22, 0.24, 0.18])
            inset.plot(z, Iaxis, lw=1.5)
            inset.set_title("On-axis intensity", fontsize=8)
            inset.set_xlabel("z (mm)", fontsize=8)
            inset.set_ylabel("norm", fontsize=8)
            inset.tick_params(labelsize=7)
            N_mm = near_field_length(D_mm, lam_int)
            if N_mm and np.isfinite(N_mm):
                inset.axvline(N_mm, color="r", lw=1, ls="--")
        except Exception:
            pass

        fig.canvas.draw_idle()

    btn_sim.on_clicked(run_simulation)

    # Mouse clicks for selections
    click_buffer = []

    def on_mouse_click(event):
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return
        if state["selecting_line"]:
            click_buffer.append((event.xdata, event.ydata))
            ax_img.scatter([event.xdata], [event.ydata], c="cyan", s=40)
            fig.canvas.draw_idle()
            if len(click_buffer) >= 2:
                state["line_pts"] = (click_buffer[0], click_buffer[1])
                click_buffer.clear()
                state["selecting_line"] = False
                update_all()
        elif state["selecting_diam"]:
            click_buffer.append((event.xdata, event.ydata))
            ax_img.scatter([event.xdata], [event.ydata], c="lime", s=40)
            fig.canvas.draw_idle()
            if len(click_buffer) >= 2:
                p0, p1 = click_buffer[0], click_buffer[1]
                state["diam_pts"] = (p0, p1)
                click_buffer.clear()
                state["selecting_diam"] = False
                # compute D in mm
                dx = p1[0] - p0[0];
                dy = p1[1] - p0[1]
                d_px = float(np.hypot(dx, dy))
                state["D_mm"] = d_px * state["mm_per_px"]
                update_all()

    fig.canvas.mpl_connect('button_press_event', on_mouse_click)

    # ---- First draw ----
    state["selecting_line"] = True  # ask for a line once so spectrum isn't empty
    update_all()
    plt.show()


if __name__ == "__main__":
    main()
