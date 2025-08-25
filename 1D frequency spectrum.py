#!/usr/bin/env python3
"""
Wave-frequency analyzer with:
- 2-point physical calibration (mm -> µm/px)
- Background removal (local-mean subtraction via Gaussian, radius in mm)
- Band-pass filtering (low/high cutoffs in cycles/mm) for both image (2D) and line profile (1D)
- Standing-wave helpers (intensity case -> λ_field = 2/ν_peak)

Steps:
1) Pick an image.
2) CALIBRATION: click two points with known real distance; enter that distance in mm.
3) CLEANUP:
   - Enter background radius (mm) for local-mean subtraction (0 = skip).
   - Enter low/high cutoffs (cycles/mm) for band-pass (blank = default).
4) ANALYSIS: click two points to define the line.
5) View: filtered image+line, 1D profile (orig vs filtered), 1D FFT (orig vs filtered, cycles/mm),
   and 2D FFT (log) of the filtered image with axes in cycles/mm.
6) Optionally enter wave speed (m/s) to estimate temporal frequency (Hz/MHz).

Requirements: numpy, matplotlib, scikit-image, tkinter
    pip install numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tkinter import Tk, filedialog, simpledialog, messagebox
from skimage import io, color
from skimage.filters import gaussian
from skimage.measure import profile_line


# ---------- Small helpers ----------
def info(msg, title="Info"):
    root = Tk(); root.withdraw(); root.update()
    messagebox.showinfo(title, msg)
    root.destroy()

def ask_float(title, prompt, initial=None, minvalue=None):
    """Fixed: uses 'minvalue' (matches tkinter.simpledialog.askfloat)."""
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

def select_two_points(gray, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.set_title(title + "\n(click TWO points, then close this window)")
    ax.set_axis_off()
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise SystemExit("Two points not selected.")
    (x0, y0), (x1, y1) = pts
    return (x0, y0), (x1, y1)

def normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    mn, mx = np.min(a), np.max(a)
    if mx > mn:
        return (a - mn) / (mx - mn)
    return np.zeros_like(a)

def to_grayscale_float(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img


# ---------- Frequency-domain helpers ----------
def compute_2d_fft(gray):
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    mag = np.abs(Fshift)
    mag_log = np.log1p(mag)
    mag_log /= (mag_log.max() + 1e-12)
    return mag_log

def bandpass_2d_fft(image, mm_per_px, low_cmm=None, high_cmm=None):
    """
    Ideal (hard) annular mask in 2D frequency domain.
    low_cmm, high_cmm are cutoffs in cycles/mm. Use None/0 to disable a side.
    """
    H, W = image.shape
    # Frequency grids in cycles/pixel
    fx = np.fft.fftshift(np.fft.fftfreq(W, d=1.0))
    fy = np.fft.fftshift(np.fft.fftfreq(H, d=1.0))
    FX, FY = np.meshgrid(fx, fy)
    R_c_per_px = np.sqrt(FX**2 + FY**2)  # radial frequency in cycles/pixel

    # cycles/mm -> cycles/pixel
    def to_c_per_px(cmm):
        return cmm * mm_per_px

    low_cpx = to_c_per_px(low_cmm) if (low_cmm is not None and low_cmm > 0) else 0.0
    high_cpx = to_c_per_px(high_cmm) if (high_cmm is not None and high_cmm > 0) else None

    F = np.fft.fft2(image)
    Fshift = np.fft.fftshift(F)

    mask = np.ones_like(R_c_per_px, dtype=bool)
    if low_cpx > 0:
        mask &= (R_c_per_px >= low_cpx)
    if high_cpx is not None:
        mask &= (R_c_per_px <= high_cpx)

    Fshift_f = Fshift * mask
    If = np.fft.ifft2(np.fft.ifftshift(Fshift_f)).real
    return If

def gaussian_background_subtract(image, radius_mm, mm_per_px):
    """
    Subtract local mean estimated by Gaussian blur with sigma = radius_px.
    radius_mm = 0 or None -> skip (return original).
    """
    if radius_mm is None or radius_mm <= 0:
        return image
    sigma_px = max(radius_mm / mm_per_px, 0.0)
    if sigma_px < 1e-6:
        return image
    bg = gaussian(image, sigma=sigma_px, preserve_range=True)
    out = image - bg
    return out

def line_fft(profile):
    n = len(profile)
    profile = profile - np.mean(profile)
    if n > 1:
        # light taper to reduce leakage
        w = np.hanning(n)
        profile = profile * w
    spec = np.fft.fft(profile)
    freqs_px = np.fft.fftfreq(n, d=1.0)  # cycles per pixel
    mag = np.abs(spec) / max(n, 1)
    return freqs_px, mag

def bandpass_1d_fft(profile, mm_per_px, low_cmm=None, high_cmm=None):
    """
    Band-pass the 1D profile via FFT masking.
    """
    n = len(profile)
    spec = np.fft.fft(profile - np.mean(profile))
    freqs_px = np.fft.fftfreq(n, d=1.0)  # cycles/pixel

    def to_c_per_px(cmm):
        return cmm * mm_per_px

    low_cpx = to_c_per_px(low_cmm) if (low_cmm is not None and low_cmm > 0) else 0.0
    high_cpx = to_c_per_px(high_cmm) if (high_cmm is not None and high_cmm > 0) else None

    mask = np.ones_like(freqs_px, dtype=bool)
    if low_cpx > 0:
        mask &= (np.abs(freqs_px) >= low_cpx)
    if high_cpx is not None:
        mask &= (np.abs(freqs_px) <= high_cpx)

    spec_f = spec * mask
    prof_f = np.fft.ifft(spec_f).real
    return prof_f


# ---------- Main ----------
def main():
    # 1) Image
    path = pick_file()
    img = io.imread(path)
    gray0 = to_grayscale_float(img)

    # 2) CALIBRATION: two points + known distance (mm)
    (cx0, cy0), (cx1, cy1) = select_two_points(
        gray0, "CALIBRATION: choose two points with a known real distance"
    )
    pixel_distance = float(np.hypot(cx1 - cx0, cy1 - cy0))  # px
    known_mm = ask_float(
        "Calibration distance",
        "Enter known distance between the two points (mm):",
        initial=1.0, minvalue=1e-12
    )
    if known_mm is None:
        raise SystemExit("Calibration distance is required.")
    um_per_px = (known_mm * 1000.0) / pixel_distance   # µm/px
    mm_per_px = um_per_px / 1000.0                     # mm/px
    px_m = mm_per_px * 1e-3                            # m/px

    # Nyquist (cycles/mm)
    nyq_cmm = 1.0 / (2.0 * mm_per_px)

    # 3) CLEANUP PARAMETERS
    bg_radius_mm = ask_float(
        "Background removal",
        "Enter local-mean radius (mm) for background subtraction (0 = skip):",
        initial=0.2, minvalue=0.0
    )
    cuts = ask_string(
        "Band-pass cutoffs",
        f"Enter low,high cutoffs in cycles/mm (comma-separated).\n"
        f"Examples: 0, {nyq_cmm:.3g}  (no HP / no LP)\n"
        f"           0.2, 4\n"
        f"Leave blank for default (0 to Nyquist ≈ {nyq_cmm:.3g})."
    )
    low_cmm = 0.0
    high_cmm = nyq_cmm
    if cuts:
        try:
            parts = [p.strip() for p in cuts.split(",")]
            if len(parts) >= 1 and parts[0] != "":
                low_cmm = max(0.0, float(parts[0]))
            if len(parts) >= 2 and parts[1] != "":
                high_cmm = float(parts[1])
        except Exception:
            info("Could not parse cutoffs. Using defaults.", "Band-pass")
    if high_cmm is None or high_cmm <= 0:
        high_cmm = nyq_cmm
    low_cmm = max(0.0, min(low_cmm, nyq_cmm))
    high_cmm = max(low_cmm + 1e-9, min(high_cmm, nyq_cmm))  # ensure low < high

    # Apply cleanup to image: background removal -> 2D band-pass
    gray_bg = gaussian_background_subtract(gray0, bg_radius_mm, mm_per_px)
    gray1 = bandpass_2d_fft(gray_bg, mm_per_px, low_cmm=low_cmm, high_cmm=high_cmm)
    gray_disp = normalize01(gray1)

    # Optional wave speed (for temporal freq estimate)
    speed_str = ask_string(
        "Wave speed (optional)",
        "Enter wave speed in m/s (e.g., 1480 water, 1540 tissue).\nLeave blank to skip:"
    )
    v_mps = None
    if speed_str:
        try:
            v_mps = float(speed_str)
            if v_mps <= 0:
                v_mps = None
        except ValueError:
            info("Could not parse wave speed. Proceeding without it.", "Wave speed")

    # 4) ANALYSIS LINE
    (x0, y0), (x1, y1) = select_two_points(
        gray_disp, "ANALYSIS: choose two points to define the line"
    )

    # Profiles: original + filtered-image
    prof_orig = profile_line(gray0, (y0, x0), (y1, x1),
                             mode="reflect", order=1, linewidth=1, reduce_func=None)
    if prof_orig.ndim > 1:
        prof_orig = prof_orig.mean(axis=1)
    prof_imgfilt = profile_line(gray_disp, (y0, x0), (y1, x1),
                                mode="reflect", order=1, linewidth=1, reduce_func=None)
    if prof_imgfilt.ndim > 1:
        prof_imgfilt = prof_imgfilt.mean(axis=1)

    # Additional 1D band-pass on the profile (helps for noisy lines)
    prof_bp = bandpass_1d_fft(prof_imgfilt, mm_per_px, low_cmm=low_cmm, high_cmm=high_cmm)

    # FFTs (orig vs filtered)
    freqs_px_o, mag_o = line_fft(prof_orig)
    freqs_px_f, mag_f = line_fft(prof_bp)
    freqs_cmm_o = freqs_px_o / mm_per_px
    freqs_cmm_f = freqs_px_f / mm_per_px

    # Dominant non-DC peak from filtered
    if len(mag_f) > 1:
        idx_peak = 1 + np.argmax(mag_f[1:])
    else:
        idx_peak = 0
    fpk_cmm = max(freqs_cmm_f[idx_peak], 0.0)

    # Wavelengths
    lambda_direct_mm = (1.0 / fpk_cmm) if fpk_cmm > 0 else np.inf
    lambda_intensity_mm = (2.0 / fpk_cmm) if fpk_cmm > 0 else np.inf  # standing-wave intensity case

    # Optional temporal frequency estimate
    f_direct_Hz = f_intensity_Hz = None
    if v_mps and np.isfinite(lambda_direct_mm):
        lambda_direct_m = lambda_direct_mm * 1e-3
        lambda_intensity_m = lambda_intensity_mm * 1e-3
        f_direct_Hz = v_mps / lambda_direct_m if lambda_direct_m > 0 else None
        f_intensity_Hz = v_mps / lambda_intensity_m if lambda_intensity_m > 0 else None

    # 2D FFT (of filtered image) with physical axes
    f2d = compute_2d_fft(gray1)
    H, W = gray1.shape
    fx_m = np.fft.fftshift(np.fft.fftfreq(W, d=px_m))  # cycles/m
    fy_m = np.fft.fftshift(np.fft.fftfreq(H, d=px_m))
    fx_mm = fx_m / 1000.0
    fy_mm = fy_m / 1000.0
    extent = [fx_mm[0], fx_mm[-1], fy_mm[0], fy_mm[-1]]

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.imshow(gray_disp, cmap="gray", interpolation="nearest")
    ax.plot([x0, x1], [y0, y1], "-r", lw=2)
    ax.scatter([x0, x1], [y0, y1], c="yellow", s=40)
    ax.set_title("Filtered image (bg removed + band-pass) with analysis line")
    ax.set_axis_off()

    ax = axes[0, 1]
    im = ax.imshow(f2d, cmap="gray", extent=extent, origin="lower", aspect="auto")
    ax.set_title("2D FFT magnitude (log) of filtered image")
    ax.set_xlabel("fx (cycles/mm)")
    ax.set_ylabel("fy (cycles/mm)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Norm. log magnitude")

    ax = axes[1, 0]
    x_pix = np.linspace(0, 1, len(prof_orig)) * len(prof_orig)
    ax.plot(x_pix, prof_orig, lw=1.1, label="Original")
    ax.plot(x_pix, prof_bp, lw=1.4, label="Filtered (bg+BP)")
    ax.set_xlabel("Distance along line (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title("Line intensity profile (orig vs filtered)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(freqs_cmm_o, mag_o, lw=1.0, label="Original")
    ax.plot(freqs_cmm_f, mag_f, lw=1.6, label="Filtered (bg+BP)")
    ax.set_xlim(0, nyq_cmm)
    ax.set_xlabel("Spatial frequency (cycles/mm)")
    ax.set_ylabel("Magnitude")
    ax.set_title("1D FFT along line")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    if np.isfinite(fpk_cmm) and fpk_cmm > 0:
        ax.axvline(fpk_cmm, ls="--", lw=1)
        ax.text(fpk_cmm, max(mag_f)*0.9, f"peak ≈ {fpk_cmm:.3g} c/mm",
                rotation=90, va="top", ha="right")

    # Optional top x-axis for temporal frequency if v given
    if v_mps:
        def cmm_to_Hz(x):
            return v_mps * (x * 1000.0)  # cycles/mm -> cycles/m -> Hz
        def Hz_to_cmm(x):
            return x / (v_mps * 1000.0)
        secax = ax.secondary_xaxis('top', functions=(cmm_to_Hz, Hz_to_cmm))
        secax.set_xlabel("Temporal frequency (Hz)")
        def fmt_Hz(val, _pos):
            return f"{val/1e6:.3g} MHz" if abs(val) >= 1e6 else f"{val:.3g} Hz"
        secax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_Hz))

    plt.tight_layout()
    plt.show()

    # ---- Console summary ----
    print("\n=== Calibration ===")
    print(f"Pixel distance: {pixel_distance:.6g} px")
    print(f"Known distance: {known_mm:.6g} mm")
    print(f"Pixel size: {um_per_px:.6g} µm/px  ({mm_per_px:.6g} mm/px)")

    print("\n=== Cleanup parameters ===")
    print(f"Background radius: {bg_radius_mm if bg_radius_mm is not None else 0:.6g} mm  "
          f"(sigma ≈ {(bg_radius_mm/mm_per_px) if (bg_radius_mm and mm_per_px) else 0:.3g} px)")
    print(f"Band-pass: low={low_cmm:.6g} c/mm, high={high_cmm:.6g} c/mm  (Nyquist ≈ {nyq_cmm:.6g} c/mm)")

    print("\n=== Analysis results (from filtered profile) ===")
    print(f"Dominant spatial peak: {fpk_cmm:.6g} cycles/mm")
    print(f"λ_direct (1/peak): {lambda_direct_mm:.6g} mm")
    print(f"λ_field (intensity case, 2/peak): {lambda_intensity_mm:.6g} mm")
    if v_mps:
        if f_direct_Hz:
            print(f"Estimated f_direct: {f_direct_Hz/1e6:.6g} MHz")
        if f_intensity_Hz:
            print(f"Estimated f_intensity (standing-wave): {f_intensity_Hz/1e6:.6g} MHz")

if __name__ == "__main__":
    main()
