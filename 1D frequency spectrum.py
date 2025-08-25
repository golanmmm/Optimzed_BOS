#!/usr/bin/env python3
"""
Wave-frequency analyzer with 2-point physical calibration.

Workflow:
1) Pick an image.
2) CALIBRATION: click two points with a known real distance; enter that distance in mm.
   -> Script computes pixel size (µm/px).
3) ANALYSIS: click two points to define a line of interest.
4) Displays:
   - Image + selected line
   - 1D intensity profile along the line
   - 1D FFT magnitude vs spatial frequency (cycles/mm)
   - 2D FFT magnitude (log) with axes in cycles/mm
5) Optionally enter wave speed (m/s) to estimate temporal frequency (Hz/MHz).

Standing-wave note:
If your image shows intensity (∝ field^2), the strongest spectral peak
often appears at ~2/λ_field. The script reports both:
  λ_direct = 1/ν_peak
  λ_field  = 2/ν_peak  (intensity case, often correct for standing waves)

Requirements: numpy, matplotlib, scikit-image, tkinter
    pip install numpy matplotlib scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, simpledialog, messagebox
from skimage import io, color
from skimage.measure import profile_line

# ---------- UI helpers ----------
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

def ask_float(title, prompt, initial=None, minval=None):
    root = Tk(); root.withdraw(); root.update()
    val = simpledialog.askfloat(title, prompt, initialvalue=initial, minvalue=minval)
    root.destroy()
    return val

def ask_string(title, prompt):
    root = Tk(); root.withdraw(); root.update()
    s = simpledialog.askstring(title, prompt)
    root.destroy()
    return s

def info(msg, title="Info"):
    root = Tk(); root.withdraw(); root.update()
    messagebox.showinfo(title, msg)
    root.destroy()

# ---------- Image / FFT helpers ----------
def to_grayscale_float(img):
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = color.rgb2gray(img)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img

def compute_2d_fft(gray):
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    mag = np.abs(Fshift)
    mag_log = np.log1p(mag)
    mag_log /= (mag_log.max() + 1e-12)
    return mag_log

def line_fft(profile):
    n = len(profile)
    profile = profile - np.mean(profile)
    spec = np.fft.fft(profile)
    freqs_px = np.fft.fftfreq(n, d=1.0)   # cycles/pixel
    mag = np.abs(spec) / max(n, 1)
    m = freqs_px >= 0
    return freqs_px[m], mag[m]

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

# ---------- Main ----------
def main():
    # 1) Image
    path = pick_file()
    img = io.imread(path)
    gray = to_grayscale_float(img)

    # 2) CALIBRATION: two points + known distance (mm)
    (cx0, cy0), (cx1, cy1) = select_two_points(
        gray, "CALIBRATION: choose two points with a known real distance"
    )
    pixel_distance = float(np.hypot(cx1 - cx0, cy1 - cy0))  # in pixels
    known_mm = ask_float("Calibration distance",
                         "Enter known distance between the two points (mm):",
                         initial=1.0, minval=1e-9)
    if known_mm is None:
        raise SystemExit("Calibration distance is required.")
    # Compute pixel size
    um_per_px = (known_mm * 1000.0) / pixel_distance   # µm/px
    px_m = um_per_px * 1e-6                             # meters/pixel

    info(f"Calibration complete:\n"
         f"Pixel distance: {pixel_distance:.3f} px\n"
         f"Known distance: {known_mm:.6g} mm\n"
         f"Pixel size: {um_per_px:.6g} µm/px")

    # Optional wave speed (for temporal frequency estimate)
    speed_str = ask_string("Wave speed (optional)",
                           "Enter wave speed in m/s (e.g., 1480 water, 1540 tissue).\nLeave blank to skip:")
    v_mps = None
    if speed_str:
        try:
            v_mps = float(speed_str)
            if v_mps <= 0:
                v_mps = None
        except ValueError:
            info("Could not parse wave speed. Proceeding without it.", "Wave speed")

    # 3) ANALYSIS LINE: two points
    (x0, y0), (x1, y1) = select_two_points(
        gray, "ANALYSIS: choose two points to define the line"
    )

    # Profile along the line
    prof = profile_line(gray, (y0, x0), (y1, x1), mode="reflect", order=1, linewidth=1, reduce_func=None)
    if prof.ndim > 1:
        prof = prof.mean(axis=1)

    # 1D FFT along the line -> cycles/mm
    freqs_px, mag = line_fft(prof)
    freqs_c_per_mm = (freqs_px / px_m) / 1000.0  # cycles/mm

    # Dominant non-DC peak
    if len(mag) > 1:
        idx_peak = 1 + np.argmax(mag[1:])
    else:
        idx_peak = 0
    fpk_c_per_mm = max(freqs_c_per_mm[idx_peak], 0.0)

    # Wavelengths
    lambda_direct_mm = (1.0 / fpk_c_per_mm) if fpk_c_per_mm > 0 else np.inf
    lambda_intensity_mm = (2.0 / fpk_c_per_mm) if fpk_c_per_mm > 0 else np.inf  # standing-wave intensity case

    # Optional temporal frequency estimate
    f_direct_Hz = f_intensity_Hz = None
    if v_mps and np.isfinite(lambda_direct_mm):
        lambda_direct_m = lambda_direct_mm * 1e-3
        lambda_intensity_m = lambda_intensity_mm * 1e-3
        f_direct_Hz = v_mps / lambda_direct_m if lambda_direct_m > 0 else None
        f_intensity_Hz = v_mps / lambda_intensity_m if lambda_intensity_m > 0 else None

    # 2D FFT with physical frequency axes (cycles/mm)
    f2d = compute_2d_fft(gray)
    H, W = gray.shape
    fx_m = np.fft.fftshift(np.fft.fftfreq(W, d=px_m))  # cycles/m
    fy_m = np.fft.fftshift(np.fft.fftfreq(H, d=px_m))  # cycles/m
    fx_mm = fx_m / 1000.0
    fy_mm = fy_m / 1000.0
    extent = [fx_mm[0], fx_mm[-1], fy_mm[0], fy_mm[-1]]

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    ax.imshow(gray, cmap="gray", interpolation="nearest")
    ax.plot([x0, x1], [y0, y1], "-r", lw=2)
    ax.scatter([x0, x1], [y0, y1], c="yellow", s=40)
    ax.set_title("Image (with analysis line)")
    ax.set_axis_off()

    ax = axes[0, 1]
    im = ax.imshow(f2d, cmap="gray", extent=extent, origin="lower", aspect="auto")
    ax.set_title("2D FFT magnitude (log)")
    ax.set_xlabel("fx (cycles/mm)")
    ax.set_ylabel("fy (cycles/mm)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Norm. log magnitude")

    ax = axes[1, 0]
    x_pix = np.linspace(0, 1, len(prof)) * len(prof)
    ax.plot(x_pix, prof, lw=1.5)
    ax.set_xlabel("Distance along line (pixels)")
    ax.set_ylabel("Intensity")
    ax.set_title("Line intensity profile")

    ax = axes[1, 1]
    ax.plot(freqs_c_per_mm, mag, lw=1.8)
    ax.set_xlabel("Spatial frequency (cycles/mm)")
    ax.set_ylabel("Magnitude")
    ax.set_title("1D FFT along line")
    ax.grid(True, alpha=0.3)
    if np.isfinite(fpk_c_per_mm) and fpk_c_per_mm > 0:
        ax.axvline(fpk_c_per_mm, ls="--", lw=1)
        ax.text(fpk_c_per_mm, max(mag)*0.9, f"peak ≈ {fpk_c_per_mm:.3g} c/mm",
                rotation=90, va="top", ha="right")

    # Top x-axis as temporal frequency (if speed given)
    if v_mps:
        def cmm_to_Hz(x):
            return v_mps * (x * 1000.0)  # cycles/mm -> cycles/m -> Hz
        def Hz_to_cmm(x):
            return x / (v_mps * 1000.0)
        secax = ax.secondary_xaxis('top', functions=(cmm_to_Hz, Hz_to_cmm))
        secax.set_xlabel("Temporal frequency (Hz)")
        import matplotlib.ticker as mticker
        def fmt_Hz(val, _pos):
            return f"{val/1e6:.3g} MHz" if abs(val) >= 1e6 else f"{val:.3g} Hz"
        secax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_Hz))

    plt.tight_layout()
    plt.show()

    # ---- Console summary ----
    print("\n=== Calibration ===")
    print(f"Pixel distance: {pixel_distance:.6g} px")
    print(f"Known distance: {known_mm:.6g} mm")
    print(f"Pixel size: {um_per_px:.6g} µm/px  ({px_m:.3e} m/px)")

    print("\n=== Analysis results ===")
    print(f"Dominant spatial peak: {fpk_c_per_mm:.6g} cycles/mm")
    print(f"λ_direct (1/peak): {lambda_direct_mm:.6g} mm")
    print(f"λ_field (intensity case, 2/peak): {lambda_intensity_mm:.6g} mm")
    if v_mps:
        if f_direct_Hz:
            print(f"Estimated f_direct: {f_direct_Hz/1e6:.6g} MHz")
        if f_intensity_Hz:
            print(f"Estimated f_intensity (standing-wave): {f_intensity_Hz/1e6:.6g} MHz")

if __name__ == "__main__":
    main()
