#!/usr/bin/env python3
"""
BOS‑helper GUI
────────────────────────────────────────────────────────────────────────
• BOS geometry calculator  (schematic drawn to canvas)
• Checkerboard PNG export (A4 @ 600 dpi, 2 px squares default)
• Remove‑blank‑images utility  (delete every n‑th)
• Image‑subtraction tool  (scrub filters, calibration & measurements,
  export filtered video & annotated snapshot)

Requires:  Python ≥3.8,  Tkinter, OpenCV‑Python, NumPy
"""

import math, os, glob, shutil, itertools
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import numpy as np
import cv2
from PIL import Image, ImageTk


# ───────────────────────── BOS formulas ──────────────────────────────
def sensitivity(f, za, zd):
    if za <= f or zd <= 0:
        raise ValueError("Need ZA>f and ZD>0")
    denom = zd + za - f
    if abs(denom) < 1e-12:
        raise ValueError("Geometry denominator≈0")
    return f * zd / denom


def coc_object(S, fnum, sensor, fov):
    """circle‑of‑confusion in object plane (mm)."""
    return S / (fnum * (1 + sensor / fov))


def blur_loss(CoC, obj, k=1 / 3):
    """empirical loss factor"""
    return math.exp(-k * CoC / obj)


def solve_ZD(S, f, ZA):
    den = S - f
    if abs(den) < 1e-12:
        raise ValueError("Cannot solve ZD (S≈f)")
    zd = S * (f - ZA) / den
    if zd <= 0:
        raise ValueError("Solved ZD ≤ 0")
    return zd


def solve_ZA(S, f, ZD):
    za = (f * ZD + S * (f - ZD)) / S
    if za <= f:
        raise ValueError("Solved ZA ≤ f")
    return za


# ───────────── Checkerboard generator (600 dpi, A4) ──────────────────
def generate_checkerboard(square_px, gap_px, fn):
    dpi = 600
    mm2in = 1 / 25.4
    width = int(210 * dpi * mm2in)
    height = int(297 * dpi * mm2in)
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    pitch = square_px + gap_px
    for i in range(0, height, pitch):
        for j in range(0, width, pitch):
            if ((i // pitch) + (j // pitch)) % 2 == 0:
                img[i : i + square_px, j : j + square_px] = 0
    cv2.imwrite(fn, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# ─────────────────────────── Main GUI ────────────────────────────────
class BOSGui(tk.Tk):
    PAD = 5

    # ───────────────── init ──────────────────
    def __init__(self):
        super().__init__()
        self.title("BOS Setup Helper")
        self._scrub_state = None

        self._make_vars()
        self._build_layout()
        self._build_menu()

    # ─────────────── helpers ──────────────────
    def _varf(self, key):
        txt = self.v[key].get().strip()
        return None if not txt else float(txt)

    # ───────── build widgets ──────────────────
    def _make_vars(self):
        keys = (
            "fov",
            "sensor",
            "npx",
            "obj",
            "fnum",
            "focal",
            "ZA",
            "ZD",
            "ZB",
            "square",
            "gap",
        )
        self.v = {k: tk.StringVar() for k in keys}
        # sensible defaults
        self.v["fov"].set("20")
        self.v["sensor"].set("13.6")
        self.v["npx"].set("1024")
        self.v["obj"].set("0.75")
        self.v["fnum"].set("32")
        self.v["focal"].set("110")
        self.v["ZB"].set("290")
        self.v["square"].set("2")
        self.v["gap"].set("1")
        self.goal = tk.StringVar(value="resolution")

    def _build_layout(self):
        frm = ttk.Frame(self, padding=self.PAD)
        frm.grid()

        # numeric inputs
        rows = [
            ("FOV [mm]", "fov"),
            ("Sensor [mm]", "sensor"),
            ("Pixels", "npx"),
            ("Object size [mm]", "obj"),
            ("f‑number", "fnum"),
            ("Focal length [mm]", "focal"),
            ("ZA [mm]", "ZA"),
            ("ZD [mm]", "ZD"),
            ("ZB [mm]", "ZB"),
            ("Checker square [px]", "square"),
            ("Checker gap [px]", "gap"),
        ]
        for i, (lbl, key) in enumerate(rows):
            ttk.Label(frm, text=lbl).grid(row=i, column=0, sticky="w")
            ttk.Entry(frm, width=10, textvariable=self.v[key]).grid(
                row=i, column=1, padx=(0, self.PAD)
            )

        rb = len(rows)
        ttk.Label(frm, text="Optimise for").grid(row=rb, column=0, sticky="w")
        ttk.Radiobutton(frm, text="Resolution", value="resolution", variable=self.goal).grid(
            row=rb, column=1, sticky="w"
        )
        ttk.Radiobutton(frm, text="Signal", value="signal", variable=self.goal).grid(
            row=rb, column=2, sticky="w"
        )

        ttk.Button(frm, text="Calculate", command=self.calculate).grid(
            row=rb + 1, column=0, columnspan=3, pady=(4, self.PAD), sticky="ew"
        )
        ttk.Button(frm, text="Export checkerboard", command=self.export_checker).grid(
            row=rb + 2, column=0, columnspan=3, sticky="ew"
        )

        # results text + schematic canvas
        self.txt = tk.Text(frm, width=62, height=12, bg="#f6f6f6")
        self.txt.grid(row=0, column=3, rowspan=rb + 3, padx=(self.PAD, 0))
        self.canvas = tk.Canvas(frm, width=560, height=155, bg="white", bd=1, relief="sunken")
        self.canvas.grid(row=rb + 3, column=0, columnspan=4, pady=self.PAD)

    def _build_menu(self):
        mb = tk.Menu(self)
        # tools
        tools = tk.Menu(mb, tearoff=0)
        tools.add_command(label="Remove blank images", command=self.remove_images)
        tools.add_command(label="Import images for subtraction", command=self.import_subtraction)
        mb.add_cascade(label="Tools", menu=tools)
        # help
        helpm = tk.Menu(mb, tearoff=0)
        helpm.add_command(label="Parameter help", command=self.show_help)
        mb.add_cascade(label="Help", menu=helpm)
        self.config(menu=mb)

    # ────────────────── calculation & schematic ───────────────────────
    def calculate(self):
        # grab & validate numbers
        try:
            fov = float(self.v["fov"].get())
            sensor = float(self.v["sensor"].get())
            obj = float(self.v["obj"].get())
            fnum = float(self.v["fnum"].get())
            focal = float(self.v["focal"].get())
        except ValueError:
            return messagebox.showerror("Input", "Please enter valid numbers")

        npx = self._varf("npx")
        ZA_in, ZD_in, ZB_in = self._varf("ZA"), self._varf("ZD"), self._varf("ZB")

        factor = 3.0 if self.goal.get() == "signal" else 1.0
        CoC_target = factor * obj
        S_req = CoC_target * fnum * (1 + sensor / fov)

        try:
            ZA, ZD = self._solve_geometry(S_req, focal, ZA_in, ZD_in, ZB_in)
            S = sensitivity(focal, ZA, ZD)
        except Exception as e:
            return messagebox.showerror("Geometry", str(e))

        CoC = coc_object(S, fnum, sensor, fov)
        smin = CoC / 2
        B = blur_loss(CoC, obj)
        Seff = S * B
        ratio = CoC / obj

        # build results text
        self.txt.delete("1.0", tk.END)
        t = self.txt.insert
        t(tk.END, f"ZA={ZA:.1f} mm   ZD={ZD:.1f} mm   ZB={ZA+ZD:.1f} mm\n\n")
        t(tk.END, f"S={S:.3f} mm/px   CoC={CoC:.3f} mm ({ratio:.2f}×obj)\n")
        t(tk.END, f"s_min≈{smin:.3f} mm   B={B:.3f}   S_eff={Seff:.3f}\n\n")

        if npx:
            pxmm = npx / fov
            t(tk.END, f"Sampling ≈ {pxmm:.1f} px/mm\n")
            t(tk.END, f"   →  object ≈ {obj*pxmm:.1f} px\n\n")
        t(tk.END, "Blue, green arrows show ZA & ZD. Red circle=blur CoC.")

        # schematic
        self._draw_schematic(ZA, ZD, CoC)

    def _draw_schematic(self, ZA, ZD, coc_mm):
        W, H = 560, 155
        top, base = 35, H - 30
        total = ZA + ZD
        c = self.canvas
        c.delete("all")
        if total <= 0:
            return

        def X(d):  # map mm → canvas x
            return 60 + (W - 120) * d / total

        cam_x, obj_x, bkg_x = X(0), X(ZA), X(total)
        # camera/object/background lines
        c.create_line(cam_x, top, cam_x, base, width=3)
        c.create_line(obj_x, top, obj_x, base, fill="blue", width=3)
        c.create_line(bkg_x, top, bkg_x, base, fill="green", width=3)

        arrow = dict(arrow=tk.LAST, width=2)
        c.create_line(cam_x, base, obj_x, base, fill="blue", **arrow)
        c.create_line(obj_x, base - 18, bkg_x, base - 18, fill="green", **arrow)

        # blur circle
        pxmm = (W - 120) / total
        r = max(4, min(coc_mm * pxmm / 2, 45))
        c.create_oval(obj_x - r, top - r, obj_x + r, top + r, outline="red", width=2)

        # labels
        c.create_text((cam_x + obj_x) / 2, base + 12, text=f"ZA {ZA:.0f} mm", fill="blue")
        c.create_text((obj_x + bkg_x) / 2, base - 6, text=f"ZD {ZD:.0f} mm", fill="green")
        c.create_text(obj_x, top - r - 8, text="CoC", fill="red")

    # ───────────────────────── file tools ──────────────────────────────
    def export_checker(self):
        fn = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")]
        )
        if not fn:
            return
        sz = int(self.v["square"].get())
        gap = int(self.v["gap"].get())
        generate_checkerboard(sz, gap, fn)
        messagebox.showinfo("Checkerboard", f"Saved to:\n{fn}")

    # ---- remove blank images: delete every n‑th ----------------------
    def remove_images(self):
        path = filedialog.askdirectory(title="Choose folder with images")
        if not path:
            return
        n = simpledialog.askinteger("Remove images", "Delete every n‑th image (n≥2)?", minvalue=2)
        if not n:
            return
        imgs = sorted(
            glob.glob(os.path.join(path, "*.*")),
            key=lambda x: os.path.getmtime(x)
        )
        targets = imgs[n-1::n]
        if not targets:
            return messagebox.showinfo("No files", "Nothing to delete.")
        if not messagebox.askyesno("Confirm", f"Delete {len(targets)} files?"):
            return
        for f in targets:
            os.remove(f)
        messagebox.showinfo("Done", f"Deleted {len(targets)} images.")

    # ---- subtraction import / scrub window --------------------------
    def import_subtraction(self):
        files = filedialog.askopenfilenames(
            title="Select images", filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not files:
            return
        first = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        self._scrub_state = {
            "files": list(files),
            "first": first,
            "gain": 100,
            "lp": 1,
            "hp": 1,
            "idx": 0,
            "annotated": None,
        }
        self._open_scrub()

    # ---------- scrub window ----------------------------------------
    def _open_scrub(self):
        st = self._scrub_state
        files, first = st["files"], st["first"]

        win = tk.Toplevel(self)
        win.title("Scrub & Filters")
        win.minsize(420, 320)
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        # preview label
        lbl = tk.Label(win)
        lbl.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # sliders frame
        sf = ttk.Frame(win)
        sf.grid(row=1, column=0, sticky="ew", padx=5)
        sf.grid_columnconfigure((0, 1, 2, 3), weight=1)
        sliders = {
            "Frame": tk.Scale(sf, from_=0, to=len(files) - 1, orient="horizontal", label="Frame"),
            "Gain": tk.Scale(sf, from_=0, to=15000, orient="horizontal", label="Gain x100"),
            "LP": tk.Scale(sf, from_=1, to=99, orient="horizontal", label="Low‑pass"),
            "HP": tk.Scale(sf, from_=1, to=99, orient="horizontal", label="High‑pass"),
        }
        sliders["Frame"].set(0)
        sliders["Gain"].set(100)
        sliders["LP"].set(1)
        sliders["HP"].set(1)
        for i, sc in enumerate(sliders.values()):
            sc.grid(row=0, column=i, sticky="ew", padx=2)

        # buttons
        bf = ttk.Frame(win)
        bf.grid(row=2, column=0, sticky="ew", pady=4, padx=5)
        bf.grid_columnconfigure((0, 1, 2), weight=1)
        ttk.Button(bf, text="Calibrate & Measure", command=lambda: self._calibrate_measure(st)).grid(
            row=0, column=0, sticky="ew", padx=3
        )
        ttk.Button(bf, text="Export video", command=lambda: self._export_video(st)).grid(
            row=0, column=1, sticky="ew", padx=3
        )
        ttk.Button(bf, text="Export snapshot", command=lambda: self._export_snapshot(st)).grid(
            row=0, column=2, sticky="ew", padx=3
        )

        # update preview
        def update(_=None):
            idx = sliders["Frame"].get()
            gain = sliders["Gain"].get()
            lp = sliders["LP"].get() | 1
            hp = sliders["HP"].get() | 1

            img = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
            diff = cv2.subtract(img, first)
            scaled = cv2.convertScaleAbs(diff, alpha=gain / 100.0)
            lp_img = cv2.GaussianBlur(scaled, (lp, lp), 0) if lp > 1 else scaled
            if hp > 1:
                hp_blur = cv2.GaussianBlur(lp_img, (hp, hp), 0)
                out = cv2.subtract(lp_img, hp_blur)
            else:
                out = lp_img

            # store current processed full‑res
            st.update(idx=idx, gain=gain, lp=lp, hp=hp, processed=out)

            # scaled display
            wlbl, hlbl = lbl.winfo_width(), lbl.winfo_height()
            h0, w0 = out.shape
            scale = min(wlbl / w0 if wlbl else 1, hlbl / h0 if hlbl else 1, 1.0)
            disp = cv2.resize(out, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            lbl.imgtk = imgtk
            lbl.config(image=imgtk)

        # bind
        for sc in sliders.values():
            sc.config(command=update)
        lbl.bind("<Configure>", lambda e: update())
        update()

    # -------------- calibration window -------------------------------
    def _calibrate_measure(self, st):
        if "processed" not in st:
            return
        gray = st["processed"]
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        win = "Calibration & Measurement"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(win, img)

        pts, px_per_mm = [], None

        def on_mouse(evt, x, y, flags, _):
            nonlocal px_per_mm, img
            if evt == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
                pts.append((x, y))
                cv2.imshow(win, img)

                if len(pts) % 2 == 0 and px_per_mm is not None:
                    (x1, y1), (x2, y2) = pts[-2], pts[-1]
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    dist_px = math.hypot(x2 - x1, y2 - y1)
                    dist_mm = dist_px / px_per_mm
                    cv2.putText(
                        img,
                        f"{dist_mm:.2f} mm",
                        (min(x1, x2), min(y1, y2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow(win, img)

        cv2.setMouseCallback(win, on_mouse)
        messagebox.showinfo("Calibrate", "Click 2 points to set scale")
        while len(pts) < 2:
            cv2.waitKey(50)

        real_mm = simpledialog.askfloat("Calibration", "Real distance (mm):", minvalue=0.0)
        if not real_mm:
            cv2.destroyWindow(win)
            return
        (x1, y1), (x2, y2) = pts[:2]
        px_per_mm = math.hypot(x2 - x1, y2 - y1) / real_mm
        messagebox.showinfo("Calibrated", f"{px_per_mm:.3f} px/mm\nEsc to finish measuring.")
        while True:
            if cv2.waitKey(50) & 0xFF == 27:
                break
        cv2.destroyWindow(win)
        # store annotated
        st["annotated"] = img

    # -------------- export helpers ----------------------------------
    def _export_video(self, st):
        fn = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI", "*.avi")])
        if not fn:
            return
        files, first = st["files"], st["first"]
        gain, lp, hp = st["gain"], st["lp"], st["hp"]
        h, w = st["processed"].shape
        vw = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"XVID"), 1.0, (w, h), False)
        for f in files:
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            diff = cv2.subtract(g, first)
            scaled = cv2.convertScaleAbs(diff, alpha=gain / 100.0)
            lp_img = cv2.GaussianBlur(scaled, (lp, lp), 0) if lp > 1 else scaled
            if hp > 1:
                hp_blur = cv2.GaussianBlur(lp_img, (hp, hp), 0)
                out = cv2.subtract(lp_img, hp_blur)
            else:
                out = lp_img
            vw.write(out)
        vw.release()
        messagebox.showinfo("Video", f"Saved video:\n{fn}")

    def _export_snapshot(self, st):
        out = st.get("annotated", st.get("processed"))
        if out is None:
            return messagebox.showerror("Snapshot", "Nothing to save")
        fn = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not fn:
            return
        cv2.imwrite(fn, out)
        messagebox.showinfo("Snapshot", f"Saved snapshot:\n{fn}")

    # ───────────────────── geometry helper ───────────────────────────
    def _solve_geometry(self, S_req, f, ZA, ZD, ZB):
        if ZB and (ZA or ZD):
            raise ValueError("Use ZB or ZA/ZD—not both.")
        if ZA is not None and ZD is not None:
            return ZA, ZD
        if ZA is not None:
            ZD = solve_ZD(S_req, f, ZA)
            return ZA, ZD
        if ZD is not None:
            ZA = solve_ZA(S_req, f, ZD)
            return ZA, ZD
        # nothing known → pick ZA≈ZD≈ZB/2 then scale to S_req
        ZBv = ZB or 1000
        ZA = ZBv / 2
        ZD = ZBv - ZA
        S0 = sensitivity(f, ZA, ZD)
        ZD *= S_req / S0
        ZA = ZBv - ZD
        if ZA <= f:
            raise ValueError("Cannot fit ZA>f in chosen ZB.")
        return ZA, ZD

    # ─────────────────────────── help ────────────────────────────────
    def show_help(self):
        messagebox.showinfo(
            "Parameter help",
            "FOV – object‑plane field width captured by sensor\n"
            "Sensor – physical sensor length along FOV axis\n"
            "Pixels – number of pixels along that axis\n"
            "Object – size of feature you need to resolve\n"
            "f‑number – lens f/‑ratio (aperture)\n"
            "Focal – lens focal length\n"
            "ZA – camera→object (mm), ZD – object→background (mm).\n"
            "Leave ZA or ZD blank to let program solve it.",
        )


if __name__ == "__main__":
    BOSGui().mainloop()
