#!/usr/bin/env python3
"""
bos_gui.py  –  BOS‑setup helper with checkerboard & image tools
------------------------------------------------------------------------
• Tkinter GUI (Python ≥3.8)
• BOS geometry calculator + PDF export
• Checkerboard PNG export
• Tools:
    – Remove blank images
    – Import images for subtraction → interactive scrub/gain/filters +
      measurement + final filtered‑video export + snapshot
"""

import os
import math
import glob
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ─── BOS formulas ───────────────────────────────────────────────────
def sensitivity(f, ZA, ZD):
    if ZA <= f or ZD <= 0:
        raise ValueError("Require ZA > focal and ZD > 0.")
    denom = ZD + ZA - f
    if abs(denom) < 1e-12:
        raise ValueError("Geometry denominator ≈ 0.")
    return f * ZD / denom

def coc_object(S, fnum, sensor, fov):
    return S / (fnum * (1 + sensor / fov))

blur_loss = lambda CoC, obj, k=1/3: math.exp(-k * CoC / obj)

def solve_ZD(S, f, ZA):
    den = S - f
    if abs(den) < 1e-12:
        raise ValueError("Cannot solve ZD (S≈f).")
    ZD = S * (f - ZA) / den
    if ZD <= 0:
        raise ValueError("Solved ZD ≤ 0.")
    return ZD

def solve_ZA(S, f, ZD):
    ZA = (f * ZD + S * (f - ZD)) / S
    if ZA <= f:
        raise ValueError("Solved ZA ≤ f.")
    return ZA

# ─── Checkerboard generator ─────────────────────────────────────────
def generate_checkerboard_image(square_size, spacing, output_file):
    dpi = 600
    mm2in = 1 / 25.4
    w_px = int(210 * dpi * mm2in)
    h_px = int(297 * dpi * mm2in)
    cols = w_px // (square_size + spacing)
    rows = h_px // (square_size + spacing)
    img = np.full((h_px, w_px, 3), 255, dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            top = i * (square_size + spacing)
            left = j * (square_size + spacing)
            color = (0, 0, 0) if (i + j) % 2 == 0 else (255, 255, 255)
            br = (left + square_size, top + square_size)
            cv2.rectangle(img, (left, top), br, color, -1)
    cv2.imwrite(output_file, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# ─── Main GUI ───────────────────────────────────────────────────────
class BOSGui(tk.Tk):
    PAD = 5

    def __init__(self):
        super().__init__()
        self.title("BOS Setup Calculator")

        menubar = tk.Menu(self)
        tools = tk.Menu(menubar, tearoff=0)
        tools.add_command(label="Remove blank images", command=self.remove_images)
        tools.add_command(label="Import images for subtraction", command=self.import_subtraction)
        menubar.add_cascade(label="Tools", menu=tools)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label="Parameters Explanation", command=self.show_help)
        menubar.add_cascade(label="Help", menu=helpm)

        self.config(menu=menubar)
        self._make_vars()
        self._build_gui()

    def _make_vars(self):
        keys = ("fov","sensor","npx","obj","fnum","focal","ZA","ZD","ZB","square_size","spacing")
        self.v = {k: tk.StringVar() for k in keys}
        # defaults
        self.v["fov"].set("20")
        self.v["sensor"].set("13.6")
        self.v["npx"].set("1024")
        self.v["obj"].set("0.75")
        self.v["fnum"].set("32")
        self.v["focal"].set("110")
        self.v["ZB"].set("290")
        self.v["square_size"].set("2")
        self.v["spacing"].set("1")
        self.goal = tk.StringVar(value="resolution")

    def _build_gui(self):
        frm = ttk.Frame(self, padding=self.PAD)
        frm.grid()
        rows = [
            ("FOV [mm]",      "fov"),
            ("Sensor [mm]",   "sensor"),
            ("Pixels",        "npx"),
            ("Object [mm]",   "obj"),
            ("f‑number",      "fnum"),
            ("Focal [mm]",    "focal"),
            ("ZA [mm]",       "ZA"),
            ("ZD [mm]",       "ZD"),
            ("ZB [mm]",       "ZB"),
            ("Checker sq [px]","square_size"),
            ("Checker gp [px]","spacing"),
        ]
        for i, (lbl, k) in enumerate(rows):
            ttk.Label(frm, text=lbl).grid(row=i, column=0, sticky="w")
            ttk.Entry(frm, width=12, textvariable=self.v[k]).grid(row=i, column=1, padx=(0,self.PAD))

        rb = len(rows)
        ttk.Label(frm, text="Goal").grid(row=rb, column=0, sticky="w")
        for c, (txt, val) in enumerate((("Signal","signal"),("Resolution","resolution"))):
            ttk.Radiobutton(frm, text=txt, variable=self.goal, value=val).grid(row=rb, column=1+c)

        ttk.Button(frm, text="Calculate", command=self.calculate).grid(row=rb+1, column=0, columnspan=2, pady=(3,self.PAD))
        ttk.Button(frm, text="Export PDF", command=self.export_pdf).grid(row=rb+1, column=2, columnspan=2, pady=(3,self.PAD))
        ttk.Button(frm, text="Export Checkerboard", command=self.export_checkerboard).grid(row=rb+2, column=0, columnspan=4)

        self.txt = tk.Text(frm, width=64, height=12, bg="#f5f5f5", relief="sunken")
        self.txt.grid(row=0, column=3, rowspan=rb+3, padx=(self.PAD,0))

        self.canvas = tk.Canvas(frm, width=560, height=155, bg="white", bd=1, relief="solid")
        self.canvas.grid(row=rb+3, column=0, columnspan=5, pady=self.PAD)

    def _num(self, key):
        t = self.v[key].get().strip()
        return None if not t else float(t)

    def _solve_geometry(self, S_req, f, ZA, ZD, ZB):
        if ZB and (ZA or ZD):
            raise ValueError("Use ZB or ZA/ZD, not both.")
        if ZA is not None and ZD is not None:
            if ZB and abs((ZA+ZD)-ZB) > 1e-6:
                raise ValueError("ZA+ZD ≠ ZB.")
            return ZA, ZD
        if ZA is not None:
            z = solve_ZD(S_req, f, ZA)
            if ZB and ZA+z > ZB+1e-6:
                raise ValueError("ZA+ZD > ZB.")
            return ZA, z
        if ZD is not None:
            a = solve_ZA(S_req, f, ZD)
            if ZB and a+ZD > ZB+1e-6:
                raise ValueError("ZA+ZD > ZB.")
            return a, ZD
        ZBv = ZB or max(6*f,1000)
        ZA_v = max(f*1.2, ZBv/2)
        ZD_v = ZBv - ZA_v
        S0 = sensitivity(f, ZA_v, ZD_v)
        ZD_v *= (S_req/S0)
        ZA_v = ZBv - ZD_v
        if ZA_v <= f:
            raise ValueError("Cannot fit ZA>f within ZB.")
        return ZA_v, ZD_v

    # ────────────────────────── Calculation ─────────────────────────────
    def calculate(self):
        try:
            fov, sensor, obj, fnum, focal = (
                float(self.v["fov"].get()),
                float(self.v["sensor"].get()),
                float(self.v["obj"].get()),
                float(self.v["fnum"].get()),
                float(self.v["focal"].get())
            )
        except ValueError:
            return messagebox.showerror("Input","Enter valid numbers")

        npx   = self._num("npx")
        ZA_in = self._num("ZA"); ZD_in = self._num("ZD"); ZB_in = self._num("ZB")
        goal  = self.goal.get()
        factor = 3.0 if goal=="signal" else 1.0
        CoC_t  = factor * obj
        S_req  = CoC_t * fnum * (1 + sensor/fov)

        try:
            ZA, ZD = self._solve_geometry(S_req, focal, ZA_in, ZD_in, ZB_in)
            S       = sensitivity(focal, ZA, ZD)
        except Exception as e:
            return messagebox.showerror("Geometry", str(e))

        CoC   = coc_object(S, fnum, sensor, fov)
        s_min = CoC/2
        B     = blur_loss(CoC, obj)
        S_eff = S * B
        ratio = CoC / obj

        pxmm   = npx/fov if npx else None
        objpx  = obj * pxmm if pxmm else None
        sq_mm  = 2.0/pxmm if pxmm else None
        gp_mm  = 1.0/pxmm if pxmm else None

        self.txt.delete(1.0, tk.END)
        add = self.txt.insert
        add(tk.END, f"ZA={ZA:.2f} mm  ZD={ZD:.2f} mm  ZB={ZA+ZD:.2f} mm\n\n")
        add(tk.END, f"S={S:.3f} mm/px  CoC={CoC:.3f} mm ({ratio:.2f}×obj)\n")
        add(tk.END, f"s_min≈{s_min:.3f} mm  B={B:.3f}  S_eff={S_eff:.3f}\n\n")
        if pxmm:
            add(tk.END, f"Sampling:{pxmm:.2f} px/mm  obj={objpx:.1f} px\n")
            add(tk.END, f"Square={sq_mm:.3f} mm  Gap={gp_mm:.3f} mm\n\n")
        add(tk.END, "Advice: tweak ZD, focal or f‑number to optimize blur.")

        self._draw_schematic(ZA, ZD, CoC)

    # ─────────────────────────── file ops ──────────────────────────────
    def export_pdf(self):
        npx = self._num("npx"); fov = float(self.v["fov"].get())
        if not npx or fov <= 0:
            return messagebox.showerror("Export","Enter pixels & FOV")
        ss = int(self.v["square_size"].get()); sp = int(self.v["spacing"].get())
        dpi = 600; mm2in = 1/25.4
        Wpx = int(210*dpi*mm2in); Hpx = int(297*dpi*mm2in)
        tile = np.full((ss+sp,ss+sp),255,dtype=np.uint8); tile[:ss,:ss]=0
        pattern = np.tile(tile,(Hpx//tile.shape[0]+1, Wpx//tile.shape[1]+1))[:Hpx,:Wpx]
        fn = filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")])
        if not fn: return
        fig = plt.figure(figsize=(210*mm2in,297*mm2in),dpi=dpi)
        ax = fig.add_axes([0,0,1,1])
        ax.imshow(pattern, cmap="gray",vmin=0,vmax=255,interpolation="nearest")
        ax.axis("off")
        fig.savefig(fn, dpi=dpi, format="pdf", pad_inches=0)
        plt.close(fig)
        messagebox.showinfo("Exported", f"PDF saved:\n{fn}")

    def export_checkerboard(self):
        ss = int(self.v["square_size"].get()); sp = int(self.v["spacing"].get())
        fn = filedialog.asksaveasfilename(defaultextension=".png",filetypes=[("PNG","*.png")])
        if not fn: return
        generate_checkerboard_image(ss, sp, fn)
        messagebox.showinfo("Done", f"PNG saved:\n{fn}")

    def remove_images(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder:
            return
        n = simpledialog.askinteger("Delete every Nth","Enter N:",minvalue=1)
        if not n:
            return
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder,e)))
        files.sort()
        deleted = 0
        for i, f in enumerate(files, start=1):
            if i % n == 0:
                try:
                    os.remove(f)
                    deleted += 1
                except:
                    pass
        messagebox.showinfo("Done", f"Deleted {deleted}/{len(files)} images")

    # ───────────────────────── Import & Measure ─────────────────────────
    def import_subtraction(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Image files",("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"))]
        )
        if not files:
            return
        self.subtract_files = list(files)
        self.first_gray = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        self.measure_distance()  # directly launch

    def measure_distance(self):
        files = getattr(self, 'subtract_files', None)
        first = getattr(self, 'first_gray', None)
        if not files or first is None:
            return messagebox.showerror("Error","No subtraction data")

        count = len(files)
        win = "Scrub & Filters (s=select, Esc=cancel)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        sel_idx  = [0]
        gain_val = [100]
        lp_val   = [1]
        hp_val   = [1]

        def update(idx):
            gray = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
            diff = cv2.subtract(gray, first)
            scaled = cv2.convertScaleAbs(diff, alpha=gain_val[0]/100.0)
            # low‑pass
            k_lp = lp_val[0] | 1
            lp_img = cv2.GaussianBlur(scaled,(k_lp,k_lp),0) if k_lp>1 else scaled
            # high‑pass
            k_hp = hp_val[0] | 1
            if k_hp>1:
                hp_blur = cv2.GaussianBlur(lp_img,(k_hp,k_hp),0)
                out = cv2.subtract(lp_img, hp_blur)
            else:
                out = lp_img
            disp = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            cv2.imshow(win, disp)
            self.current_display = disp.copy()

        cv2.createTrackbar("Frame",    win, 0,   count-1, lambda v:(sel_idx.__setitem__(0,v), update(v)))
        cv2.createTrackbar("Gain x100",win,100, 2000,    lambda v:(gain_val.__setitem__(0,v), update(sel_idx[0])))
        cv2.createTrackbar("Low‑pass", win, 1,   99,      lambda v:(lp_val.__setitem__(0,max(1,v)), update(sel_idx[0])))
        cv2.createTrackbar("High‑pass",win, 1,   99,      lambda v:(hp_val.__setitem__(0,max(1,v)), update(sel_idx[0])))

        update(0)
        while True:
            k = cv2.waitKey(50) & 0xFF
            if k == ord('s'):
                idx = sel_idx[0]
                break
            if k == 27:
                idx = None
                break

        cv2.destroyWindow(win)
        if idx is None:
            return

        img = self.current_display.copy()
        pts = []
        cv2.namedWindow("Measure", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Measure", img)

        def on_mouse(e, x, y, flags, param):
            if e == cv2.EVENT_LBUTTONDOWN:
                pts.append((x,y))
                cv2.circle(img, (x,y), 5, (0,0,255), -1)
                if len(pts)%2 == 0 and 'px_per_mm' in locals():
                    (x1,y1),(x2,y2) = pts[-2], pts[-1]
                    px = math.hypot(x2-x1, y2-y1)
                    mm = px/px_per_mm
                    cv2.line(img, (x1,y1),(x2,y2), (0,255,0), 2)
                    cv2.putText(img, f"{mm:.2f} mm", (min(x1,x2),min(y1,y2)-10),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                cv2.imshow("Measure", img)

        cv2.setMouseCallback("Measure", on_mouse)
        messagebox.showinfo("Calibrate", "Click 2 points to set scale")
        while len(pts) < 2:
            cv2.waitKey(50)
        (cx1,cy1),(cx2,cy2) = pts[:2]
        px_dist = math.hypot(cx2-cx1, cy2-cy1)
        real_mm = simpledialog.askfloat("Calibrate","Enter real distance (mm):", minvalue=0.0)
        if real_mm is None:
            cv2.destroyWindow("Measure")
            return
        px_per_mm = px_dist/real_mm
        messagebox.showinfo("Calibrated", f"{px_per_mm:.3f} px/mm")
        messagebox.showinfo("Measure", "Click pairs to measure. Esc when done.")

        while True:
            if cv2.waitKey(50) & 0xFF == 27:
                break

        # final filtered video save
        save_vid = filedialog.asksaveasfilename(defaultextension=".avi",
            filetypes=[("AVI","*.avi"),("MP4","*.mp4")])
        if save_vid:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            h, w, _ = self.current_display.shape
            vw = cv2.VideoWriter(save_vid, fourcc, 1.0, (w,h), False)
            for f in files:
                gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                diff = cv2.subtract(gray, first)
                scaled = cv2.convertScaleAbs(diff, alpha=gain_val[0]/100.0)
                k_lp = lp_val[0] | 1
                lp_img = cv2.GaussianBlur(scaled,(k_lp,k_lp),0) if k_lp>1 else scaled
                k_hp = hp_val[0] | 1
                if k_hp>1:
                    hp_blur = cv2.GaussianBlur(lp_img,(k_hp,k_hp),0)
                    out = cv2.subtract(lp_img, hp_blur)
                else:
                    out = lp_img
                vw.write(out)
            vw.release()
            messagebox.showinfo("Saved", f"Filtered video saved to:\n{save_vid}")

        # snapshot
        snap = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if snap:
            cv2.imwrite(snap, img)
            messagebox.showinfo("Saved", f"Snapshot saved to:\n{snap}")
        cv2.destroyWindow("Measure")

    # ──────────────────────────── Schematic & Help ───────────────────────────
    def _draw_schematic(self, ZA, ZD, CoC):
        c = self.canvas; c.delete("all")
        W, H = 560, 155; top, base = 40, H-30; total = ZA + ZD
        X = lambda d: 60 + d/total*(W-120)
        cam, obj, bkg = X(0), X(ZA), X(total)
        for x, col in ((cam,"black"), (obj,"blue"), (bkg,"green")):
            c.create_line(x, top, x, base, fill=col, width=3)
        arrow = dict(arrow=tk.LAST, width=2)
        c.create_line(cam, base, obj, base, fill="blue", **arrow)
        c.create_line(obj, base-20, bkg, base-20, fill="green", **arrow)
        pxmm = (W-120)/total; r_px = max(4, min(CoC*pxmm/2,45))
        c.create_oval(obj-r_px, top-r_px, obj+r_px, top+r_px, outline="red", width=2)
        c.create_text((cam+obj)/2, base+12, text=f"ZA {ZA:.0f} mm", fill="blue")
        c.create_text((obj+bkg)/2, base-8, text=f"ZD {ZD:.0f} mm", fill="green")
        c.create_text(obj, top-r_px-8, text="blur", fill="red")

    def show_help(self):
        messagebox.showinfo("Parameters Explanation",
            "FOV      – field‑of‑view width in object plane (mm)\n"
            "Sensor   – sensor size along FOV axis (mm)\n"
            "Pixels   – number of pixels along that axis\n"
            "Object   – target feature size (mm)\n"
            "f‑number – lens aperture ratio\n"
            "focal    – lens focal length (mm)\n"
            "ZA       – camera→object distance (mm)\n"
            "ZD       – object→background distance (mm)\n"
            "ZB       – total camera→background span (mm)\n"
            "Checker sq/spacing – checkerboard px sizes\n\n"
            "Tools:\n"
            "  • Remove blank images\n"
            "  • Import images for subtraction → scrub/gain/filters + measure"
        )

if __name__ == "__main__":
    BOSGui().mainloop()
