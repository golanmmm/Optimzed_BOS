#!/usr/bin/env python3
"""
bos_gui.py  –  BOS‑setup helper with checkerboard & image cleanup tools
------------------------------------------------------------------------
• Tkinter GUI (Python ≥3.8)
• BOS geometry calculator + PDF export
• Checkerboard PNG export: 2 px black / 1 px white gaps at 600 DPI A4
• “Remove blank images” tool: delete every Nth image in a folder
"""

import os
import math
import glob
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ─── BOS core formulas ──────────────────────────────────────────────
def sensitivity(f: float, ZA: float, ZD: float) -> float:
    if ZA <= f or ZD <= 0:
        raise ValueError("Require ZA > focal and ZD > 0.")
    denom = ZD + ZA - f
    if abs(denom) < 1e-12:
        raise ValueError("Geometry denominator ≈ 0.")
    return f * ZD / denom

def coc_object(S: float, fnum: float, sensor: float, fov: float) -> float:
    return S / (fnum * (1 + sensor / fov))

blur_loss = lambda CoC, obj, k=1/3: math.exp(-k * CoC / obj)

def solve_ZD(S: float, f: float, ZA: float) -> float:
    den = S - f
    if abs(den) < 1e-12:
        raise ValueError("Cannot solve ZD (S ≈ f).")
    ZD = S * (f - ZA) / den
    if ZD <= 0:
        raise ValueError("Solved ZD ≤ 0.")
    return ZD

def solve_ZA(S: float, f: float, ZD: float) -> float:
    ZA = (f * ZD + S * (f - ZD)) / S
    if ZA <= f:
        raise ValueError("Solved ZA ≤ f.")
    return ZA

# ─── Checkerboard generator ─────────────────────────────────────────
def generate_checkerboard_image(square_size: int, spacing: int, output_file: str):
    """Create A4 checkerboard PNG at 600 DPI with given px square/gap."""
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
            color = (0,0,0) if (i+j)%2==0 else (255,255,255)
            br = (left+square_size, top+square_size)
            cv2.rectangle(img, (left, top), br, color, -1)
    cv2.imwrite(output_file, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# ─── Main GUI ───────────────────────────────────────────────────────
class BOSGui(tk.Tk):
    PAD = 5

    def __init__(self):
        super().__init__()
        self.title("BOS Setup Calculator")
        # Menu bar with Help
        menubar = tk.Menu(self)
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Parameters Explanation", command=self.show_help)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

        self._make_vars()
        self._build_gui()

    def _make_vars(self):
        keys = ("fov","sensor","npx","obj","fnum","focal","ZA","ZD","ZB",
                "square_size","spacing")
        self.v = {k: tk.StringVar() for k in keys}
        # updated defaults
        self.v["fov"].set("20")           # 20 mm
        self.v["sensor"].set("13.6")      # 13.6 mm
        self.v["npx"].set("1024")         # 1024 px
        self.v["obj"].set("0.75")         # 0.75 mm
        self.v["fnum"].set("32")          # f/32
        self.v["focal"].set("110")        # 110 mm
        self.v["ZB"].set("290")           # 290 mm total
        self.v["square_size"].set("2")    # 2 px
        self.v["spacing"].set("1")        # 1 px
        self.goal = tk.StringVar(value="resolution")

    def _build_gui(self):
        frm = ttk.Frame(self, padding=self.PAD); frm.grid()
        rows = [
            ("FOV [mm]"                , "fov"),
            ("Sensor size [mm]"        , "sensor"),
            ("Sensor pixels (FOV)"     , "npx"),
            ("Object size [mm]"        , "obj"),
            ("f‑number"                , "fnum"),
            ("Focal length f [mm]"     , "focal"),
            ("ZA (cam→obj) [mm]"       , "ZA"),
            ("ZD (obj→bkg) [mm]"       , "ZD"),
            ("Total ZB [mm]"           , "ZB"),
            ("Checker square [px]"     , "square_size"),
            ("Checker spacing [px]"    , "spacing"),
        ]
        for r,(lbl,key) in enumerate(rows):
            ttk.Label(frm, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(frm, width=12, textvariable=self.v[key]).grid(
                row=r, column=1, padx=(0,self.PAD), pady=1)

        rb = len(rows)
        ttk.Label(frm, text="Goal").grid(row=rb, column=0, sticky="w")
        for c,(txt,val) in enumerate((("Signal","signal"),("Resolution","resolution"))):
            ttk.Radiobutton(frm, text=txt, variable=self.goal, value=val
            ).grid(row=rb, column=1+c)

        ttk.Button(frm, text="Calculate", command=self.calculate).grid(
            row=rb+1, column=0, columnspan=2, pady=(3,self.PAD))
        ttk.Button(frm, text="Export PDF", command=self.export_pdf).grid(
            row=rb+1, column=2, columnspan=2, pady=(3,self.PAD))
        ttk.Button(frm, text="Export Checkerboard", command=self.export_checkerboard).grid(
            row=rb+2, column=0, columnspan=4, pady=(self.PAD,0))
        ttk.Button(frm, text="Remove blank images tool", command=self.remove_images).grid(
            row=rb+3, column=0, columnspan=4, pady=(self.PAD,0))

        self.txt = tk.Text(frm, width=64, height=12, bg="#f5f5f5", relief="sunken")
        self.txt.grid(row=0, column=3, rowspan=rb+4, padx=(self.PAD,0))

        self.canvas = tk.Canvas(frm, width=560, height=155, bg="white",
                                relief="solid", bd=1)
        self.canvas.grid(row=rb+4, column=0, columnspan=5, pady=(self.PAD,0))

    def _num(self, key):
        txt = self.v[key].get().strip()
        return None if txt=="" else float(txt)

    def _solve_geometry(self, S, f, ZA, ZD, ZB):
        if ZB and (ZA or ZD):
            raise ValueError("If ZB set, leave ZA & ZD blank.")
        if ZA is not None and ZD is not None:
            if ZB and abs((ZA+ZD)-ZB)>1e-6:
                raise ValueError("ZA+ZD ≠ ZB.")
            return ZA, ZD
        if ZA is not None:
            z = solve_ZD(S,f,ZA)
            if ZB and ZA+z>ZB+1e-6:
                raise ValueError("ZA+ZD > ZB.")
            return ZA, z
        if ZD is not None:
            a = solve_ZA(S,f,ZD)
            if ZB and a+ZD>ZB+1e-6:
                raise ValueError("ZA+ZD > ZB.")
            return a, ZD
        ZBv = ZB or max(6*f,1000)
        ZA_v = max(f*1.2, ZBv/2)
        ZD_v = ZBv - ZA_v
        S0 = sensitivity(f, ZA_v, ZD_v)
        scale = S/S0
        ZD_v *= scale
        ZA_v = ZBv - ZD_v
        if ZA_v<=f:
            raise ValueError("Cannot fit ZA > f within ZB.")
        return ZA_v, ZD_v

    def calculate(self):
        try:
            fov=float(self.v["fov"].get()); sensor=float(self.v["sensor"].get())
            obj=float(self.v["obj"].get()); fnum=float(self.v["fnum"].get())
            focal=float(self.v["focal"].get())
        except ValueError:
            messagebox.showerror("Input error","Enter valid numbers"); return

        npx=self._num("npx"); ZA_in=self._num("ZA"); ZD_in=self._num("ZD"); ZB_in=self._num("ZB")
        goal=self.goal.get()
        factor=3.0 if goal=="signal" else 1.0
        CoC_t=factor*obj; S_req=CoC_t*fnum*(1+sensor/fov)

        try:
            ZA,ZD=self._solve_geometry(S_req,focal,ZA_in,ZD_in,ZB_in)
            S=sensitivity(focal,ZA,ZD)
        except ValueError as e:
            messagebox.showerror("Geometry error",str(e)); return

        CoC=coc_object(S,fnum,sensor,fov); s_min=CoC/2; B=blur_loss(CoC,obj); S_eff=S*B; ratio=CoC/obj
        pxmm=npx/fov if npx else None; objpx=obj*pxmm if pxmm else None
        square_mm=2.0/pxmm if pxmm else None; gap_mm=1.0/pxmm if pxmm else None

        self.txt.delete(1.0,tk.END); add=self.txt.insert
        add(tk.END,f"ZA={ZA:.2f} mm  ZD={ZD:.2f} mm  ZB={ZA+ZD:.2f} mm\n\n")
        add(tk.END,f"S={S:.3f} mm/px deflection\n")
        add(tk.END,f"CoC={CoC:.3f} mm ({ratio:.2f}×obj)\n")
        add(tk.END,f"s_min≈CoC/2={s_min:.3f} mm\n")
        add(tk.END,f"B={B:.3f}  S_eff={S_eff:.3f} mm\n\n")
        if pxmm:
            add(tk.END,f"Sampling:{pxmm:.2f} px/mm  obj={objpx:.1f} px\n")
            add(tk.END,f"Square={square_mm:.3f} mm (2 px)  Gap={gap_mm:.3f} mm (1 px)\n\n")
        add(tk.END,"Advice: tweak ZD, focal or f# to optimize blur.\n")
        self._draw_schematic(ZA,ZD,CoC)

    def export_pdf(self):
        npx=self._num("npx"); fov=float(self.v["fov"].get())
        if not npx or fov<=0:
            messagebox.showerror("Export error","Enter sensor pixels & FOV"); return

        ss=int(self.v["square_size"].get()); sp=int(self.v["spacing"].get())
        dpi=600; mm2in=1/25.4
        Wpx=int(210*dpi*mm2in); Hpx=int(297*dpi*mm2in)
        tile=np.full((ss+sp,ss+sp),255,dtype=np.uint8)
        tile[:ss,:ss]=0
        reps_y=Hpx//tile.shape[0]+1; reps_x=Wpx//tile.shape[1]+1
        pattern=np.tile(tile,(reps_y,reps_x))[:Hpx,:Wpx]

        fn=filedialog.asksaveasfilename(defaultextension=".pdf",
            filetypes=[("PDF","*.pdf")])
        if not fn: return

        fig=plt.figure(figsize=(210*mm2in,297*mm2in),dpi=dpi)
        ax=fig.add_axes([0,0,1,1])
        ax.imshow(pattern,cmap="gray",vmin=0,vmax=255,interpolation="nearest")
        ax.axis("off")
        fig.savefig(fn,dpi=dpi,format="pdf",pad_inches=0)
        plt.close(fig)
        messagebox.showinfo("Exported",f"Checkerboard PDF saved to:\n{fn}")

    def export_checkerboard(self):
        ss=int(self.v["square_size"].get()); sp=int(self.v["spacing"].get())
        fn=filedialog.asksaveasfilename(defaultextension=".png",
            filetypes=[("PNG","*.png")])
        if not fn: return
        generate_checkerboard_image(ss, sp, fn)
        messagebox.showinfo("Done",f"Checkerboard PNG saved to:\n{fn}")

    def remove_images(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder: return
        n = simpledialog.askinteger("Delete every Nth", "Enter N (e.g. 3):", minvalue=1)
        if not n: return
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        files.sort()
        total = len(files); deleted=0
        for idx, fpath in enumerate(files, start=1):
            if idx % n == 0:
                try:
                    os.remove(fpath)
                    deleted += 1
                except:
                    pass
        messagebox.showinfo("Done", f"Deleted {deleted} of {total} images in\n{folder}")

    def _draw_schematic(self, ZA, ZD, CoC):
        c=self.canvas; c.delete("all")
        W,H=560,155; top,base=40,H-30; total=ZA+ZD
        X=lambda d:60 + d/total*(W-120)
        cam,obj,bkg=X(0),X(ZA),X(total)
        for x,col in ((cam,"black"),(obj,"blue"),(bkg,"green")):
            c.create_line(x,top,x,base,width=3,fill=col)
        arr=dict(arrow=tk.LAST,width=2)
        c.create_line(cam,base,obj,base,fill="blue",**arr)
        c.create_line(obj,base-20,bkg,base-20,fill="green",**arr)
        pxmm=(W-120)/total; r_px=max(4,min(CoC*pxmm/2,45))
        c.create_oval(obj-r_px,top-r_px,obj+r_px,top+r_px,outline="red",width=2)
        c.create_text((cam+obj)/2,base+12,text=f"ZA {ZA:.0f} mm",fill="blue")
        c.create_text((obj+bkg)/2,base-8,text=f"ZD {ZD:.0f} mm",fill="green")
        c.create_text(obj,top-r_px-8,text="blur",fill="red")

    def show_help(self):
        messagebox.showinfo("Parameters Explanation",
            "FOV      – field‑of‑view width in object plane (mm)\n"
            "Sensor   – camera sensor size along FOV axis (mm)\n"
            "npx      – number of pixels along that axis\n"
            "Object   – target feature size in mm\n"
            "f‑number – lens aperture ratio\n"
            "focal    – lens focal length (mm)\n"
            "ZA       – camera→object distance (mm)\n"
            "ZD       – object→background distance (mm)\n"
            "ZB       – total camera→background span (mm)\n"
            "Square   – checkerboard square size in px\n"
            "Spacing  – checkerboard gap size in px\n\n"
            "Use 'Export Checkerboard' to save a PNG.\n"
            "Use 'Remove blank images tool' to delete every Nth image in a folder."
        )

if __name__=="__main__":
    BOSGui().mainloop()
