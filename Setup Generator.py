#!/usr/bin/env python3
"""
bos_gui.py  –  BOS‑setup helper with fixed 2 px squares & 1 px spacing
--------------------------------------------------------------------
• Tkinter‑only GUI (Python ≥3.8)
• Checkerboard squares: always 2 px side, 1 px gap (in object plane)
• All other behavior unchanged
"""

import math
import tkinter as tk
from tkinter import ttk, messagebox

# ─── core BOS formulas ──────────────────────────────────────────────
def sensitivity(f: float, ZA: float, ZD: float) -> float:
    """S = f·ZD / (ZD + ZA – f)"""
    if ZA <= f or ZD <= 0:
        raise ValueError("Require ZA > focal length and ZD > 0.")
    denom = ZD + ZA - f
    if abs(denom) < 1e-12:
        raise ValueError("Geometry causes zero denominator.")
    return f * ZD / denom

def coc_object(S: float, fnum: float, sensor: float, fov: float) -> float:
    """Circle of confusion (blur) in object plane."""
    return S / (fnum * (1 + sensor / fov))

blur_loss = lambda CoC, obj, k=1/3: math.exp(-k * CoC / obj)

def solve_ZD(S: float, f: float, ZA: float) -> float:
    den = S - f
    if den == 0:
        raise ValueError("Cannot solve ZD (S ≈ focal length).")
    ZD = S * (f - ZA) / den
    if ZD <= 0:
        raise ValueError("Solved ZD ≤ 0.")
    return ZD

def solve_ZA(S: float, f: float, ZD: float) -> float:
    ZA = (f * ZD + S * (f - ZD)) / S
    if ZA <= f:
        raise ValueError("Solved ZA ≤ focal length.")
    return ZA

# ─── main GUI ───────────────────────────────────────────────────────
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

        # StringVars for entries
        keys = ("fov","sensor","npx","obj","fnum","focal","ZA","ZD","ZB")
        self.v = {k: tk.StringVar() for k in keys}
        # defaults
        self.v["fov"].set("160")
        self.v["sensor"].set("25.6")
        self.v["obj"].set("2.4")
        self.v["fnum"].set("16")
        self.v["focal"].set("180")

        self.goal = tk.StringVar(value="signal")
        self._build_gui()

    def _build_gui(self):
        frm = ttk.Frame(self, padding=self.PAD)
        frm.grid()

        rows = [
            ("FOV [mm]"                 , "fov"),
            ("Sensor size [mm]"         , "sensor"),
            ("Sensor pixels (along FOV)", "npx"),
            ("Object size [mm]"         , "obj"),
            ("f‑number"                 , "fnum"),
            ("Focal length f [mm]"      , "focal"),
            ("ZA cam→obj [mm]"          , "ZA"),
            ("ZD obj→bkg [mm]"          , "ZD"),
            ("ZB cam→bkg span [mm]"     , "ZB"),
        ]
        for r,(lbl,key) in enumerate(rows):
            ttk.Label(frm, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(frm, width=10, textvariable=self.v[key]).grid(
                row=r, column=1, padx=(0,self.PAD), pady=1)

        rb = len(rows)
        ttk.Label(frm, text="Goal").grid(row=rb, column=0, sticky="w")
        for c,(txt,val) in enumerate((("Signal","signal"),("Resolution","resolution"))):
            ttk.Radiobutton(frm, text=txt, variable=self.goal, value=val
            ).grid(row=rb, column=1+c)

        ttk.Button(frm, text="Calculate", command=self.calculate).grid(
            row=rb+1, column=0, columnspan=3, pady=(3,self.PAD)
        )

        self.txt = tk.Text(frm, width=64, height=12, bg="#f5f5f5", relief="sunken")
        self.txt.grid(row=0, column=3, rowspan=rb+2, padx=(self.PAD,0))

        self.canvas = tk.Canvas(frm, width=560, height=155, bg="white",
                                relief="solid", bd=1)
        self.canvas.grid(row=rb+2, column=0, columnspan=4, pady=(self.PAD,0))

    def _num(self, key):
        txt = self.v[key].get().strip()
        return None if txt == "" else float(txt)

    def _solve_geometry(self, S, f, ZA, ZD, ZB):
        if ZB and (ZA or ZD):
            raise ValueError("If ZB set, leave ZA & ZD blank.")
        if ZA and ZD:
            if ZB and abs((ZA+ZD)-ZB)>1e-6:
                raise ValueError("ZA+ZD ≠ ZB.")
            return ZA, ZD
        if ZA and not ZD:
            z = solve_ZD(S,f,ZA)
            if ZB and ZA+z>ZB+1e-6:
                raise ValueError("ZA+ZD > ZB.")
            return ZA, z
        if ZD and not ZA:
            a = solve_ZA(S,f,ZD)
            if ZB and a+ZD>ZB+1e-6:
                raise ValueError("ZA+ZD > ZB.")
            return a, ZD
        ZB_val = ZB or max(6*f,1000)
        ZA_val = max(f*1.2, ZB_val/2)
        ZD_val = ZB_val - ZA_val
        S0 = sensitivity(f,ZA_val,ZD_val)
        scale = S/S0
        ZD_val *= scale
        ZA_val = ZB_val - ZD_val
        if ZA_val<=f:
            raise ValueError("Cannot fit ZA>f within ZB.")
        return ZA_val, ZD_val

    def calculate(self):
        try:
            fov    = float(self.v["fov"].get())
            sensor = float(self.v["sensor"].get())
            obj    = float(self.v["obj"].get())
            fnum   = float(self.v["fnum"].get())
            focal  = float(self.v["focal"].get())
        except ValueError:
            messagebox.showerror("Input error","Enter valid numbers")
            return

        npx   = self._num("npx")
        ZA_in = self._num("ZA")
        ZD_in = self._num("ZD")
        ZB_in = self._num("ZB")
        goal  = self.goal.get()

        factor   = 3.0 if goal=="signal" else 1.0
        CoC_t    = factor * obj
        S_target = CoC_t * fnum * (1 + sensor/fov)

        try:
            ZA, ZD = self._solve_geometry(S_target, focal, ZA_in, ZD_in, ZB_in)
            S       = sensitivity(focal, ZA, ZD)
        except ValueError as e:
            messagebox.showerror("Geometry error", str(e))
            return

        CoC     = coc_object(S, fnum, sensor, fov)
        s_min   = CoC/2
        B       = blur_loss(CoC, obj)
        S_eff   = S * B
        ratio   = CoC / obj

        pxmm       = npx/fov if npx else None
        objpx      = obj*pxmm if pxmm else None
        square_mm  = 2/pxmm if pxmm else None   # 2 px square
        spacing_mm = 1/pxmm if pxmm else None   # 1 px gap

        self.txt.delete(1.0,tk.END)
        add = self.txt.insert

        add(tk.END, "──── Geometry ────\n")
        add(tk.END, f"ZA (cam→obj)       : {ZA:.2f} mm\n")
        add(tk.END, f"ZD (obj→bkg)       : {ZD:.2f} mm\n")
        add(tk.END, f"Total span ZB      : {ZA+ZD:.2f} mm\n\n")

        add(tk.END, "──── BOS Metrics ────\n")
        add(tk.END, f"S   sensitivity     : {S:.3f} mm/px deflection\n")
        add(tk.END, f"CoC blur           : {CoC:.3f} mm ({ratio:.2f}×object)\n")
        add(tk.END, f"s_min ≈ CoC/2       : {s_min:.3f} mm\n")
        add(tk.END, f"B   blur‑loss       : {B:.3f}\n")
        add(tk.END, f"S_eff = S·B        : {S_eff:.3f} mm\n\n")

        if pxmm:
            add(tk.END, "──── Sampling ────\n")
            add(tk.END, f"Pixels/mm           : {pxmm:.2f} px/mm\n")
            add(tk.END, f"Object width        : {objpx:.1f} px\n")
            add(tk.END, f"Checker square size : {square_mm:.3f} mm (2 px)\n")
            add(tk.END, f"Checker spacing     : {spacing_mm:.3f} mm (1 px)\n\n")

        add(tk.END, "──── Advice ────\n")
        if goal=="signal":
            if 2.5<=ratio<=3.5:
                add(tk.END, "✓ Blur≈3×object → optimal for signal\n")
            elif ratio<2.5:
                add(tk.END, "• Blur too small → increase ZD or focal length\n")
            else:
                add(tk.END, "• Blur too large → decrease ZD or open aperture\n")
        else:
            if ratio<=1.2:
                add(tk.END, "✓ Blur ≤ object → good resolution\n")
            else:
                add(tk.END, "• Blur > object → reduce ZD or close aperture\n")
        if pxmm and pxmm<5:
            add(tk.END, "• Low sampling (<5 px/mm) → more pixels or smaller FOV\n")

        self._draw_schematic(ZA, ZD, CoC)

    def _draw_schematic(self, ZA, ZD, CoC):
        c = self.canvas; c.delete("all")
        W,H = 560,155; top,base=40,H-30; total=ZA+ZD
        X = lambda d: 60 + d/total*(W-120)
        cam_x,obj_x,bkg_x = X(0),X(ZA),X(total)
        for x,col in ((cam_x,"black"),(obj_x,"blue"),(bkg_x,"green")):
            c.create_line(x,top,x,base,width=3,fill=col)
        arrow = dict(arrow=tk.LAST,width=2)
        c.create_line(cam_x,base,obj_x,base,fill="blue",**arrow)
        c.create_line(obj_x,base-20,bkg_x,base-20,fill="green",**arrow)
        c.create_text((cam_x+obj_x)/2,base+12,text=f"ZA {ZA:.0f} mm",fill="blue")
        c.create_text((obj_x+bkg_x)/2,base-8,text=f"ZD {ZD:.0f} mm",fill="green")
        pxmm = (W-120)/total; r_px = max(4, min(CoC*pxmm/2,45))
        c.create_oval(obj_x-r_px,top-r_px,obj_x+r_px,top+r_px,outline="red",width=2)
        c.create_text(obj_x,top-r_px-8,text="blur",fill="red")

    def show_help(self):
        text = (
            "INPUT PARAMETERS:\n"
            " FOV    : field‑of‑view width in object plane (mm)\n"
            " Sensor : camera sensor size along FOV axis (mm)\n"
            " npx    : number of camera pixels along that axis\n"
            " Object : characteristic flow feature size (mm)\n"
            " f#     : lens f‑number (focal / aperture)\n"
            " f      : lens focal length (mm)\n"
            " ZA     : camera→object distance (mm)\n"
            " ZD     : object→background distance (mm)\n"
            " ZB     : camera→background total span (mm)\n\n"
            "GOAL:\n"
            " 'Signal'     ⇒ maximize BOS signal (ideal blur ≈3×object)\n"
            " 'Resolution' ⇒ maximize resolution (blur ≤object)\n\n"
            "OUTPUT METRICS:\n"
            " S            : geometric sensitivity (mm shift / px deflection)\n"
            " CoC          : blur diameter in object plane (mm)\n"
            " s_min        : ≈CoC/2 minimum resolvable feature size (mm)\n"
            " B            : blur‑loss factor (1=no loss)\n"
            " S_eff        : effective sensitivity = S×B (mm)\n"
            " px/mm        : digital sampling (px per mm)\n"
            " obj_px       : object width in pixels\n"
            " square_mm    : checkerboard square side (mm) = 2 px\n"
            " spacing_mm   : gap between squares (mm) = 1 px\n"
        )
        w = tk.Toplevel(self)
        w.title("Parameter Explanations")
        txt = tk.Text(w, width=72, height=28, wrap="word")
        txt.pack(expand=True, fill="both", padx=10, pady=10)
        txt.insert("1.0", text)
        txt.config(state="disabled")


if __name__ == "__main__":
    BOSGui().mainloop()
