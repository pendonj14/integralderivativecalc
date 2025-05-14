import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.integrate import quad
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, messagebox

# ─── SuperscriptEntry ─────────────────────────────────────────────────────────
class SuperscriptEntry(ttk.Entry):
    """
    An Entry that converts '^'+next char into Unicode superscript,
    for digits and lowercase letters.
    """
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._sup_next = False
        self._sup_map = {
            # digits
            '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴',
            '5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹',
            # letters
            'a':'ᵃ','b':'ᵇ','c':'ᶜ','d':'ᵈ','e':'ᵉ','f':'ᶠ',
            'g':'ᵍ','h':'ʰ','i':'ⁱ','j':'ʲ','k':'ᵏ','l':'ˡ',
            'm':'ᵐ','n':'ⁿ','o':'ᵒ','p':'ᵖ','r':'ʳ','s':'ˢ',
            't':'ᵗ','u':'ᵘ','v':'ᵛ','w':'ʷ','x':'ˣ','y':'ʸ','z':'ᶻ',
            # symbols
            '+':'⁺','-':'⁻','=':'⁼','(':'⁽',')':'⁾'
        }
        self.bind("<Key>", self._on_key)

    def _on_key(self, ev):
        # If user typed '^', catch it and set flag
        if ev.char == '^':
            self._sup_next = True
            return "break"
        # If flag set, insert unicode superscript
        if self._sup_next and ev.char:
            sup = self._sup_map.get(ev.char, ev.char)
            idx = self.index(tk.INSERT)
            self.insert(idx, sup)
            self._sup_next = False
            return "break"
        # Otherwise normal processing
        return None

# ─── Main Visualizer ───────────────────────────────────────────────────────────
class FunctionVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("KimUlator")
        self.root.geometry("750x750")
        self.root.configure(bg="#ffffff")

        self.x = sp.symbols('x')

        # ─── Styles ────────────────────────────────────────────────────────────
        self.style = ttk.Style()
        self.style.theme_use("clam")
        for w in ("TFrame", "TLabelframe", "TLabel", "TEntry"):
            self.style.configure(w, background="#ffffff", foreground="#000000")

        # ─── Function Visualizer Panel ─────────────────────────────────────────
        ttk.Label(self.root, text="Function Visualizer", font=("Arial", 12, "bold")).pack()
        viz_panel = ttk.Frame(self.root)
        viz_panel.pack(pady=10)

        # Row 1: function entry
        row1 = ttk.Frame(viz_panel); row1.pack(pady=5)
        ttk.Label(row1, text="f(x) =").pack(side=tk.LEFT, padx=5)
        self.function_entry = SuperscriptEntry(row1, width=30)
        self.function_entry.pack(side=tk.LEFT, padx=5)
        self.function_entry.insert(0, "x²")

        # Row 2: x-range
        row2 = ttk.Frame(viz_panel); row2.pack(pady=5)
        ttk.Label(row2, text="x min:").pack(side=tk.LEFT, padx=5)
        self.x_min_entry = ttk.Entry(row2, width=8)
        self.x_min_entry.pack(side=tk.LEFT, padx=5)
        self.x_min_entry.insert(0, "-5")
        ttk.Label(row2, text="x max:").pack(side=tk.LEFT, padx=5)
        self.x_max_entry = ttk.Entry(row2, width=8)
        self.x_max_entry.pack(side=tk.LEFT, padx=5)
        self.x_max_entry.insert(0, "5")

        # Row 3: display options
        row3 = ttk.Frame(viz_panel); row3.pack(pady=5)
        self.show_original = tk.BooleanVar(value=True)
        self.show_derivative = tk.BooleanVar(value=True)
        self.show_integral = tk.BooleanVar(value=True)
        for txt, var, color in [
            ("Original", self.show_original, "#ffffff"),     # light blue
            ("Derivative", self.show_derivative, "#ffffff"), # light green
            ("Integral", self.show_integral, "#ffffff")      # light yellow
        ]:
            tk.Checkbutton(row3, text=txt, variable=var, bg=color,
                           activebackground=color, selectcolor=color).pack(side=tk.LEFT, padx=10)

        # Row 4: buttons
        row4 = ttk.Frame(viz_panel); row4.pack(pady=10)
        for txt, cmd in [
            ("Visualize", self.visualize),
            ("Clear", self.clear),
            ("Save Graph", self.save_graph)
        ]:
            tk.Button(row4, text=txt, command=cmd,
                      bg="#dddddd", fg="black",
                      highlightbackground="black", highlightthickness=1).pack(side=tk.LEFT, padx=5)

        # ─── Numerical & Symbolic Results ────────────────────────────────────
        ttk.Label(self.root, text="Numerical & Symbolic Results", font=("Arial", 12, "bold")).pack()
        res_panel = ttk.Frame(self.root); res_panel.pack(pady=10)

        # top row: point + calculate
        top = ttk.Frame(res_panel); top.pack(pady=(0, 10))
        ttk.Label(top, text="Point x =").pack(side=tk.LEFT, padx=5)
        self.point_entry = ttk.Entry(top, width=8)
        self.point_entry.pack(side=tk.LEFT, padx=5)
        self.point_entry.insert(0, "0")
        tk.Button(top, text="Calculate", command=self.calculate_results,
                  bg="#dddddd", fg="black",
                  ).pack(side=tk.LEFT, padx=5)

        # bottom: two columns
        bottom = ttk.Frame(res_panel); bottom.pack()
        num_col = ttk.Frame(bottom); num_col.pack(side=tk.LEFT, padx=50)

        def new_lbl(parent, text):
            fr = ttk.Frame(parent); fr.pack(pady=10)
            ttk.Label(fr, text=text).pack(side=tk.LEFT, padx=5)
            lbl = ttk.Label(fr, text="----", width=10); lbl.pack(side=tk.LEFT)
            return lbl

        self.func_value_label     = new_lbl(num_col, "f(x) =")
        self.deriv_value_label    = new_lbl(num_col, "f'(x) =")
        self.integral_value_label = new_lbl(num_col, "∫f(x)dx =")

        sym_col = ttk.Frame(bottom); sym_col.pack(side=tk.LEFT, padx=50)
        self.sym_deriv_label = new_lbl(sym_col, "f'(x) sym =")
        self.sym_int_label   = new_lbl(sym_col, "∫f(x)dx sym =")

        # ─── Plot Canvas ─────────────────────────────────────────────────────────
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # placeholders
        self.original_func = None
        self.expr = None
        self.x_vals = None


    # ─── Parsing ──────────────────────────────────────────────────────────────────
    def parse_function(self, func_str):
        x = self.x

        # 0) convert unicode superscripts → ** syntax
        sup_back = {
            '⁰':'0','¹':'1','²':'2','³':'3','⁴':'4',
            '⁵':'5','⁶':'6','⁷':'7','⁸':'8','⁹':'9',
            'ᵃ':'a','ᵇ':'b','ᶜ':'c','ᵈ':'d','ᵉ':'e','ᶠ':'f',
            'ᵍ':'g','ʰ':'h','ⁱ':'i','ʲ':'j','ᵏ':'k','ˡ':'l',
            'ᵐ':'m','ⁿ':'n','ᵒ':'o','ᵖ':'p','ʳ':'r','ˢ':'s',
            'ᵗ':'t','ᵘ':'u','ᵛ':'v','ʷ':'w','ˣ':'x','ʸ':'y','ᶻ':'z'
        }
        for sup,normal in sup_back.items():
            func_str = func_str.replace(sup, f"**{normal}")

        # 1) caret → **
        func_str = func_str.replace("^", "**")

        # 2) ln → log
        func_str = re.sub(r'\bln\b','log', func_str)

        # 3a) implicit * between number→letter or (
        func_str = re.sub(r'(\d)([a-zA-Z\(])', r'\1*\2', func_str)
        # 3b) implicit * for x or ) → (
        func_str = re.sub(r'([x\)])\(', r'\1*(', func_str)
        # 3c) implicit * for ) → x
        func_str = re.sub(r'\)(x)', r')*\1', func_str)

        allowed = {
            'x': x, 'e':sp.E, 'pi':sp.pi,
            'sin':sp.sin,'cos':sp.cos,'tan':sp.tan,'cot':sp.cot,
            'sec':sp.sec,'csc':sp.csc,
            'log':sp.log,'sqrt':sp.sqrt,'exp':sp.exp,'abs':sp.Abs
        }

        try:
            expr = sp.sympify(func_str, locals=allowed)
            func = sp.lambdify(x, expr, modules=['numpy'])
            return expr, func
        except Exception as e:
            messagebox.showerror("Error", f"Invalid function: {e}")
            return None, None

    # ─── Plotting ─────────────────────────────────────────────────────────────────
    def visualize(self):
        func_str = self.function_entry.get()
        try:
            x_min = float(self.x_min_entry.get())
            x_max = float(self.x_max_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid x-range"); return
        if x_min >= x_max:
            messagebox.showerror("Error", "x_min must be less than x_max"); return

        expr, func = self.parse_function(func_str)
        if func is None: return

        self.expr = expr
        self.original_func = func
        self.x_vals = np.linspace(x_min, x_max, 1000)
        self.ax.clear()

        if self.show_original.get():
            try:
                self.ax.plot(self.x_vals, func(self.x_vals), label="f(x)")
            except Exception as e:
                messagebox.showwarning("Warning", f"Plot f(x) error: {e}")

        if self.show_derivative.get():
            def deriv_num(xi):
                h=1e-6
                return (func(xi+h)-func(xi-h))/(2*h)
            try:
                y = np.array([deriv_num(xi) for xi in self.x_vals])
                self.ax.plot(self.x_vals, y, label="f'(x)")
            except Exception as e:
                messagebox.showwarning("Warning", f"Plot f'(x) error: {e}")

        if self.show_integral.get():
            try:
                y = np.array([quad(func, x_min, xi)[0] for xi in self.x_vals])
                self.ax.plot(self.x_vals, y, label="∫f(x)dx")
            except Exception as e:
                messagebox.showwarning("Warning", f"Plot ∫f error: {e}")

        self.ax.set_title(f"f(x) = {func_str}")
        self.ax.grid(True,linestyle="--",alpha=0.5)
        self.ax.legend()
        self.canvas.draw()

    # ─── Numeric + Symbolic Results ───────────────────────────────────────────────
    def calculate_results(self):
        if not self.original_func or not self.expr:
            messagebox.showwarning("Warning", "Visualize a function first"); return

        try:
            xi = float(self.point_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid point"); return

        # numeric f(x)
        try:
            self.func_value_label.config(text=f"{self.original_func(xi):.3f}")
        except:
            self.func_value_label.config(text="Error")

        # numeric f'(x)
        try:
            h=1e-6
            dv=(self.original_func(xi+h)-self.original_func(xi-h))/(2*h)
            self.deriv_value_label.config(text=f"{dv:.3f}")
        except:
            self.deriv_value_label.config(text="Error")

        # numeric ∫
        try:
            xm=float(self.x_min_entry.get())
            iv=quad(self.original_func,xm,xi)[0]
            self.integral_value_label.config(text=f"{iv:.3f}")
        except:
            self.integral_value_label.config(text="Error")

        # prepare superscript converter
        uni_sup = {
            '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴',
            '5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹',
            'a':'ᵃ','b':'ᵇ','c':'ᶜ','d':'ᵈ','e':'ᵉ','f':'ᶠ',
            'g':'ᵍ','h':'ʰ','i':'ⁱ','j':'ʲ','k':'ᵏ','l':'ˡ',
            'm':'ᵐ','n':'ⁿ','o':'ᵒ','p':'ᵖ','r':'ʳ','s':'ˢ',
            't':'ᵗ','u':'ᵘ','v':'ᵛ','w':'ʷ','x':'ˣ','y':'ʸ','z':'ᶻ',
            '+':'⁺','-':'⁻','(':'⁽',')':'⁾'
        }
        def superscriptify(s):
            # exp(...) → e^...
            s = re.sub(r'exp\(([^)]+)\)', r'e^\1', s)
            # ** → ^ for digits+letters
            s = s.replace('**','^')
            # drop *
            s = s.replace('*','')
            # now convert ^something into real superscript
            def rep(m):
                return ''.join(uni_sup.get(ch,ch) for ch in m.group(1))
            return re.sub(r'\^([0-9A-Za-z\+\-\=\(\)]+)', rep, s)

        # symbolic derivative
        try:
            sd = sp.diff(self.expr,self.x)
            self.sym_deriv_label.config(text=superscriptify(str(sd)))
        except:
            self.sym_deriv_label.config(text="Error")

        # symbolic integral
        try:
            si = sp.integrate(self.expr,self.x)
            self.sym_int_label.config(text=superscriptify(str(si)))
        except:
            self.sym_int_label.config(text="Error")

    # ─── Helpers ───────────────────────────────────────────────────────────────────
    def clear(self):
        self.function_entry.delete(0,tk.END); self.function_entry.insert(0,"x²")
        self.x_min_entry.delete(0,tk.END);    self.x_min_entry.insert(0,"-5")
        self.x_max_entry.delete(0,tk.END);    self.x_max_entry.insert(0,"5")
        self.point_entry.delete(0,tk.END);    self.point_entry.insert(0,"0")
        for lbl in (self.func_value_label,self.deriv_value_label,
                    self.integral_value_label,self.sym_deriv_label,
                    self.sym_int_label):
            lbl.config(text="----")
        self.ax.clear(); self.ax.grid(True,linestyle="--",alpha=0.5)
        self.canvas.draw()

    def save_graph(self):
        try:
            self.fig.savefig("function_visualization.png",dpi=300,bbox_inches="tight")
            messagebox.showinfo("Saved","Graph saved as function_visualization.png")
        except Exception as e:
            messagebox.showerror("Error",f"Save failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FunctionVisualizer(root)
    root.mainloop()
