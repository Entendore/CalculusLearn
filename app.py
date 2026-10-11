import sys
import numpy as np
import sympy as sp
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QGroupBox, 
    QSizePolicy, QDoubleSpinBox, QComboBox, QGridLayout, QScrollArea, QStatusBar, 
    QCheckBox, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

DARK_STYLESHEET = """
    QMainWindow { background-color: #1e1e1e; }
    QWidget { color: #e0e0e0; background-color: #1e1e1e; }
    QScrollArea { border: none; background-color: transparent; }
    QScrollArea > QWidget > QWidget { background-color: transparent; }
    QTabWidget::pane { border: 1px solid #3a3a3a; background: #252526; }
    QTabBar::tab { background: #2d2d2d; color: #a0a0a0; padding: 10px 20px; border: 1px solid #3a3a3a; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
    QTabBar::tab:selected { background: #007acc; color: white; }
    QGroupBox { border: 1px solid #3a3a3a; border-radius: 6px; margin-top: 12px; padding-top: 14px; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #007acc; }
    QPushButton { background-color: #007acc; color: white; border: none; padding: 6px; border-radius: 4px; }
    QPushButton:hover { background-color: #0098ff; }
    QPushButton:pressed { background-color: #005a9e; }
    QPushButton:checked { background-color: #28a745; }
    QLineEdit, QDoubleSpinBox, QComboBox { background-color: #2d2d2d; border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px; min-height: 20px; }
    QSlider::groove:horizontal { border: 1px solid #3a3a3a; height: 4px; background: #2d2d2d; border-radius: 2px; }
    QSlider::handle:horizontal { background: #007acc; width: 14px; margin: -6px 0; border-radius: 7px; }
    QLabel { background-color: transparent; }
    QStatusBar { background-color: #007acc; color: white; font-weight: bold; }
    QToolBar { background-color: #2d2d2d; border: none; }
    QToolBar QToolButton { background-color: transparent; color: #e0e0e0; }
"""

class LabeledSlider(QWidget):
    def __init__(self, label_text, min_val, max_val, init_val, step=0.1):
        super().__init__()
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        top_layout = QHBoxLayout()
        self.label = QLabel(label_text)
        self.spin = QDoubleSpinBox(); self.spin.setRange(min_val, max_val)
        self.spin.setSingleStep(step); self.spin.setDecimals(2); self.spin.setValue(init_val)
        top_layout.addWidget(self.label); top_layout.addStretch(); top_layout.addWidget(self.spin)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * 100), int(max_val * 100)); self.slider.setValue(int(init_val * 100))
        self.spin.valueChanged.connect(self.sync_slider_from_spin)
        self.slider.valueChanged.connect(self.sync_spin_from_slider)
        layout.addLayout(top_layout); layout.addWidget(self.slider)
        
    def sync_slider_from_spin(self, val):
        self.slider.blockSignals(True); self.slider.setValue(int(val * 100)); self.slider.blockSignals(False)
        
    def sync_spin_from_slider(self, val):
        self.spin.blockSignals(True); self.spin.setValue(val / 100.0); self.spin.blockSignals(False)

class GraphCanvas(QWidget):
    def __init__(self, status_bar):
        super().__init__()
        self.status_bar = status_bar
        self.figure = Figure(figsize=(7, 6), facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.nav_toolbar = NavToolbar(self.canvas, self)
        
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.canvas); layout.addWidget(self.nav_toolbar)
        
        self.current_x_vals, self.current_y_vals = None, None
        self.hover_vline, self.hover_point = None, None
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
    def on_mouse_move(self, event):
        if event.inaxes and self.current_x_vals is not None:
            idx = np.argmin(np.abs(self.current_x_vals - event.xdata))
            x_exact, y_exact = self.current_x_vals[idx], self.current_y_vals[idx]
            if not np.isnan(y_exact):
                if self.hover_vline is None:
                    self.hover_vline = event.inaxes.axvline(x_exact, color='white', linestyle=':', alpha=0.5)
                    self.hover_point, = event.inaxes.plot([x_exact], [y_exact], 'o', color='white', markersize=6, zorder=10)
                else:
                    self.hover_vline.set_xdata(x_exact); self.hover_point.set_data([x_exact], [y_exact])
                self.canvas.draw_idle(); self.status_bar.showMessage(f"Trace: x={x_exact:.3f}, y={y_exact:.3f}", 2000); return
        self.status_bar.showMessage("Ready", 2000)

    def set_trace_data(self, x, y):
        self.current_x_vals = x; self.current_y_vals = y
        self.hover_vline = None; self.hover_point = None

class BaseModule(QWidget):
    def __init__(self, canvas, status_bar):
        super().__init__()
        self.canvas = canvas
        self.status_bar = status_bar
        self.x, self.y, self.t, self.n_sym = sp.symbols('x y t n')
        
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setFixedWidth(380)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget); self.controls_layout.setSpacing(8)
        self.scroll.setWidget(self.controls_widget)
        
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.main_layout.addWidget(self.scroll)
        self.main_layout.addWidget(self.canvas, 1)
        
        self.setup_ui()
        
    def insert_symbol(self, line_edit, symbol):
        cp = line_edit.cursorPosition(); txt = line_edit.text()
        line_edit.setText(txt[:cp] + symbol + txt[cp:]); line_edit.setCursorPosition(cp + len(symbol))
        line_edit.setFocus()

    def create_function_input(self, default="sin(x)*x", presets=None, var1='x'):
        group = QGroupBox("1. Define Function"); layout = QVBoxLayout()
        input_layout = QHBoxLayout(); input_layout.addWidget(QLabel(f"f({var1}) ="))
        self.func_input = QComboBox(); self.func_input.setEditable(True); self.func_input.setInsertPolicy(QComboBox.InsertAtTop)
        self.func_input.addItem(default); self.func_input.setCurrentText(default)
        self.func_input.lineEdit().returnPressed.connect(self.update_plot)
        input_layout.addWidget(self.func_input); layout.addLayout(input_layout)
        
        kp = QGridLayout(); kp.setSpacing(2)
        symbols = ["x**2", "x**3", "sqrt(x)", "pi", "E", "sin(x)", "cos(x)", "tan(x)", "exp(x)", "log(x)", "abs(x)", "1/x"]
        for i, sym in enumerate(symbols):
            btn = QPushButton(sym); btn.setFixedHeight(24)
            btn.clicked.connect(lambda c, s=sym: self.insert_symbol(self.func_input.lineEdit(), s))
            kp.addWidget(btn, i // 4, i % 4)
        layout.addLayout(kp)
        
        if presets:
            pp = QGridLayout()
            for i, p in enumerate(presets):
                btn = QPushButton(p); btn.setFixedHeight(24)
                btn.clicked.connect(lambda c, x=p: (self.func_input.setCurrentText(x), self.update_plot()))
                pp.addWidget(btn, i // 4, i % 4)
            layout.addLayout(pp)
            
        btn = QPushButton("Update Graph"); btn.clicked.connect(self.update_plot)
        layout.addWidget(btn); group.setLayout(layout)
        self.controls_layout.addWidget(group)

    def add_to_history(self, text):
        if self.func_input.findText(text) == -1: self.func_input.insertItem(0, text)
        self.func_input.setCurrentText(text)

    def create_view_controls(self, is_3d=False):
        group = QGroupBox("Graph View Settings"); layout = QGridLayout()
        layout.addWidget(QLabel("X-Min:"), 0, 0)
        self.x_min_spin = QDoubleSpinBox(); self.x_min_spin.setRange(-100, 100); self.x_min_spin.setValue(-10)
        self.x_min_spin.valueChanged.connect(self.update_plot); layout.addWidget(self.x_min_spin, 0, 1)
        layout.addWidget(QLabel("X-Max:"), 0, 2)
        self.x_max_spin = QDoubleSpinBox(); self.x_max_spin.setRange(-100, 100); self.x_max_spin.setValue(10)
        self.x_max_spin.valueChanged.connect(self.update_plot); layout.addWidget(self.x_max_spin, 0, 3)
        if not is_3d:
            layout.addWidget(QLabel("Y-Min:"), 1, 0)
            self.y_min_spin = QDoubleSpinBox(); self.y_min_spin.setRange(-100, 100); self.y_min_spin.setValue(-15)
            self.y_min_spin.valueChanged.connect(self.update_plot); layout.addWidget(self.y_min_spin, 1, 1)
            layout.addWidget(QLabel("Y-Max:"), 1, 2)
            self.y_max_spin = QDoubleSpinBox(); self.y_max_spin.setRange(-100, 100); self.y_max_spin.setValue(15)
            self.y_max_spin.valueChanged.connect(self.update_plot); layout.addWidget(self.y_max_spin, 1, 3)
        self.btn_reset_view = QPushButton("Reset View"); self.btn_reset_view.clicked.connect(self.reset_view)
        layout.addWidget(self.btn_reset_view, 2, 0, 1, 4); group.setLayout(layout)
        self.controls_layout.addWidget(group); self.controls_layout.addStretch()

    def reset_view(self):
        self.x_min_spin.setValue(-10); self.x_max_spin.setValue(10)
        if hasattr(self, 'y_min_spin'): self.y_min_spin.setValue(-15); self.y_max_spin.setValue(15)
        self.update_plot()

    def setup_canvas_axes(self, ax):
        ax.clear(); ax.set_facecolor("#252526")
        ax.axhline(0, color='#555555', linewidth=1); ax.axvline(0, color='#555555', linewidth=1)
        ax.grid(True, linestyle='--', alpha=0.3, color='#555555')
        ax.tick_params(axis='x', colors='#cccccc'); ax.tick_params(axis='y', colors='#cccccc')
        for spine in ax.spines.values(): spine.set_color('#555555')
        ax.set_xlim(self.x_min_spin.value(), self.x_max_spin.value())
        if hasattr(self, 'y_min_spin'): ax.set_ylim(self.y_min_spin.value(), self.y_max_spin.value())

    def safe_evaluate(self, expr, x_vals):
        f_lambda = sp.lambdify(self.x, expr, "numpy"); y_vals = f_lambda(x_vals)
        if isinstance(y_vals, (int, float)): y_vals = np.full_like(x_vals, y_vals, dtype=float)
        y_vals = np.where(np.isreal(y_vals), y_vals, np.nan); y_vals = np.where(np.abs(y_vals) > 1e5, np.nan, y_vals)
        return y_vals, f_lambda

# --- CALCULUS I MODULES ---

class LimitModule(BaseModule):
    def setup_ui(self):
        self.create_function_input(default="1/x", presets=["1/x", "sin(x)/x", "x**2-1", "1/(x-2)"])
        pg = QGroupBox("2. Point Approaching (a)"); pl = QVBoxLayout()
        self.a_slider = LabeledSlider("a =", -10.0, 10.0, 1.0)
        self.a_slider.spin.valueChanged.connect(self.update_plot); pl.addWidget(self.a_slider); pg.setLayout(pl)
        self.controls_layout.addWidget(pg)
        rg = QGroupBox("3. Limit Analysis"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        av = self.a_slider.value()
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 400)
        yv, _ = self.safe_evaluate(expr, xv); self.canvas.set_trace_data(xv, yv)
        ax.plot(xv, yv, '#4ec9b0', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')
        try:
            ll, lr, lt = sp.limit(expr, self.x, av, '-'), sp.limit(expr, self.x, av, '+'), sp.limit(expr, self.x, av)
            fa = expr.subs(self.x, av)
            if ll == sp.zoo or lr == sp.zoo: ax.set_ylim(-50, 50)
            ltxt, rtxt = "Inf", "Inf"
            if ll != sp.zoo and ll != -sp.zoo: lv = float(ll); ax.plot(av - 0.2, lv, '<', color='#f44747', markersize=10, zorder=5); ltxt = f"{lv:.2f}"
            if lr != sp.zoo and lr != -sp.zoo: rv = float(lr); ax.plot(av + 0.2, rv, '>', color='#569cd6', markersize=10, zorder=5); rtxt = f"{rv:.2f}"
            self.result_label.setText(f"f(x)={expr}\n\nx -> {av:.2f}\n\nLeft: {ltxt}\nRight: {rtxt}\nTwo-Sided: {lt}\n\nf({av:.2f})={fa}\nContinuous? {'YES' if ll==lr==fa else 'NO'}")
            ax.axvline(x=av, color='#dcdcaa', linestyle='--', label=f'x = {av:.2f}')
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Limits & Continuity", color='white'); self.canvas.canvas.draw()

class DerivativeModule(BaseModule):
    def setup_ui(self):
        self.create_function_input()
        pg = QGroupBox("2. Evaluation Point (x)"); pl = QVBoxLayout()
        self.point_slider = LabeledSlider("x =", -10.0, 10.0, 0.0)
        self.point_slider.spin.valueChanged.connect(self.update_plot); pl.addWidget(self.point_slider); pg.setLayout(pl); self.controls_layout.addWidget(pg)
        hg = QGroupBox("3. Secant Line Step (h)"); hl = QVBoxLayout()
        self.h_slider = LabeledSlider("h =", 0.01, 5.0, 1.0)
        self.h_slider.spin.valueChanged.connect(self.update_plot); hl.addWidget(self.h_slider)
        self.btn_animate = QPushButton("▶ Play h -> 0 Animation"); self.btn_animate.setCheckable(True)
        self.btn_animate.clicked.connect(self.toggle_animation); hl.addWidget(self.btn_animate); hg.setLayout(hl); self.controls_layout.addWidget(hg)
        rg = QGroupBox("4. Math Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls()
        self.timer = QTimer(self); self.timer.timeout.connect(self.animate_step); self.update_plot()

    def toggle_animation(self):
        if self.btn_animate.isChecked(): self.btn_animate.setText("⏸ Stop"); self.h_slider.spin.setValue(5.0); self.timer.start(100)
        else: self.btn_animate.setText("▶ Play h -> 0 Animation"); self.timer.stop()

    def animate_step(self):
        ch = self.h_slider.value()
        if ch > 0.05: self.h_slider.spin.setValue(ch - 0.1)
        else: self.timer.stop(); self.btn_animate.setChecked(False); self.btn_animate.setText("▶ Play h -> 0 Animation")

    def update_plot(self):
        xp, hv = self.point_slider.value(), self.h_slider.value()
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 400)
        yv, _ = self.safe_evaluate(expr, xv); self.canvas.set_trace_data(xv, yv)
        try:
            d2 = sp.diff(expr, self.x, 2); d2v = sp.lambdify(self.x, d2, "numpy")(xv)
            if isinstance(d2v, (int, float)): d2v = np.full_like(xv, d2v, dtype=float)
            ax.fill_between(xv, self.y_min_spin.value(), self.y_max_spin.value(), where=(d2v > 0), color='#28a745', alpha=0.05)
            ax.fill_between(xv, self.y_min_spin.value(), self.y_max_spin.value(), where=(d2v < 0), color='#dc3545', alpha=0.05)
        except: pass
        ax.plot(xv, yv, '#4ec9b0', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')
        try:
            yp = float(expr.subs(self.x, xp)); d1 = sp.diff(expr, self.x); s = float(d1.subs(self.x, xp))
            ax.plot(xv, yp + s * (xv - xp), '#f44747', '--', linewidth=1.5, label=f"Tangent (m={s:.2f})")
            yhp = float(expr.subs(self.x, xp + hv)); ss = (yhp - yp) / hv
            ax.plot(xv, yp + ss * (xv - xp), '#dcdcaa', ':', linewidth=1.5, label=f"Secant (m={ss:.2f})")
            ax.plot(xp, yp, 'o', color='#f44747', markersize=8, zorder=5); ax.plot(xp + hv, yhp, 's', color='#dcdcaa', markersize=6, zorder=5)
            self.result_label.setText(f"f(x)={expr}\nf'(x)={d1}\nf''(x)={d2}\n\nPoint x={xp:.2f}\nf({xp:.2f})={yp:.2f}\nf'({xp:.2f})={s:.2f}\n\nSecant h={hv:.2f}\nSecant Slope={ss:.2f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Derivatives & Concavity", color='white'); self.canvas.canvas.draw()

class AnalysisModule(BaseModule):
    def setup_ui(self):
        self.create_function_input(default="x**3 - 3*x**2")
        rg = QGroupBox("2. Curve Analysis"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def find_roots(self, x, y):
        r = []
        for i in range(len(x)-1):
            if np.isnan(y[i]) or np.isnan(y[i+1]): continue
            if y[i] * y[i+1] < 0: r.append((x[i] + x[i+1])/2)
            elif abs(y[i]) < 1e-5 and (i == 0 or abs(y[i-1]) > 1e-5): r.append(x[i])
        return r

    def update_plot(self):
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 1000)
        yv, _ = self.safe_evaluate(expr, xv); self.canvas.set_trace_data(xv, yv)
        ax.plot(xv, yv, '#4ec9b0', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')
        try:
            d1, d2 = sp.diff(expr, self.x), sp.diff(expr, self.x, 2)
            dy = sp.lambdify(self.x, d1, "numpy")(xv); d2y = sp.lambdify(self.x, d2, "numpy")(xv)
            if isinstance(dy, (int, float)): dy = np.full_like(xv, dy, dtype=float)
            if isinstance(d2y, (int, float)): d2y = np.full_like(xv, d2y, dtype=float)
            r, c, i = self.find_roots(xv, yv), self.find_roots(xv, dy), self.find_roots(xv, d2y)
            for x in r: ax.plot(x, 0, 'o', color='#dcdcaa', markersize=8, zorder=5)
            for x in c:
                yc = float(expr.subs(self.x, x))
                if not np.isnan(yc):
                    idx = np.argmin(np.abs(xv - x))
                    ax.plot(x, yc, '^' if d2y[idx] < 0 else 'v', color='#f44747' if d2y[idx] < 0 else '#569cd6', markersize=10, zorder=5)
            for x in i:
                yi = float(expr.subs(self.x, x))
                if not np.isnan(yi): ax.plot(x, yi, 's', color='#9b59b6', markersize=8, zorder=5)
            self.result_label.setText(f"f(x)={expr}\nf'={d1}\nf''={d2}\n\nRoots: {[f'{v:.2f}' for v in r]}\nExtrema: {[f'{v:.2f}' for v in c]}\nInflection: {[f'{v:.2f}' for v in i]}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Curve Analysis", color='white'); self.canvas.canvas.draw()

class NewtonModule(BaseModule):
    def setup_ui(self):
        self.create_function_input(default="x**2 - 2")
        ig = QGroupBox("2. Initial Guess (x0)"); il = QVBoxLayout()
        self.x0_slider = LabeledSlider("x0 =", -10.0, 10.0, 2.0)
        self.x0_slider.spin.valueChanged.connect(self.update_plot); il.addWidget(self.x0_slider); ig.setLayout(il); self.controls_layout.addWidget(ig)
        ng = QGroupBox("3. Iterations (N)"); nl = QVBoxLayout()
        self.n_slider = LabeledSlider("N =", 1, 5, 3, step=1); self.n_slider.spin.setDecimals(0)
        self.n_slider.spin.valueChanged.connect(self.update_plot); nl.addWidget(self.n_slider); ng.setLayout(nl); self.controls_layout.addWidget(ng)
        rg = QGroupBox("4. Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        x0, n_val = self.x0_slider.value(), int(self.n_slider.value())
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 400)
        yv, _ = self.safe_evaluate(expr, xv); self.canvas.set_trace_data(xv, yv)
        ax.plot(xv, yv, '#4ec9b0', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')
        try:
            d1 = sp.diff(expr, self.x); vals = [x0]
            for _ in range(n_val):
                fx, dfx = float(expr.subs(self.x, vals[-1])), float(d1.subs(self.x, vals[-1]))
                if abs(dfx) < 1e-6: break
                vals.append(vals[-1] - fx/dfx)
            for i in range(len(vals)-1):
                xn, xn1 = vals[i], vals[i+1]; yn = float(expr.subs(self.x, xn))
                slope = float(d1.subs(self.x, xn))
                tx = np.linspace(xn - 2, xn + 2, 10); ty = yn + slope * (tx - xn)
                ax.plot(tx, ty, '#f44747', '--', linewidth=1)
                ax.plot([xn, xn], [yn, 0], color='#dcdcaa', linestyle=':', linewidth=1)
                ax.plot([xn, xn1], [0, float(expr.subs(self.x, xn1))], color='#dcdcaa', linestyle=':', linewidth=1)
                ax.plot(xn, yn, 'o', color='#f44747', markersize=8, zorder=5)
            ax.plot(vals[-1], 0, 's', color='#569cd6', markersize=10, zorder=5)
            self.result_label.setText(f"f(x)={expr}\nf'(x)={d1}\n\nIterations:\n" + "\n".join([f"x_{i} = {v:.4f}" for i, v in enumerate(vals)]))
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Newton's Method", color='white'); self.canvas.canvas.draw()

# --- CALCULUS II MODULES ---

class IntegralModule(BaseModule):
    def setup_ui(self):
        self.create_function_input()
        bg = QGroupBox("2. Bounds (a to b)"); bl = QVBoxLayout()
        self.a_slider = LabeledSlider("a =", -10.0, 10.0, 0.0); self.b_slider = LabeledSlider("b =", -10.0, 10.0, 3.0)
        bl.addWidget(self.a_slider); bl.addWidget(self.b_slider); bg.setLayout(bl); self.controls_layout.addWidget(bg)
        rg = QGroupBox("3. Riemann Approximation"); rl = QVBoxLayout(); ml = QHBoxLayout(); ml.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox(); self.method_combo.addItems(["Exact Area", "Left", "Right", "Midpoint", "Trapezoidal"])
        self.method_combo.currentIndexChanged.connect(self.update_plot); ml.addWidget(self.method_combo); rl.addLayout(ml)
        self.n_slider = LabeledSlider("N =", 1, 50, 10, step=1); self.n_slider.spin.setDecimals(0)
        self.n_slider.spin.valueChanged.connect(self.update_plot); rl.addWidget(self.n_slider)
        self.btn_animate = QPushButton("▶ Play N -> ∞"); self.btn_animate.setCheckable(True)
        self.btn_animate.clicked.connect(self.toggle_animation); rl.addWidget(self.btn_animate); rg.setLayout(rl); self.controls_layout.addWidget(rg)
        resg = QGroupBox("4. Results"); resl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); resl.addWidget(self.result_label); resg.setLayout(resl)
        self.controls_layout.addWidget(resg); self.create_view_controls()
        self.timer = QTimer(self); self.timer.timeout.connect(self.animate_step); self.update_plot()

    def toggle_animation(self):
        if self.btn_animate.isChecked(): self.btn_animate.setText("⏸ Stop"); self.n_slider.spin.setValue(1); self.timer.start(150)
        else: self.btn_animate.setText("▶ Play N -> ∞"); self.timer.stop()

    def animate_step(self):
        cn = int(self.n_slider.value())
        if cn < 50: self.n_slider.spin.setValue(cn + 1)
        else: self.timer.stop(); self.btn_animate.setChecked(False); self.btn_animate.setText("▶ Play N -> ∞")

    def update_plot(self):
        av, bv, nv, method = self.a_slider.value(), self.b_slider.value(), int(self.n_slider.value()), self.method_combo.currentText()
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 400)
        yv, fl = self.safe_evaluate(expr, xv); self.canvas.set_trace_data(xv, yv)
        ax.plot(xv, yv, '#4ec9b0', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')
        try:
            ie = sp.integrate(expr, self.x); ea = float(sp.integrate(expr, (self.x, av, bv))); aa = ea
            if method != "Exact Area":
                fa, fb = min(av, bv), max(av, bv); dx = (fb - fa) / nv; aa = 0
                for i in range(nv):
                    if method == "Left": xr, hr = fa + i * dx, fl(fa + i * dx)
                    elif method == "Right": xr, hr = fa + (i + 1) * dx, fl(fa + (i + 1) * dx)
                    elif method == "Midpoint": xr, hr = fa + (i + 0.5) * dx, fl(fa + (i + 0.5) * dx)
                    elif method == "Trapezoidal":
                        xl, xr2 = fa + i * dx, fa + (i + 1) * dx; hl, hr2 = fl(xl), fl(xr2)
                        ax.fill_between([xl, xr2], [hl, hr2], 0, color='#569cd6', alpha=0.4, edgecolor='#fff', linewidth=0.5)
                        aa += 0.5 * (hl + hr2) * dx; continue
                    ax.add_patch(plt.Rectangle((xr, 0), dx, hr, facecolor='#569cd6', alpha=0.4, edgecolor='#fff', linewidth=0.5)); aa += hr * dx
                if av > bv: aa = -aa
            fa, fb = min(av, bv), max(av, bv); xf = np.linspace(fa, fb, 100); yf, _ = self.safe_evaluate(expr, xf)
            ax.fill_between(xf, yf, 0, where=(yf >= 0), color='#4ec9b0', alpha=0.1)
            ax.fill_between(xf, yf, 0, where=(yf < 0), color='#f44747', alpha=0.1)
            ax.axvline(x=av, color='#dcdcaa', linestyle='--'); ax.axvline(x=bv, color='#dcdcaa', linestyle='--')
            self.result_label.setText(f"f(x)={expr}\n∫f(x)dx={ie}+C\n\nExact Area:\n∫[{av:.2f} to {bv:.2f}]={ea:.4f}\n\nMethod: {method}\nApprox: {aa:.4f}\nError: {abs(ea-aa):.4f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Integrals & Riemann Sums", color='white'); self.canvas.canvas.draw()

class AreaBetweenModule(BaseModule):
    def setup_ui(self):
        g = QGroupBox("1. Define Functions f(x) and g(x)"); l = QVBoxLayout()
        fl = QHBoxLayout(); fl.addWidget(QLabel("f(x) =")); self.f_input = QLineEdit("x**2")
        self.f_input.returnPressed.connect(self.update_plot); fl.addWidget(self.f_input); l.addLayout(fl)
        gl = QHBoxLayout(); gl.addWidget(QLabel("g(x) =")); self.g_input = QLineEdit("x")
        self.g_input.returnPressed.connect(self.update_plot); gl.addWidget(self.g_input); l.addLayout(gl)
        b = QPushButton("Update Graph"); b.clicked.connect(self.update_plot); l.addWidget(b); g.setLayout(l); self.controls_layout.addWidget(g)
        bg = QGroupBox("2. Bounds (a to b)"); bl = QVBoxLayout()
        self.a_slider = LabeledSlider("a =", -10.0, 10.0, -1.0); self.b_slider = LabeledSlider("b =", -10.0, 10.0, 2.0)
        bl.addWidget(self.a_slider); bl.addWidget(self.b_slider); bg.setLayout(bl); self.controls_layout.addWidget(bg)
        rg = QGroupBox("3. Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        av, bv = self.a_slider.value(), self.b_slider.value()
        try: ef, eg = sp.sympify(self.f_input.text()), sp.sympify(self.g_input.text())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 400)
        yf, fl = self.safe_evaluate(ef, xv); yg, gl = self.safe_evaluate(eg, xv)
        self.canvas.set_trace_data(xv, yf)
        ax.plot(xv, yf, '#4ec9b0', linewidth=2, label=f'f(x)={sp.latex(ef)}'); ax.plot(xv, yg, '#569cd6', linewidth=2, label=f'g(x)={sp.latex(eg)}')
        try:
            fa, fb = min(av, bv), max(av, bv); xf = np.linspace(fa, fb, 500)
            yff, ygf = fl(xf), gl(xf)
            if isinstance(yff, (int, float)): yff = np.full_like(xf, yff)
            if isinstance(ygf, (int, float)): ygf = np.full_like(xf, ygf)
            na = np.trapz(np.abs(yff - ygf), xf)
            ax.fill_between(xf, yff, ygf, where=(yff >= ygf), color='#4ec9b0', alpha=0.3, interpolate=True)
            ax.fill_between(xf, yff, ygf, where=(yff < ygf), color='#569cd6', alpha=0.3, interpolate=True)
            ax.axvline(x=av, color='#dcdcaa', linestyle='--'); ax.axvline(x=bv, color='#dcdcaa', linestyle='--')
            self.result_label.setText(f"f(x)={ef}\ng(x)={eg}\n\nBounds: {av:.2f} to {bv:.2f}\n\nArea = ∫|f(x)-g(x)|dx\n\nNumerical Area:\nA = {na:.4f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Area Between Curves", color='white'); self.canvas.canvas.draw()

class VolumeModule(BaseModule):
    def setup_ui(self):
        self.create_function_input(default="sin(x)")
        bg = QGroupBox("2. Bounds (a to b)"); bl = QVBoxLayout()
        self.a_slider = LabeledSlider("a =", -10.0, 10.0, -3.0); self.b_slider = LabeledSlider("b =", -10.0, 10.0, 3.0)
        bl.addWidget(self.a_slider); bl.addWidget(self.b_slider); bg.setLayout(bl); self.controls_layout.addWidget(bg)
        rg = QGroupBox("3. Volume Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(is_3d=True); self.update_plot()

    def update_plot(self):
        av, bv = self.a_slider.value(), self.b_slider.value()
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111, projection='3d'); ax.clear(); ax.set_facecolor("#252526")
        self.canvas.set_trace_data(None, None)
        try:
            fa, fb = min(av, bv), max(av, bv); xv = np.linspace(fa, fb, 50)
            fl = sp.lambdify(self.x, expr, "numpy"); R = fl(xv)
            if isinstance(R, (int, float)): R = np.full_like(xv, R, dtype=float)
            R = np.abs(R); Th = np.linspace(0, 2*np.pi, 50); X, Th = np.meshgrid(xv, Th)
            Y, Z = R * np.cos(Th), R * np.sin(Th); ax.plot_surface(X, Y, Z, alpha=0.7, color='#4ec9b0', edgecolor='none')
            ve = sp.integrate(sp.pi * expr**2, self.x); vv = float(sp.integrate(sp.pi * expr**2, (self.x, fa, fb)))
            self.result_label.setText(f"f(x)={expr}\n\nBounds: {fa:.2f} to {fb:.2f}\n\nV = π ∫[f(x)]² dx\n\nIntegral: {ve}\n\nVolume: {vv:.4f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.set_xlabel('X', color='#cccccc'); ax.set_ylabel('Y', color='#cccccc'); ax.set_zlabel('Z', color='#cccccc')
        ax.tick_params(colors='#cccccc'); ax.set_title("Volume of Revolution", color='white'); self.canvas.canvas.draw()

class ArcLengthModule(BaseModule):
    def setup_ui(self):
        self.create_function_input(default="x**2")
        bg = QGroupBox("2. Bounds (a to b)"); bl = QVBoxLayout()
        self.a_slider = LabeledSlider("a =", -10.0, 10.0, -3.0); self.b_slider = LabeledSlider("b =", -10.0, 10.0, 3.0)
        bl.addWidget(self.a_slider); bl.addWidget(self.b_slider); bg.setLayout(bl); self.controls_layout.addWidget(bg)
        rg = QGroupBox("3. Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        av, bv = self.a_slider.value(), self.b_slider.value()
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 400)
        yv, _ = self.safe_evaluate(expr, xv); self.canvas.set_trace_data(xv, yv)
        ax.plot(xv, yv, '#4ec9b0', linewidth=2, alpha=0.5, label=f'$f(x) = {sp.latex(expr)}$')
        try:
            fa, fb = min(av, bv), max(av, bv); xa = np.linspace(fa, fb, 400); ya, _ = self.safe_evaluate(expr, xa)
            ax.plot(xa, ya, '#f44747', linewidth=3, label="Arc Length")
            d1 = sp.diff(expr, self.x); L_expr = sp.sqrt(1 + d1**2)
            try: L = float(sp.integrate(L_expr, (self.x, fa, fb)))
            except: L = float(np.trapz(sp.lambdify(self.x, L_expr, "numpy")(xa), xa))
            try: SA = float(sp.integrate(2*sp.pi*expr*L_expr, (self.x, fa, fb)))
            except: SA = float(np.trapz(2*np.pi*ya*sp.lambdify(self.x, L_expr, "numpy")(xa), xa))
            self.result_label.setText(f"f(x)={expr}\nf'(x)={d1}\n\nBounds: {fa:.2f} to {fb:.2f}\n\nArc Length:\nL = ∫√(1+[f']²) dx\nL = {L:.4f}\n\nSurface Area:\nSA = 2π ∫ f(x)√(1+[f']²) dx\nSA = {SA:.4f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Arc Length & Surface Area", color='white'); self.canvas.canvas.draw()

class TaylorModule(BaseModule):
    def setup_ui(self):
        self.create_function_input(default="exp(x)")
        cg = QGroupBox("2. Series Center (a)"); cl = QVBoxLayout()
        self.a_slider = LabeledSlider("a =", -10.0, 10.0, 0.0)
        self.a_slider.spin.valueChanged.connect(self.update_plot); cl.addWidget(self.a_slider); cg.setLayout(cl); self.controls_layout.addWidget(cg)
        og = QGroupBox("3. Polynomial Order (n)"); ol = QVBoxLayout()
        self.n_slider = LabeledSlider("n =", 1, 15, 3, step=1); self.n_slider.spin.setDecimals(0)
        self.n_slider.spin.valueChanged.connect(self.update_plot); ol.addWidget(self.n_slider); og.setLayout(ol); self.controls_layout.addWidget(og)
        rg = QGroupBox("4. Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        av, nv = self.a_slider.value(), int(self.n_slider.value())
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        xv = np.linspace(self.x_min_spin.value(), self.x_max_spin.value(), 400)
        yv, _ = self.safe_evaluate(expr, xv); self.canvas.set_trace_data(xv, yv)
        ax.plot(xv, yv, '#4ec9b0', linewidth=2, label=f'$f(x) = {sp.latex(expr)}$')
        try:
            te = sp.series(expr, self.x, self.x - av, nv + 1).removeO(); tl = sp.lambdify(self.x, te, "numpy")(xv)
            if isinstance(tl, (int, float)): tl = np.full_like(xv, tl, dtype=float)
            ax.plot(xv, tl, '#f44747', '--', linewidth=1.5, label=f'$P_{nv}(x)$')
            ax.plot(av, float(expr.subs(self.x, av)), 'o', color='#f44747', markersize=8, zorder=5)
            self.result_label.setText(f"f(x)={expr}\n\nCenter a={av:.2f}\nOrder n={nv}\n\nTaylor Poly:\nP_{nv}(x)={te}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Taylor Series", color='white'); self.canvas.canvas.draw()

class SequenceModule(BaseModule):
    def setup_ui(self):
        g = QGroupBox("1. Define Sequence a_n"); l = QVBoxLayout()
        il = QHBoxLayout(); il.addWidget(QLabel("a_n =")); self.func_input = QLineEdit("1/n")
        self.func_input.returnPressed.connect(self.update_plot); il.addWidget(self.func_input); l.addLayout(il)
        pl = QGridLayout()
        for i, p in enumerate(["1/n", "1/n**2", "(1+1/n)**n", "(-1)**n", "n/(n+1)", "1/factorial(n)"]):
            b = QPushButton(p); b.setFixedHeight(24); b.clicked.connect(lambda c, x=p: (self.func_input.setText(x), self.update_plot())); pl.addWidget(b, i // 3, i % 3)
        l.addLayout(pl); b = QPushButton("Update Graph"); b.clicked.connect(self.update_plot); l.addWidget(b); g.setLayout(l); self.controls_layout.addWidget(g)
        ng = QGroupBox("2. Number of Terms (N)"); nl = QVBoxLayout()
        self.n_slider = LabeledSlider("N =", 1, 50, 20, step=1); self.n_slider.spin.setDecimals(0)
        self.n_slider.spin.valueChanged.connect(self.update_plot); nl.addWidget(self.n_slider); ng.setLayout(nl); self.controls_layout.addWidget(ng)
        rg = QGroupBox("3. Convergence"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        nv = int(self.n_slider.value())
        try: expr = sp.sympify(self.func_input.text())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        try:
            n = np.arange(1, nv + 1); al = sp.lambdify(self.n_sym, expr, "numpy"); A = al(n)
            if isinstance(A, (int, float)): A = np.full_like(n, A, dtype=float)
            ax.plot(n, A, 'o', color='#4ec9b0', markersize=6, label=f'$a_n = {sp.latex(expr)}$')
            S = np.cumsum(A); ax.plot(n, S, 's-', color='#f44747', markersize=4, linewidth=1, label=f'$S_n = \\sum a_n$')
            self.canvas.set_trace_data(n, A)
            ls = sp.limit(expr, self.n_sym, sp.oo)
            try: sv = float(sp.summation(expr, (self.n_sym, 1, sp.oo))); st = f"{sv:.4f} (Converges)"
            except: st = "Diverges or complex"
            self.result_label.setText(f"a_n={expr}\n\nlim a_n = {ls}\n\nSeries Sum:\n  {st}\n\nLast Term:\n  a_{nv}={A[-1]:.4f}\nPartial Sum:\n  S_{nv}={S[-1]:.4f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Sequences & Series", color='white'); self.canvas.canvas.draw()

class ODEModule(BaseModule):
    def setup_ui(self):
        g = QGroupBox("1. Differential Equation"); l = QVBoxLayout()
        il = QHBoxLayout(); il.addWidget(QLabel("dy/dx =")); self.func_input = QLineEdit("x - y")
        self.func_input.returnPressed.connect(self.update_plot); il.addWidget(self.func_input); l.addLayout(il)
        pl = QGridLayout()
        for i, p in enumerate(["x - y", "y", "x*y", "sin(x)", "y**2 - x", "1 - y/x"]):
            b = QPushButton(p); b.setFixedHeight(24); b.clicked.connect(lambda c, x=p: (self.func_input.setText(x), self.update_plot())); pl.addWidget(b, i // 3, i % 3)
        l.addLayout(pl); b = QPushButton("Update Graph"); b.clicked.connect(self.update_plot); l.addWidget(b); g.setLayout(l); self.controls_layout.addWidget(g)
        ig = QGroupBox("2. Initial Condition (x0, y0)"); il = QVBoxLayout()
        self.x0_slider = LabeledSlider("x0 =", -5.0, 5.0, 0.0); self.y0_slider = LabeledSlider("y0 =", -5.0, 5.0, 0.0)
        self.x0_slider.spin.valueChanged.connect(self.update_plot); self.y0_slider.spin.valueChanged.connect(self.update_plot)
        il.addWidget(self.x0_slider); il.addWidget(self.y0_slider); ig.setLayout(il); self.controls_layout.addWidget(ig)
        rg = QGroupBox("3. Info"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        x0, y0 = self.x0_slider.value(), self.y0_slider.value()
        try: expr = sp.sympify(self.func_input.text()); fl = sp.lambdify((self.x, self.y), expr, "numpy")
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        x_min, x_max, y_min, y_max = self.x_min_spin.value(), self.x_max_spin.value(), self.y_min_spin.value(), self.y_max_spin.value()
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20)); self.canvas.set_trace_data(None, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            U, V = np.ones_like(X), fl(X, Y); N = np.sqrt(U**2 + V**2); U, V = U/N, V/N
        ax.quiver(X, Y, U, V, color='#569cd6', alpha=0.6)
        try:
            xs, ys = [x0], [y0]; step = 0.05; cx, cy = x0, y0
            while cx < x_max: cy = cy + step * fl(cx, cy); cx += step; xs.append(cx); ys.append(cy)
            cx, cy = x0, y0
            while cx > x_min: cy = cy - step * fl(cx, cy); cx -= step; xs.insert(0, cx); ys.insert(0, cy)
            ax.plot(xs, ys, '#f44747', linewidth=2, label="Solution"); ax.plot(x0, y0, 'o', color='#f44747', markersize=8, zorder=5)
            self.result_label.setText(f"dy/dx={expr}\n\ny({x0:.2f})={y0:.2f}\n\nRed curve is the\napproximate solution.")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Differential Eqs", color='white'); self.canvas.canvas.draw()

# --- CALCULUS III MODULES ---

class ParametricModule(BaseModule):
    def setup_ui(self):
        g = QGroupBox("1. Parametric Equations"); l = QVBoxLayout()
        xl = QHBoxLayout(); xl.addWidget(QLabel("x(t) =")); self.x_input = QLineEdit("cos(t)")
        self.x_input.returnPressed.connect(self.update_plot); xl.addWidget(self.x_input); l.addLayout(xl)
        yl = QHBoxLayout(); yl.addWidget(QLabel("y(t) =")); self.y_input = QLineEdit("sin(t)")
        self.y_input.returnPressed.connect(self.update_plot); yl.addWidget(self.y_input); l.addLayout(yl)
        b = QPushButton("Update Graph"); b.clicked.connect(self.update_plot); l.addWidget(b); g.setLayout(l); self.controls_layout.addWidget(g)
        tg = QGroupBox("2. Time (t)"); tl = QVBoxLayout()
        self.t_slider = LabeledSlider("t =", 0.0, 6.28, 3.14)
        self.t_slider.spin.valueChanged.connect(self.update_plot); tl.addWidget(self.t_slider); tg.setLayout(tl); self.controls_layout.addWidget(tg)
        rg = QGroupBox("3. Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        tv = self.t_slider.value()
        try: xe, ye = sp.sympify(self.x_input.text()), sp.sympify(self.y_input.text())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax); ax.set_aspect('equal', adjustable='box')
        tvl = np.linspace(0, 6.28, 400); self.canvas.set_trace_data(None, None)
        try:
            X, Y = sp.lambdify(self.t, xe, "numpy")(tvl), sp.lambdify(self.t, ye, "numpy")(tvl)
            ax.plot(X, Y, '#4ec9b0', linewidth=2, label='Curve')
            dx, dy = sp.diff(xe, self.t), sp.diff(ye, self.t); ae = sp.sqrt(dx**2 + dy**2)
            al = float(sp.integrate(ae, (self.t, 0, 2*sp.pi)))
            px, py = float(xe.subs(self.t, tv)), float(ye.subs(self.t, tv))
            vx, vy = float(dx.subs(self.t, tv)), float(dy.subs(self.t, tv))
            ax.plot(px, py, 'o', color='#f44747', markersize=8, zorder=5)
            ax.quiver(px, py, vx, vy, color='#f44747', scale=5, zorder=5, label="Velocity")
            self.result_label.setText(f"x(t)={xe}\ny(t)={ye}\n\nArc Length(0-2π): {al:.4f}\n\nAt t={tv:.2f}:\nPos=({px:.2f},{py:.2f})\nVel=<{vx:.2f},{vy:.2f}>")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Parametric Equations", color='white'); self.canvas.canvas.draw()

class PolarModule(BaseModule):
    def setup_ui(self):
        g = QGroupBox("1. Polar Equation r(t)"); l = QVBoxLayout()
        il = QHBoxLayout(); il.addWidget(QLabel("r = ")); self.func_input = QLineEdit("2*cos(4*t)")
        self.func_input.returnPressed.connect(self.update_plot); il.addWidget(self.func_input); l.addLayout(il)
        pl = QGridLayout()
        for i, p in enumerate(["2*cos(4*t)", "1+cos(t)", "sin(3*t)", "t", "2", "2*sin(2*t)"]):
            b = QPushButton(p); b.setFixedHeight(24); b.clicked.connect(lambda c, x=p: (self.func_input.setText(x), self.update_plot())); pl.addWidget(b, i // 3, i % 3)
        l.addLayout(pl); b = QPushButton("Update Graph"); b.clicked.connect(self.update_plot); l.addWidget(b); g.setLayout(l); self.controls_layout.addWidget(g)
        sg = QGroupBox("2. Sweep Angle"); sl = QVBoxLayout()
        self.t_slider = LabeledSlider("θ_max =", 0.0, 6.283, 3.14)
        self.t_slider.spin.valueChanged.connect(self.update_plot); sl.addWidget(self.t_slider); sg.setLayout(sl); self.controls_layout.addWidget(sg)
        rg = QGroupBox("3. Results"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def update_plot(self):
        tm = self.t_slider.value()
        try: expr = sp.sympify(self.func_input.text())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax); ax.set_aspect('equal', adjustable='box')
        tvl = np.linspace(0, 2*np.pi, 1000); self.canvas.set_trace_data(None, None)
        try:
            R = sp.lambdify(self.t, expr, "numpy")(tvl)
            if isinstance(R, (int, float)): R = np.full_like(tvl, R, dtype=float)
            R = np.where(R < 0, -R, R); ta = np.where(R < 0, tvl + np.pi, tvl)
            X, Y = R * np.cos(ta), R * np.sin(ta); ax.plot(X, Y, '#4ec9b0', linewidth=2, label=f'$r = {sp.latex(expr)}$')
            ea = float(sp.integrate(expr**2 / 2, (self.t, 0, 2*sp.pi))); dr = sp.diff(expr, self.t)
            try: el = float(sp.integrate(sp.sqrt(expr**2 + dr**2), (self.t, 0, 2*sp.pi)))
            except: el = float(np.trapz(np.abs(sp.lambdify(self.t, sp.sqrt(expr**2 + dr**2), "numpy")(tvl)), tvl))
            ts = np.linspace(0, tm, 100); Rs = sp.lambdify(self.t, expr, "numpy")(ts)
            if isinstance(Rs, (int, float)): Rs = np.full_like(ts, Rs)
            Rs = np.where(Rs < 0, -Rs, Rs); tsa = np.where(Rs < 0, ts + np.pi, ts)
            ax.fill_between(Rs * np.cos(tsa), Rs * np.sin(tsa), 0, color='#f44747', alpha=0.2)
            sa = float(sp.integrate(expr**2 / 2, (self.t, 0, tm)))
            self.result_label.setText(f"r={expr}\n\nTotal Area: {ea:.4f}\nSweep Area: {sa:.4f}\nArc Length: {el:.4f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.legend(loc='upper left', facecolor='#2d2d2d', edgecolor='#555555'); ax.set_title("Polar Coordinates", color='white'); self.canvas.canvas.draw()

class MultivariableModule(BaseModule):
    def setup_ui(self):
        self.create_function_input(default="x**2 + y**2", presets=["x**2 + y**2", "sin(x)*y", "x*y", "exp(-(x**2+y**2))"], var1='x, y')
        pg = QGroupBox("2. Point (x, y)"); pl = QVBoxLayout()
        self.x_slider = LabeledSlider("x =", -5.0, 5.0, 1.0); self.y_slider = LabeledSlider("y =", -5.0, 5.0, 1.0)
        self.x_slider.spin.valueChanged.connect(self.update_plot); self.y_slider.spin.valueChanged.connect(self.update_plot)
        pl.addWidget(self.x_slider); pl.addWidget(self.y_slider); pg.setLayout(pl); self.controls_layout.addWidget(pg)
        vg = QGroupBox("3. View"); vl = QVBoxLayout(); self.chk_lock = QCheckBox("Lock Camera"); vl.addWidget(self.chk_lock); vg.setLayout(vl); self.controls_layout.addWidget(vg)
        rg = QGroupBox("4. Partials & Tangent"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(is_3d=True)
        self.azim, self.elev = -60, 25; self.update_plot()

    def update_plot(self):
        xp, yp = self.x_slider.value(), self.y_slider.value()
        try: expr = sp.sympify(self.func_input.currentText()); self.add_to_history(self.func_input.currentText())
        except: return
        ax = self.canvas.figure.add_subplot(111, projection='3d'); ax.clear(); ax.set_facecolor("#252526")
        self.canvas.set_trace_data(None, None)
        x_min, x_max = self.x_min_spin.value(), self.x_max_spin.value()
        xv, yv = np.linspace(x_min, x_max, 30), np.linspace(x_min, x_max, 30); X, Y = np.meshgrid(xv, yv)
        try:
            Z = sp.lambdify((self.x, self.y), expr, "numpy")(X, Y)
            if isinstance(Z, (int, float)): Z = np.full_like(X, Z, dtype=float)
            Z = np.where(np.isreal(Z), Z, np.nan); Z[np.abs(Z) > 100] = np.nan
            ax.plot_surface(X, Y, Z, alpha=0.6, color='#4ec9b0', edgecolor='none')
            zp = float(expr.subs({self.x: xp, self.y: yp}))
            dx, dy = sp.diff(expr, self.x), sp.diff(expr, self.y)
            sx, sy = float(dx.subs({self.x: xp, self.y: yp})), float(dy.subs({self.x: xp, self.y: yp}))
            Zt = zp + sx * (X - xp) + sy * (Y - yp)
            ax.plot_surface(X, Y, Zt, alpha=0.4, color='#f44747', edgecolor='none')
            ax.scatter(xp, yp, zp, color='#f44747', s=50, zorder=5)
            ax.quiver(xp, yp, np.nanmin(Z), sx, sy, 0, color='red', arrow_length_ratio=0.2, linewidth=2)
            self.result_label.setText(f"f(x,y)={expr}\n\nPoint: ({xp:.2f}, {yp:.2f})\nz={zp:.2f}\n\n∂f/∂x={dx}\n∂f/∂y={dy}\n\nSlope x: {sx:.2f}\nSlope y: {sy:.2f}")
            if self.chk_lock.isChecked(): ax.view_init(elev=self.elev, azim=self.azim)
            else: self.azim, self.elev = ax.azim, ax.elev
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.set_xlabel('X', color='#cccccc'); ax.set_ylabel('Y', color='#cccccc'); ax.set_zlabel('Z', color='#cccccc')
        ax.tick_params(colors='#cccccc'); ax.set_title("3D Multivariable", color='white'); self.canvas.canvas.draw()

class VectorModule(BaseModule):
    def setup_ui(self):
        g = QGroupBox("1. Vector Field F(x,y)"); l = QVBoxLayout()
        il = QHBoxLayout(); il.addWidget(QLabel("F = <")); self.p_input = QLineEdit("y"); il.addWidget(self.p_input)
        il.addWidget(QLabel(",")); self.q_input = QLineEdit("-x"); il.addWidget(self.q_input); il.addWidget(QLabel(">"))
        self.p_input.returnPressed.connect(self.update_plot); self.q_input.returnPressed.connect(self.update_plot); l.addLayout(il)
        pl = QGridLayout()
        for i, p in enumerate(["<y, -x>", "<x, y>", "<-x, -y>", "<x, 0>", "<0, y>", "<y, x>"]):
            b = QPushButton(p); b.setFixedHeight(24); b.clicked.connect(lambda c, x=p: self.set_preset(x)); pl.addWidget(b, i // 2, i % 2)
        l.addLayout(pl); b = QPushButton("Update Graph"); b.clicked.connect(self.update_plot); l.addWidget(b); g.setLayout(l); self.controls_layout.addWidget(g)
        pg = QGroupBox("2. Point (x, y)"); pl = QVBoxLayout()
        self.x_slider = LabeledSlider("x =", -5.0, 5.0, 2.0); self.y_slider = LabeledSlider("y =", -5.0, 5.0, 2.0)
        self.x_slider.spin.valueChanged.connect(self.update_plot); self.y_slider.spin.valueChanged.connect(self.update_plot)
        pl.addWidget(self.x_slider); pl.addWidget(self.y_slider); pg.setLayout(pl); self.controls_layout.addWidget(pg)
        rg = QGroupBox("3. Div & Curl"); rl = QVBoxLayout()
        self.result_label = QLabel("Calculating..."); self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-family: monospace; font-size: 13px;"); rl.addWidget(self.result_label); rg.setLayout(rl)
        self.controls_layout.addWidget(rg); self.create_view_controls(); self.update_plot()

    def set_preset(self, p):
        ps = p.split("<")[1].split(">")[0].split(",")
        self.p_input.setText(ps[0].strip()); self.q_input.setText(ps[1].strip()); self.update_plot()

    def update_plot(self):
        xp, yp = self.x_slider.value(), self.y_slider.value()
        try: P, Q = sp.sympify(self.p_input.text()), sp.sympify(self.q_input.text())
        except: return
        ax = self.canvas.figure.add_subplot(111); self.setup_canvas_axes(ax)
        x_min, x_max, y_min, y_max = self.x_min_spin.value(), self.x_max_spin.value(), self.y_min_spin.value(), self.y_max_spin.value()
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20)); self.canvas.set_trace_data(None, None)
        try:
            U, V = sp.lambdify((self.x, self.y), P, "numpy")(X, Y), sp.lambdify((self.x, self.y), Q, "numpy")(X, Y)
            if isinstance(U, (int, float)): U = np.full_like(X, U, dtype=float)
            if isinstance(V, (int, float)): V = np.full_like(X, V, dtype=float)
            M = np.hypot(U, V); M[M == 0] = 1.0
            ax.quiver(X, Y, U/M, V/M, M, cmap='coolwarm', alpha=0.7)
            dv, cv = float((sp.diff(P, self.x) + sp.diff(Q, self.y)).subs({self.x: xp, self.y: yp})), float((sp.diff(Q, self.x) - sp.diff(P, self.y)).subs({self.x: xp, self.y: yp}))
            if dv > 0.1: ax.add_patch(plt.Circle((xp, yp), 0.5, color='red', fill=False)); ax.plot(xp, yp, 'o', color='red', markersize=4)
            elif dv < -0.1: ax.add_patch(plt.Circle((xp, yp), 0.5, color='blue', fill=False)); ax.plot(xp, yp, 'o', color='blue', markersize=4)
            elif abs(cv) > 0.1: ax.add_patch(plt.Arrow(xp, yp, 0.4, 0.4 if cv > 0 else -0.4, width=0.2, color='green'))
            else: ax.plot(xp, yp, 'o', color='white', markersize=6)
            self.result_label.setText(f"F=<{P}, {Q}>\n\nAt ({xp:.2f}, {yp:.2f}):\nDiv={dv:.2f}\nCurl={cv:.2f}")
        except Exception as e: self.result_label.setText(f"Error:\n{e}")
        ax.set_title("Vector Calculus", color='white'); self.canvas.canvas.draw()


# --- WORKSPACE & MAIN WINDOW ---

class Workspace(QWidget):
    def __init__(self, modules, status_bar):
        super().__init__()
        self.status_bar = status_bar
        self.main_layout = QHBoxLayout(self)
        
        # Left Sidebar
        self.sidebar = QWidget(); self.sidebar.setFixedWidth(400)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        
        # Topic Selector
        self.topic_label = QLabel("Select Topic:"); self.topic_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.topic_combo = QComboBox()
        for name, _ in modules: self.topic_combo.addItem(name)
        self.topic_combo.setStyleSheet("font-size: 14px; padding: 6px;")
        
        # Stack for modules
        self.stack = QStackedWidget()
        self.canvas = GraphCanvas(status_bar)
        
        self.module_instances = []
        for name, mod_cls in modules:
            mod = mod_cls(self.canvas, status_bar)
            self.module_instances.append(mod)
            self.stack.addWidget(mod)
            
        self.topic_combo.currentIndexChanged.connect(self.on_topic_changed)
        self.sidebar_layout.addWidget(self.topic_label)
        self.sidebar_layout.addWidget(self.topic_combo)
        self.sidebar_layout.addWidget(self.stack)
        
        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addWidget(self.canvas, 1)
        
        self.on_topic_changed(0)
        
    def on_topic_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        self.module_instances[idx].update_plot()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate Calculus Master Suite - Consolidated")
        self.resize(1200, 800)
        self.tabs = QTabWidget(); self.setCentralWidget(self.tabs); self.setStatusBar(QStatusBar())
        
        calc1_modules = [("Limits", LimitModule), ("Derivatives", DerivativeModule), ("Curve Analysis", AnalysisModule), ("Newton's Method", NewtonModule)]
        calc2_modules = [("Integrals", IntegralModule), ("Area Between Curves", AreaBetweenModule), ("Volume of Revolution", VolumeModule), ("Arc Length & Surface Area", ArcLengthModule), ("Taylor Series", TaylorModule), ("Sequences & Series", SequenceModule), ("Differential Equations", ODEModule)]
        calc3_modules = [("Parametric Equations", ParametricModule), ("Polar Coordinates", PolarModule), ("3D Multivariable", MultivariableModule), ("Vector Calculus", VectorModule)]
        
        self.tabs.addTab(Workspace(calc1_modules, self.statusBar()), "Calculus I")
        self.tabs.addTab(Workspace(calc2_modules, self.statusBar()), "Calculus II")
        self.tabs.addTab(Workspace(calc3_modules, self.statusBar()), "Calculus III")
        
        self.statusBar().showMessage("Ready. Select a topic from the dropdown to begin learning.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow(); window.show()
    sys.exit(app.exec())