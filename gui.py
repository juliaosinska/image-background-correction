# GENAI: Gemini and Claude for UI full UI implementation (pretty much whole file)
# Gemini - base code for GUI layout, widgets, and event handling (from previous tasks)
# Claude - adaptation for this task (parameter panels for the 3 background methods, matplotlib
#          views: I / B / R images with a fixed gray scale for negative values, histograms with
#          shared bins incl. negative values, "Compare all methods" view)

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from background_methods import blur_background, tophat_background, corner_background

METHOD_BLUR = "1. Gaussian blur"
METHOD_TOPHAT = "2. White Top-hat"
METHOD_CORNER = "3. Corner sampling (bilinear)"
METHOD_COMPARE = "Compare all methods"

# Shared histogram bins for ALL images (I, B, R): one bin per integer value, incl. negatives
HIST_EDGES = np.arange(-255.5, 256.5, 1.0)


class BackgroundCorrectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Estimation and Correction")
        self.root.geometry("1300x850")

        # --- Application state ---
        self.image_array = None   # I: np.ndarray, uint8, H x W

        # --- 1. Top Control Panel ---
        control_frame = tk.LabelFrame(root, text="Control Panel", padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

        tk.Button(control_frame, text="Load Image", command=self.upload_image).pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Method:").pack(side=tk.LEFT, padx=(15, 2))
        self.method_var = tk.StringVar(value=METHOD_BLUR)
        self.method_dropdown = ttk.Combobox(
            control_frame, textvariable=self.method_var, state="readonly", width=30,
            values=[METHOD_BLUR, METHOD_TOPHAT, METHOD_CORNER, METHOD_COMPARE],
        )
        self.method_dropdown.pack(side=tk.LEFT, padx=5)
        self.method_dropdown.bind("<<ComboboxSelected>>", self.on_method_changed)

        tk.Button(control_frame, text="Process Image", command=self.process_image, bg="#e0e0e0").pack(side=tk.LEFT, padx=15)

        # Histogram log-scale toggle
        self.log_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Histogram log scale", variable=self.log_var,
                       command=self.on_log_toggled).pack(side=tk.RIGHT, padx=5)

        # --- 2. Method Parameters (one sub-panel per method) ---
        self.params_frame = tk.LabelFrame(root, text="Method Parameters", padx=10, pady=5)
        self.params_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Method 1: blur
        self.blur_frame = tk.LabelFrame(self.params_frame, text="1. Gaussian blur", padx=5, pady=3)
        self.blur_sigma_var = tk.DoubleVar(value=15.0)
        self.blur_size_var = tk.IntVar(value=61)
        self.add_spinbox(self.blur_frame, "σ:", self.blur_sigma_var, 0.5, 100, 0.5)
        self.add_spinbox(self.blur_frame, "Mask size (odd):", self.blur_size_var, 3, 301, 2)

        # Method 2: top-hat
        self.tophat_frame = tk.LabelFrame(self.params_frame, text="2. White Top-hat", padx=5, pady=3)
        self.se_shape_var = tk.StringVar(value="Disk")
        self.se_size_var = tk.IntVar(value=31)
        self.add_combobox(self.tophat_frame, "SE shape:", self.se_shape_var, ["Square", "Disk"])
        self.add_spinbox(self.tophat_frame, "SE size (odd):", self.se_size_var, 3, 301, 2)

        # Method 3: corners + bilinear interpolation
        self.corner_frame = tk.LabelFrame(self.params_frame, text="3. Corners", padx=5, pady=3)
        self.corner_size_var = tk.IntVar(value=20)
        self.corner_sigma_var = tk.DoubleVar(value=2.0)
        self.add_spinbox(self.corner_frame, "Corner size:", self.corner_size_var, 1, 500, 1)
        self.add_spinbox(self.corner_frame, "σ (for R2):", self.corner_sigma_var, 0.5, 50, 0.5)

        # --- 3. Preview: images (top) and histograms (bottom), both matplotlib ---
        preview = tk.Frame(root)
        preview.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(0, weight=3, uniform="row")
        preview.grid_rowconfigure(1, weight=2, uniform="row")

        images_frame = tk.LabelFrame(preview, text="Images")
        images_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.img_figure = Figure(figsize=(4, 2), dpi=100, layout="constrained")
        self.img_canvas = FigureCanvasTkAgg(self.img_figure, master=images_frame)
        self.img_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        hist_frame = tk.LabelFrame(preview, text="Histograms (shared bins, incl. negative values)")
        hist_frame.grid(row=1, column=0, sticky="nsew")
        self.hist_figure = Figure(figsize=(4, 2), dpi=100, layout="constrained")
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, master=hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Last drawn histogram data, so the log toggle can redraw without recomputing
        self.last_histograms = None

        # --- 4. Status Bar ---
        self.status_var = tk.StringVar(value="Ready. Please load an image.")
        tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W,
                 padx=5, pady=3).pack(side=tk.BOTTOM, fill=tk.X)

        self.update_params_visibility()

    # ------------------------------------------------------------------
    # Small widget helpers
    # ------------------------------------------------------------------
    def add_combobox(self, parent, label, variable, values):
        tk.Label(parent, text=label).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly",
                     width=9).pack(side=tk.LEFT, padx=(0, 8))

    def add_spinbox(self, parent, label, variable, from_, to, increment):
        tk.Label(parent, text=label).pack(side=tk.LEFT, padx=(5, 2))
        tk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=variable,
                   width=6).pack(side=tk.LEFT, padx=(0, 8))

    # ------------------------------------------------------------------
    # Method selection
    # ------------------------------------------------------------------
    def update_params_visibility(self):
        method = self.method_var.get()
        for frame in (self.blur_frame, self.tophat_frame, self.corner_frame):
            frame.pack_forget()

        frames = {
            METHOD_BLUR: [self.blur_frame],
            METHOD_TOPHAT: [self.tophat_frame],
            METHOD_CORNER: [self.corner_frame],
            METHOD_COMPARE: [self.blur_frame, self.tophat_frame, self.corner_frame],
        }[method]
        for frame in frames:
            frame.pack(side=tk.LEFT, padx=5)

    def on_method_changed(self, event=None):
        self.update_params_visibility()
        self.status_var.set(f"Selected: {self.method_var.get()} - click 'Process Image'")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def read_odd(self, variable, name):
        value = int(variable.get())
        if value < 3 or value % 2 == 0:
            raise ValueError(f"{name} must be an odd integer >= 3.")
        return value

    def run_method(self, method):
        """Runs one method and returns a dict: name, params, B, results [(label, R), ...]."""
        image = self.image_array

        if method == METHOD_BLUR:
            sigma = float(self.blur_sigma_var.get())
            size = self.read_odd(self.blur_size_var, "Mask size")
            B, R = blur_background(image, sigma, size)
            return {"name": "Gaussian blur", "params": f"σ={sigma}, mask {size}x{size}", "B": B,
                    "results": [("R = I − Gσ(I)", R)]}

        if method == METHOD_TOPHAT:
            shape = self.se_shape_var.get()
            size = self.read_odd(self.se_size_var, "SE size")
            B, R = tophat_background(image, shape, size)
            return {"name": "White Top-hat", "params": f"SE {shape} {size}x{size}", "B": B,
                    "results": [("R = I − (I∘SE)", R)]}

        if method == METHOD_CORNER:
            corner_size = int(self.corner_size_var.get())
            sigma = float(self.corner_sigma_var.get())
            h, w = image.shape
            if not 1 <= corner_size <= min(h, w) // 2:
                raise ValueError(f"Corner size must be between 1 and {min(h, w) // 2} for this image.")
            B, R1, R2 = corner_background(image, corner_size, sigma)
            return {"name": "Corners", "params": f"corner {corner_size}px, σ={sigma}", "B": B,
                    "results": [("R1 = I − B", R1), (f"R2 = Gσ(I) − B", R2)]}

        raise ValueError(f"Unknown method: {method}")

    def process_image(self):
        if self.image_array is None:
            self.status_var.set("Load an image first.")
            return

        method = self.method_var.get()
        methods = [METHOD_BLUR, METHOD_TOPHAT, METHOD_CORNER] if method == METHOD_COMPARE else [method]

        self.status_var.set("Computing...")
        self.root.update_idletasks()
        try:
            outputs = [self.run_method(m) for m in methods]
        except NotImplementedError as e:
            self.status_var.set(f"Not implemented yet: {e} (background_methods.py)")
            return
        except (ValueError, tk.TclError) as e:
            self.status_var.set(f"Invalid parameter: {e}")
            return

        if method == METHOD_COMPARE:
            self.show_compare(outputs)
        else:
            self.show_single(outputs[0])

    # ------------------------------------------------------------------
    # Drawing: images
    # ------------------------------------------------------------------
    @staticmethod
    def result_range(arrays):
        # One symmetric display range for all results in the view, so they can be compared
        return max(1.0, max(float(np.abs(a).max()) for a in arrays))

    def draw_image(self, ax, title, array, kind, limit=None):
        if kind == "image":
            # I and B: plain gray scale 0..255
            ax.imshow(array, cmap="gray", vmin=0, vmax=255)
        else:
            # R: symmetric range [-limit, limit]; 0 -> middle gray (display only, data not changed)
            im = ax.imshow(array, cmap="gray", vmin=-limit, vmax=limit)
            self.img_figure.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def show_single(self, out):
        results = out["results"]
        limit = self.result_range([R for _, R in results])
        panels = [("Original I", self.image_array, "image"),
                  (f"Background B\n{out['name']}: {out['params']}", out["B"], "image")]
        panels += [(label, R, "result") for label, R in results]

        self.img_figure.clear()
        axes = self.img_figure.subplots(1, len(panels), squeeze=False)[0]
        for ax, (title, array, kind) in zip(axes, panels):
            self.draw_image(ax, title, array, kind, limit)
        self.img_canvas.draw_idle()

        self.draw_histograms(
            [("I", self.image_array), ("B", out["B"])],
            [(label.split(" =")[0], R) for label, R in results],
        )
        self.status_var.set(f"{out['name']} ({out['params']}) - R display range ±{limit:.0f}")

    def show_compare(self, outputs):
        # Columns = methods; row 1 = background maps B, row 2 = corrected images R (R1 for corners)
        main_results = [out["results"][0] for out in outputs]
        limit = self.result_range([R for _, R in main_results])

        self.img_figure.clear()
        axes = self.img_figure.subplots(2, len(outputs), squeeze=False)
        for col, (out, (label, R)) in enumerate(zip(outputs, main_results)):
            self.draw_image(axes[0][col], f"B: {out['name']}\n{out['params']}", out["B"], "image")
            self.draw_image(axes[1][col], f"{out['name']}: {label}", R, "result", limit)
        self.img_canvas.draw_idle()

        self.draw_histograms(
            [("I", self.image_array)] + [(f"B {out['name']}", out["B"]) for out in outputs],
            [(f"R {out['name']}", R) for out, (_, R) in zip(outputs, main_results)],
        )
        self.status_var.set(f"Compared all methods - R display range ±{limit:.0f}")

    # ------------------------------------------------------------------
    # Drawing: histograms
    # ------------------------------------------------------------------
    def draw_histograms(self, left, right):
        # left: original + background maps, right: corrected images; same bins and x-range everywhere
        self.last_histograms = (left, right)
        self.hist_figure.clear()
        axes = self.hist_figure.subplots(1, 2, sharex=True)
        for ax, series, title in [(axes[0], left, "Original I and background B"),
                                  (axes[1], right, "Corrected R")]:
            for name, array in series:
                counts, _ = np.histogram(array.ravel(), bins=HIST_EDGES)
                ax.stairs(counts, HIST_EDGES, label=name)
            ax.axvline(0, color="gray", linestyle=":", linewidth=1)
            ax.set_xlim(HIST_EDGES[0], HIST_EDGES[-1])
            ax.set_yscale("log" if self.log_var.get() else "linear")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Intensity")
            if series:
                ax.legend(fontsize=8, loc="upper right")
        axes[0].set_ylabel("Pixel count")
        self.hist_canvas.draw_idle()

    def on_log_toggled(self):
        if self.last_histograms is not None:
            self.draw_histograms(*self.last_histograms)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.image_array = np.array(Image.open(file_path).convert("L"))

        # Show only the original and its histogram until 'Process Image' is clicked
        self.img_figure.clear()
        self.draw_image(self.img_figure.add_subplot(111), "Original I", self.image_array, "image")
        self.img_canvas.draw_idle()
        self.draw_histograms([("I", self.image_array)], [])

        h, w = self.image_array.shape
        self.status_var.set(f"Loaded: {file_path} ({w}x{h}) - choose a method and click 'Process Image'")


if __name__ == "__main__":
    root = tk.Tk()
    app = BackgroundCorrectionGUI(root)
    root.mainloop()
