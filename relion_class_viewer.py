#!/usr/bin/env python3
"""
RELION 2D/3D Classification iteration viewer.

Loads every run_itXXX_model.star in a Class2D or Class3D jobXXX/ directory
and shows each class in a grid. For Class3D the central Z slice of each
class volume is shown (matching relion_display). For Class2D the 2D class
image is read directly from the run_itXXX_classes.mrcs stack via the RELION
`index@path` reference notation. A slider scrubs through iterations.

GUI is Qt (PyQt6) + pyqtgraph. The raster QPainter renderer underneath is
much faster than a matplotlib equivalent for repeated image updates, which
is what slider scrubbing on a 50-panel grid does.

Note: there is a --opengl flag but it's largely cosmetic. pyqtgraph's
useOpenGL=True only accelerates line plots, NOT ImageItem (what we use). It
also breaks frequently over X11 forwarding due to indirect-GLX limitations
(cascade of "QPainter not active" errors). Leave it off unless you know
why you need it.

Usage:
    python relion_class_viewer.py Class3D/job020 --rows 2 --cols 4
    python relion_class_viewer.py Class2D/job001 --rows 5 --cols 10
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets


# --- shared parsing / loading helpers --------------------------------------

IT_RE = re.compile(r"run_it(\d+)_model\.star$")


@dataclass
class SliceData:
    """One panel: the (downsampled) image plus precomputed display stats.

    Caching mean/std/min/max here lets contrast_range avoid recomputing
    np.mean/std on every slider tick -- the dominant cost on large grids.
    """
    img: np.ndarray
    mean: float
    std: float
    min: float
    max: float


def parse_model_classes(star_path: Path) -> list[tuple[str, float]]:
    """Return [(reference_image_path, class_distribution), ...] from a model.star.

    Minimal parser: walks data blocks, finds `data_model_classes`, reads its
    `loop_` header to locate the rlnReferenceImage and rlnClassDistribution
    columns, then returns one tuple per data row.
    """
    with open(star_path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    in_classes = False
    columns: dict[str, int] = {}
    rows: list[tuple[str, float]] = []

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("data_"):
            in_classes = line == "data_model_classes"
            columns = {}
            i += 1
            continue

        if in_classes and line == "loop_":
            i += 1
            col_idx = 0
            while i < len(lines):
                lstr = lines[i].strip()
                if lstr.startswith("_"):
                    name = lstr.split()[0][1:]
                    columns[name] = col_idx
                    col_idx += 1
                    i += 1
                elif lstr == "":
                    i += 1
                else:
                    break

            ref_col = columns.get("rlnReferenceImage")
            dist_col = columns.get("rlnClassDistribution")
            if ref_col is None or dist_col is None:
                raise ValueError(
                    f"{star_path}: data_model_classes is missing rlnReferenceImage "
                    f"or rlnClassDistribution"
                )

            while i < len(lines):
                lstr = lines[i].strip()
                if lstr == "" or lstr.startswith("data_"):
                    break
                parts = lstr.split()
                if len(parts) > max(ref_col, dist_col):
                    rows.append((parts[ref_col], float(parts[dist_col])))
                i += 1
            in_classes = False
            continue

        i += 1

    if not rows:
        raise ValueError(f"{star_path}: no class rows found")
    return rows


def parse_reference_image(ref: str) -> tuple[str, int | None]:
    """Split a RELION rlnReferenceImage into (path, stack_index_0based).

    Class2D refs are stack frames written as `NNNNNN@path/to/run_itXXX_classes.mrcs`
    (1-based). Class3D refs are plain paths to a 3D MRC volume. We return
    stack_index = None for the plain case.
    """
    if "@" in ref:
        idx_str, path_str = ref.split("@", 1)
        return path_str, int(idx_str) - 1
    return ref, None


def resolve_mrc(path_str: str, job_dir: Path) -> Path:
    """Resolve an MRC file path.

    RELION writes paths relative to the project root (the directory that
    contains Class3D/ or Class2D/). We try, in order: absolute,
    project-root-relative, job-dir-relative, then cwd-relative.
    """
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p

    project_root = job_dir.parent.parent  # Class3D/jobXXX -> .
    candidates = [
        project_root / path_str,
        job_dir / path_str,
        Path.cwd() / path_str,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not locate MRC '{path_str}'. Tried: {[str(c) for c in candidates]}"
    )


def load_class_image(ref: str, job_dir: Path) -> np.ndarray:
    """Return a 2D float32 image for one class reference.

    - Class3D ref (`path.mrc`): take the central Z slice of the volume,
      matching relion_display's `img.getSlice(ZSIZE(img)/2, slice)`.
    - Class2D ref (`idx@path.mrcs`): take the indexed frame from the stack.
    """
    path_str, stack_idx = parse_reference_image(ref)
    mrc_path = resolve_mrc(path_str, job_dir)
    with mrcfile.mmap(mrc_path, mode="r", permissive=True) as m:
        data = m.data
        if stack_idx is not None:
            if data.ndim != 3:
                raise ValueError(
                    f"{mrc_path}: expected stack (3D ndarray) for ref '{ref}', got ndim={data.ndim}"
                )
            return np.array(data[stack_idx], dtype=np.float32)
        if data.ndim == 3:
            return np.array(data[data.shape[0] // 2, :, :], dtype=np.float32)
        if data.ndim == 2:
            return np.array(data, dtype=np.float32)
        raise ValueError(f"{mrc_path}: unexpected ndim={data.ndim}")


def contrast_range(s: SliceData, sigma: float, percentile: float) -> tuple[float, float]:
    """Compute display vmin/vmax for one slice.

    Modes (checked in order):
      - percentile > 0: clip at [P, 100-P] percentile -- robust to outliers.
                        Only mode that still touches the pixels at redraw time
                        (np.percentile needs to sort), so this is the slowest.
      - sigma > 0     : mean +/- sigma*std (matches relion_display
                        getImageContrast exactly). Uses cached mean/std.
      - otherwise     : raw min/max (matches relion_display sigma_contrast=0).
                        Uses cached min/max.
    """
    if percentile > 0:
        lo, hi = np.percentile(s.img, [percentile, 100.0 - percentile])
        return float(lo), float(hi)
    if sigma > 0:
        return s.mean - sigma * s.std, s.mean + sigma * s.std
    return s.min, s.max


def downsample_2d(img: np.ndarray, target: int) -> np.ndarray:
    """Resize a 2D image to exactly (target, target). NumPy only.

    RELION writes each iteration's class MRC at `_rlnCurrentImageSize`, which
    grows as RELION extends resolution during refinement. So iter000 might
    be 60x60 and iter025 720x720 in the same job. To keep every panel at a
    consistent pixel grid across iterations, this function always produces a
    target-by-target array regardless of the input shape:

      - If much larger than target: block-mean by an integer factor f
        (anti-aliased), then nearest-neighbor resize the residual.
      - If smaller or close in size: nearest-neighbor resize directly
        (effectively upsamples small early-iteration MRCs).

    target == 0 disables resizing entirely (panels will be heterogeneous).
    """
    if target <= 0:
        return img
    h, w = img.shape
    if h == target and w == target:
        return img if img.dtype == np.float32 else img.astype(np.float32)
    if max(h, w) > 2 * target:
        f = max(h, w) // target
        new_h, new_w = h // f, w // f
        cropped = img[: new_h * f, : new_w * f]
        img = cropped.reshape(new_h, f, new_w, f).mean(axis=(1, 3))
        h, w = img.shape
    if h == target and w == target:
        return img.astype(np.float32)
    ys = np.linspace(0, h - 1, target).astype(np.int32)
    xs = np.linspace(0, w - 1, target).astype(np.int32)
    return img[np.ix_(ys, xs)].astype(np.float32)


def make_tone_lut(softness: float, n: int = 256) -> np.ndarray:
    """8-bit grayscale lookup table with an optional S-curve tone map.

    The image's [vmin, vmax] range is mapped linearly to indices [0, n-1]
    by pyqtgraph; this LUT then defines what each index renders as. The
    LUT shape is what gives the display its 'feel':

      softness <= 0 : linear (y = x).
      softness  > 0 : centered tanh S-curve. Steepens midtones (where the
                      protein density lives) and softens highlights and
                      shadows so the brightest spots don't immediately
                      saturate to white. Typical useful range: 0.3 - 1.5.

    Note: this changes the perceived contrast of the *display*, it does
    not modify the underlying image data. relion_display itself does NOT
    apply tone mapping -- this is an extension for nicer cryo-EM display.
    """
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if softness <= 0:
        y = x
    else:
        k = 1.0 + softness * 4.0  # tanh steepness
        y_raw = np.tanh(k * (x - 0.5))
        y_min = np.tanh(-k * 0.5)
        y_max = np.tanh(k * 0.5)
        y = (y_raw - y_min) / (y_max - y_min)
    g = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g, np.full_like(g, 255)], axis=-1)


def make_slice_data(img: np.ndarray, target: int) -> SliceData:
    """Downsample if needed and precompute display statistics."""
    small = downsample_2d(img, target)
    return SliceData(
        img=small,
        mean=float(small.mean()),
        std=float(small.std()),
        min=float(small.min()),
        max=float(small.max()),
    )


class ClassViewer(QtWidgets.QMainWindow):
    def __init__(
        self,
        job_dir: Path,
        rows: int,
        cols: int,
        sigma: float,
        percentile: float,
        downsample: int,
        softness: float,
        preload: bool,
    ) -> None:
        super().__init__()
        self.job_dir = job_dir
        self.rows = rows
        self.cols = cols
        self.factor = max(1, int(downsample))

        self._discover_and_parse()
        # Resolve the integer factor to an actual target pixel size by reading
        # the LAST iteration's first class. RELION extends resolution across
        # iterations, so the last iteration usually has the largest MRC; using
        # it as the reference means every iteration gets resized to the SAME
        # pixel grid (downsampled if late, upsampled if early). Without this
        # the panel size would change as you scrub the slider.
        last_ref, _ = self.per_iter_rows[-1][0]
        ref_img = load_class_image(last_ref, self.job_dir)
        ref_size = max(ref_img.shape)
        self.target_size = max(1, ref_size // self.factor)
        print(f"Downsample factor: {self.factor}x  ->  panels are "
              f"{self.target_size}x{self.target_size} (reference size {ref_size})")

        self._init_cache(preload)

        # Mutable display state. sort_mode is a *policy* (orig / desc / asc);
        # the actual class order is recomputed from each iteration's
        # distributions on every redraw, so as you scrub the slider the panels
        # automatically re-rank to stay sorted within that iteration.
        self.state = {
            "sigma": float(sigma),
            "pct": float(percentile),
            "softness": max(0.0, float(softness)),
            "sort_mode": "orig",
        }

        self._build_ui()
        self._refresh_mode_label()
        self._populate_initial()
        self.setWindowTitle(f"relion_class_viewer — {self.job_dir.name}")
        self.resize(220 * self.cols + 80, 220 * self.rows + 140)

    # ---- model loading ----------------------------------------------------

    def _discover_and_parse(self) -> None:
        star_files = sorted(
            (p for p in self.job_dir.glob("run_it*_model.star") if IT_RE.search(p.name)),
            key=lambda p: int(IT_RE.search(p.name).group(1)),
        )
        if not star_files:
            raise FileNotFoundError(
                f"No run_it*_model.star files found in {self.job_dir}"
            )
        self.star_files = star_files
        self.iters = [int(IT_RE.search(p.name).group(1)) for p in star_files]
        print(f"Found {len(star_files)} iterations: it{self.iters[0]:03d} .. it{self.iters[-1]:03d}")
        self.per_iter_rows = [parse_model_classes(p) for p in star_files]
        n_classes_each = {len(r) for r in self.per_iter_rows}
        if len(n_classes_each) > 1:
            print(f"Warning: class count varies across iterations: {sorted(n_classes_each)}",
                  file=sys.stderr)
        n_classes = max(n_classes_each)
        print(f"Classes per iteration: {n_classes}")
        grid_n = self.rows * self.cols
        if grid_n < n_classes:
            print(f"Warning: grid is {self.rows}x{self.cols}={grid_n} but job has "
                  f"{n_classes} classes; showing first {grid_n}.", file=sys.stderr)
            self.n_shown = grid_n
        else:
            self.n_shown = n_classes
        self.n_classes = n_classes

    def _init_cache(self, preload: bool) -> None:
        self.slice_cache: dict[tuple[int, int], SliceData] = {}
        if not preload:
            return
        try:
            from tqdm import tqdm
            it = tqdm(
                ((i, c) for i in range(len(self.star_files))
                 for c in range(len(self.per_iter_rows[i]))),
                total=sum(len(r) for r in self.per_iter_rows),
                desc="Preloading slices",
                unit="img",
            )
        except ImportError:
            total = sum(len(r) for r in self.per_iter_rows)
            print(f"Preloading {total} slices...")
            it = ((i, c) for i in range(len(self.star_files))
                  for c in range(len(self.per_iter_rows[i])))
        nbytes = 0
        first_shape: tuple[int, int] | None = None
        for i, c in it:
            s = self._get_slice(i, c)
            nbytes += s.img.nbytes
            if first_shape is None:
                first_shape = s.img.shape
        print(f"Cached {len(self.slice_cache)} slices "
              f"({nbytes / 1024 / 1024:.1f} MB, panel shape {first_shape}).")

    def _get_slice(self, it_idx: int, cls_idx: int) -> SliceData:
        key = (it_idx, cls_idx)
        if key not in self.slice_cache:
            ref, _ = self.per_iter_rows[it_idx][cls_idx]
            self.slice_cache[key] = make_slice_data(
                load_class_image(ref, self.job_dir), self.target_size
            )
        return self.slice_cache[key]

    # ---- UI construction --------------------------------------------------

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        # Grid of panels (one PlotItem per class slot).
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.layout.setSpacing(2)
        v.addWidget(self.glw, stretch=1)

        # Tone-mapped grayscale LUT. softness=0 is plain linear gray, > 0 adds
        # a tanh S-curve via make_tone_lut.
        self._lut = make_tone_lut(self.state["softness"])

        self.plot_items: list[pg.PlotItem] = []
        self.image_items: list[pg.ImageItem] = []
        # ViewBox bounds we re-apply on every redraw, so titles changing
        # width cannot cause the ViewBox to re-fit / shift the image.
        self._view_extent = int(self.target_size)
        for r in range(self.rows):
            for c in range(self.cols):
                plot = self.glw.addPlot(row=r, col=c)
                plot.hideAxis("left")
                plot.hideAxis("bottom")
                plot.setMouseEnabled(x=False, y=False)
                plot.setMenuEnabled(False)
                plot.setAspectLocked(True)
                plot.invertY(True)  # match imshow(origin='upper'); raw MRC orientation
                img = pg.ImageItem(axisOrder="row-major")
                img.setLookupTable(self._lut)
                plot.addItem(img)
                plot.setTitle("")
                # Lock the visible range so changing titles can't shift things.
                plot.setRange(
                    xRange=(0, self._view_extent),
                    yRange=(0, self._view_extent),
                    padding=0,
                )
                plot.getViewBox().setDefaultPadding(0)
                self.plot_items.append(plot)
                self.image_items.append(img)

        # Control row: sigma, percentile, sort, save.
        ctrls = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(ctrls)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QtWidgets.QLabel("sigma_contrast"))
        self.sigma_edit = QtWidgets.QLineEdit(f"{self.state['sigma']:g}")
        self.sigma_edit.setFixedWidth(70)
        self.sigma_edit.setValidator(QtGui.QDoubleValidator(0.0, 1e9, 4))
        self.sigma_edit.setToolTip(
            "Clip display to mean ± σ·std (matches relion_display).\n"
            "0 (or empty) disables. Setting this clears 'percentile' \n"
            "since only one mode is active at a time."
        )
        # editingFinished fires on Enter or focus-loss; returnPressed is a
        # backup so Enter always works even if focus tracking misbehaves.
        self.sigma_edit.editingFinished.connect(self._on_sigma)
        self.sigma_edit.returnPressed.connect(self._on_sigma)
        h.addWidget(self.sigma_edit)

        h.addSpacing(12)
        h.addWidget(QtWidgets.QLabel("percentile"))
        self.pct_edit = QtWidgets.QLineEdit(f"{self.state['pct']:g}")
        self.pct_edit.setFixedWidth(70)
        self.pct_edit.setValidator(QtGui.QDoubleValidator(0.0, 49.999, 4))
        self.pct_edit.setToolTip(
            "Clip display to [P, 100-P] percentile (robust to outliers).\n"
            "0 (or empty) disables. Setting this clears 'sigma_contrast'.\n"
            "Higher P = more aggressive clipping = more saturation."
        )
        self.pct_edit.editingFinished.connect(self._on_pct)
        self.pct_edit.returnPressed.connect(self._on_pct)
        h.addWidget(self.pct_edit)

        h.addSpacing(12)
        h.addWidget(QtWidgets.QLabel("softness"))
        self.softness_edit = QtWidgets.QLineEdit(f"{self.state['softness']:g}")
        self.softness_edit.setFixedWidth(70)
        self.softness_edit.setValidator(QtGui.QDoubleValidator(0.0, 10.0, 4))
        self.softness_edit.setToolTip(
            "Display-only S-curve tone map. 0 = linear (matches relion_display).\n"
            "0.3-1.0 = gentle midtone expansion and softened highlights/shadows,\n"
            "giving cryo-EM slices a less harshly saturated look.\n"
            "Higher = more pronounced S-curve."
        )
        self.softness_edit.editingFinished.connect(self._on_softness)
        self.softness_edit.returnPressed.connect(self._on_softness)
        h.addWidget(self.softness_edit)

        h.addSpacing(12)
        self.mode_label = QtWidgets.QLabel("")
        self.mode_label.setStyleSheet("color: #888;")
        h.addWidget(self.mode_label)

        h.addSpacing(12)
        self.sort_btn = QtWidgets.QPushButton("sort: orig")
        self.sort_btn.clicked.connect(self._on_sort)
        h.addWidget(self.sort_btn)

        h.addSpacing(12)
        self.save_btn = QtWidgets.QPushButton("Save PNG…")
        self.save_btn.clicked.connect(self._on_save)
        h.addWidget(self.save_btn)
        h.addStretch(1)
        v.addWidget(ctrls)

        # Slider row.
        srow = QtWidgets.QWidget()
        sh = QtWidgets.QHBoxLayout(srow)
        sh.setContentsMargins(0, 0, 0, 0)
        sh.addWidget(QtWidgets.QLabel("iter"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.iters) - 1)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.valueChanged.connect(self._on_slider)
        sh.addWidget(self.slider, stretch=1)
        self.slider_label = QtWidgets.QLabel(f"it{self.iters[0]:03d}")
        self.slider_label.setMinimumWidth(60)
        sh.addWidget(self.slider_label)
        v.addWidget(srow)

        # Keyboard nav. (QShortcut moved from QtWidgets to QtGui in PyQt6.)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self,
                        activated=lambda: self.slider.setValue(self.slider.value() + 1))
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self,
                        activated=lambda: self.slider.setValue(self.slider.value() - 1))

    def _populate_initial(self) -> None:
        # _redraw_current already does the right thing for whatever iteration
        # the slider currently points at (0 at startup).
        self._redraw_current()

    # ---- redraw / handlers ------------------------------------------------

    def _current_order(self, it_idx: int) -> list[int]:
        """Class indices to display, in order, for this iteration."""
        rows = self.per_iter_rows[it_idx]
        n_total = len(rows)
        mode = self.state["sort_mode"]
        if mode == "orig":
            full = list(range(n_total))
        else:
            full = sorted(
                range(n_total),
                key=lambda k: rows[k][1],
                reverse=(mode == "desc"),
            )
        # Pad to n_shown with -1 sentinels so the panel-hiding branch still
        # works when n_total < n_shown.
        return full[: self.n_shown] + [-1] * (self.n_shown - len(full))

    def _redraw_current(self) -> None:
        it_idx = self.slider.value()
        order = self._current_order(it_idx)
        rows = self.per_iter_rows[it_idx]
        for k in range(self.rows * self.cols):
            if k >= self.n_shown:
                self.plot_items[k].hide()
                continue
            actual = order[k]
            if actual < 0 or actual >= len(rows):
                self.plot_items[k].hide()
                continue
            self.plot_items[k].show()
            s = self._get_slice(it_idx, actual)
            vmin, vmax = contrast_range(s, self.state["sigma"], self.state["pct"])
            self.image_items[k].setImage(s.img, autoLevels=False, levels=(vmin, vmax))
            _, dist = rows[actual]
            # Monospace + padded fields keep the title's pixel width constant
            # across iterations. Without this, varying digit counts would
            # change the title width, which the GraphicsLayout would absorb
            # by shifting the ViewBox by a few pixels each tick -- looking
            # like the image is "popping" between iterations.
            self.plot_items[k].setTitle(
                f'<span style="font-family: monospace; font-size: 9pt; '
                f'white-space: pre;">'
                f'Class {actual+1:>3d}: {dist*100:>6.2f} %</span>'
            )
            # Re-assert the locked view in case anything inside Qt nudged it.
            self.plot_items[k].setRange(
                xRange=(0, self._view_extent),
                yRange=(0, self._view_extent),
                padding=0,
            )
        self.slider_label.setText(f"it{self.iters[it_idx]:03d}")
        self.setWindowTitle(
            f"relion_class_viewer — {self.job_dir.name} — it{self.iters[it_idx]:03d}"
        )

    def _on_slider(self, _v: int) -> None:
        self._redraw_current()

    def _on_sigma(self) -> None:
        txt = self.sigma_edit.text().strip()
        try:
            v = 0.0 if txt == "" else float(txt)
        except ValueError:
            return
        if v < 0:
            return
        self.state["sigma"] = v
        # Make modes mutually exclusive so it's obvious which is active.
        if v > 0 and self.state["pct"] > 0:
            self.state["pct"] = 0.0
            self.pct_edit.blockSignals(True)
            self.pct_edit.setText("0")
            self.pct_edit.blockSignals(False)
        self._refresh_mode_label()
        self._redraw_current()

    def _on_pct(self) -> None:
        txt = self.pct_edit.text().strip()
        try:
            v = 0.0 if txt == "" else float(txt)
        except ValueError:
            return
        if v < 0 or v >= 50:
            return
        self.state["pct"] = v
        if v > 0 and self.state["sigma"] > 0:
            self.state["sigma"] = 0.0
            self.sigma_edit.blockSignals(True)
            self.sigma_edit.setText("0")
            self.sigma_edit.blockSignals(False)
        self._refresh_mode_label()
        self._redraw_current()

    def _on_softness(self) -> None:
        txt = self.softness_edit.text().strip()
        try:
            v = 0.0 if txt == "" else float(txt)
        except ValueError:
            return
        if v < 0:
            return
        self.state["softness"] = v
        # Regenerate the LUT and push it to every ImageItem.
        self._lut = make_tone_lut(v)
        for item in self.image_items:
            item.setLookupTable(self._lut)
        self._refresh_mode_label()
        self._redraw_current()

    def _refresh_mode_label(self) -> None:
        if self.state["pct"] > 0:
            clip = f"active: percentile = {self.state['pct']:g} (clip [{self.state['pct']:g}, {100-self.state['pct']:g}])"
        elif self.state["sigma"] > 0:
            clip = f"active: sigma_contrast = {self.state['sigma']:g} (clip mean ± {self.state['sigma']:g}·std)"
        else:
            clip = "active: raw min/max"
        if self.state["softness"] > 0:
            clip = f"{clip}, softness = {self.state['softness']:g}"
        else:
            clip = f"{clip}, softness = 0 (linear)"
        self.mode_label.setText(clip)

    def _on_sort(self) -> None:
        next_mode = {"orig": "desc", "desc": "asc", "asc": "orig"}[self.state["sort_mode"]]
        self.state["sort_mode"] = next_mode
        self.sort_btn.setText({
            "orig": "sort: orig",
            "desc": "sort: high→low",
            "asc": "sort: low→high",
        }[next_mode])
        self._redraw_current()

    def _on_save(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save figure as PNG",
            f"{self.job_dir.name}_it{self.iters[self.slider.value()]:03d}.png",
            "PNG (*.png)",
        )
        if not path:
            return
        # Grab the entire central widget so the title bar/controls are excluded.
        pixmap = self.centralWidget().grab()
        pixmap.save(path, "PNG")
        print(f"Saved {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("job_dir", type=Path, help="Path to RELION Class2D/3D jobXXX/ directory")
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--sigma-contrast", type=float, default=3.0,
                    help="mean +/- sigma*std clipping (relion_display semantics; 0 = raw min/max).")
    ap.add_argument("--percentile", type=float, default=0.0,
                    help="Clip at [P, 100-P] percentile. 0 = off. Overrides sigma when > 0.")
    ap.add_argument("--downsample", type=int, default=3,
                    help="Integer downsample factor (1, 2, 3, ...). 1 = no downsampling, "
                         "3 = each dim shrinks by 3x (e.g. 720->240). Target panel size is "
                         "computed from the LAST iteration's image size, so every iteration "
                         "lands on the same pixel grid even when RELION extends resolution "
                         "across iterations. Larger factor = faster redraws, less detail.")
    ap.add_argument("--softness", type=float, default=0.0,
                    help="Optional S-curve tone map applied to the display LUT. 0 = linear "
                         "(matches relion_display). 0.3-1.0 = gentle midtone expansion and "
                         "softened highlights/shadows for less harsh saturation on cryo-EM "
                         "class slices. Does NOT alter the underlying image data.")
    ap.add_argument("--no-preload", action="store_true")
    ap.add_argument("--opengl", action="store_true",
                    help="Switch pyqtgraph's GraphicsView viewport to QOpenGLWidget. "
                         "Does NOT accelerate ImageItem (only line plots), and frequently "
                         "produces 'QPainter not active' errors over X11 forwarding because "
                         "indirect GLX cannot give Qt a usable GL context. Leave off.")
    args = ap.parse_args()

    job_dir = args.job_dir.resolve()
    if not job_dir.is_dir():
        print(f"Error: {job_dir} is not a directory", file=sys.stderr)
        return 2

    # Configure pyqtgraph BEFORE creating the QApplication / widgets.
    pg.setConfigOptions(
        antialias=False,
        imageAxisOrder="row-major",
        useOpenGL=args.opengl,
    )

    app = QtWidgets.QApplication(sys.argv)
    w = ClassViewer(
        job_dir=job_dir,
        rows=args.rows,
        cols=args.cols,
        sigma=args.sigma_contrast,
        percentile=args.percentile,
        downsample=args.downsample,
        softness=args.softness,
        preload=not args.no_preload,
    )
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
