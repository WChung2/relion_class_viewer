# relion_class_viewer

Interactive iteration viewer for RELION 2D / 3D classification jobs.

Loads every `run_itXXX_model.star` in a `Class2D/jobXXX/` or `Class3D/jobXXX/`
directory and lays the class images out in a grid. A slider scrubs through
iterations so you can watch how each class evolves, instead of opening one
star file at a time in RELION's display dialog.

Built on PyQt6 + pyqtgraph. The matplotlib prototype was retired because Agg
rasterization of a 50-panel grid is too slow to be usable on real Class2D
jobs.

## Features

- One MRC slice per class, displayed in a `--rows × --cols` grid.
  - **Class3D**: central Z slice of each volume (matches
    `relion_display`'s `getSlice(ZSIZE/2)`).
  - **Class2D**: stack frame indexed via RELION's `NNNNNN@path/file.mrcs`
    reference notation.
- Per-panel title `Class N: P.PP %` showing `rlnClassDistribution`.
- Live contrast / tone-map text fields:
  - **`percentile`**: clip display range to `[P, 100-P]` percentile (robust
    to outliers). Default `2`. Higher `P` = more aggressive clipping = more
    saturation. `0` disables clipping (raw min/max).
  - **`softness`**: optional S-curve tone map on the display LUT (`0` =
    linear, `0.3–1.0` = gentle midtone expansion with softened highlights
    and shadows). Independent from the percentile clip.
- **Sort button** cycling **orig → high→low → low→high**. Sort policy
  re-applies on every iteration as you scrub, so the panels always show the
  current iteration's top classes.
- Iteration slider, plus **←/→** arrow-key navigation.
- **Save PNG** button (snapshots the panel grid + controls).
- Preload at startup with a `tqdm` progress bar; per-slice min/max are
  precomputed so the no-clipping mode is constant-time per redraw.
- **Block-mean + nearest-neighbor resize** at load time so every iteration's
  panel ends up at the same pixel grid. The target size is computed from the
  *last* iteration's image size divided by `--downsample F` (default 3×).
  Important because RELION writes each iteration at `_rlnCurrentImageSize`,
  which grows during refinement — without this every iteration would have a
  different panel shape.

## Installation

**Requires Python 3.9+** (uses PEP 585 generic type hints like `list[…]`).

### 1. Get the code

```bash
git clone https://github.com/WChung2/relion_class_viewer.git
cd relion_class_viewer
```

To grab updates later: `git pull`.

### 2. Install dependencies

PyQt6 over X11 forwarding needs `libxcb-cursor.so.0` (Qt ≥ 6.5 requirement),
which is missing on many Linux distros. The conda-forge path is the most
reliable on a remote GPU server:

```bash
conda install -c conda-forge \
    numpy mrcfile tqdm pyqt pyqtgraph xcb-util-cursor
```

If you'd rather use pip, you can — but on Linux you also need
`libxcb-cursor0` from your distro (Debian/Ubuntu: `sudo apt install
libxcb-cursor0`):

```bash
pip install -r requirements.txt
```

## Running

Needs an X11 display. From your laptop:

```bash
ssh -XY user@<gpu-server>          # X11 forwarding (-X untrusted + -Y trusted fallback)
echo $DISPLAY                       # should print something like localhost:10.0
xeyes                               # quick sanity check
```

Important: **VS Code remote-SSH terminals do not forward X by default.** Use
a plain `ssh -X` terminal, or set `ForwardX11 yes` in your laptop's
`~/.ssh/config` for that host and restart the VS Code remote server.

Then invoke the script with `python` + the path to `relion_class_viewer.py`.
Two common patterns:

```bash
# Pattern A: cd into the cloned repo, then call by bare filename.
cd ~/relion_class_viewer            # or wherever you cloned it
python relion_class_viewer.py <job_dir> --rows R --cols C [options]

# Pattern B: from anywhere, by absolute path.
python ~/relion_class_viewer/relion_class_viewer.py <job_dir> --rows R --cols C [options]
```

`<job_dir>` is the RELION job folder — usually an **absolute path** because
RELION jobs almost never live in the same tree as this script. Tab-completion
works in either pattern.

If you're on Python 3.9+ the script runs immediately; older Pythons exit with
a clear `requires Python 3.9 or newer` message instead of a stack trace.

### Examples

Class3D job, 6 classes in a 2×3 grid:

```bash
python relion_class_viewer.py /path/to/Class3D/job020 --rows 2 --cols 3
```

Class2D job, 50 classes in a 5×10 grid:

```bash
python relion_class_viewer.py /path/to/Class2D/job001 --rows 5 --cols 10
```

Once the window is up, tweak `percentile` (e.g. type `1` or `3`) and
optionally `softness` (e.g. `0.5`) to taste.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--rows R --cols C` | required | Grid layout. If `R*C < n_classes`, the first `R*C` are shown. With sort active, this is the top-`R*C` by distribution. |
| `--percentile P` | `2` | Display range = `[P, 100-P]` percentile. Robust to outliers. Higher `P` clips more aggressively (more saturation). `0` = raw min/max. |
| `--softness K` | `0` | **Display-only S-curve tone map** applied via the LUT. `0` = linear. `0.3–1.0` = gentle midtone expansion with softened highlights and shadows. Adjustable live in the GUI. Does not alter the underlying image data. |
| `--downsample F` | `3` | Integer downsample factor. `1` = no downsampling, `2` = half, `3` = third. Target panel size = `(last_iteration_image_size / F)`; every iteration is resized to that size so panels stay on a consistent pixel grid even when RELION extends resolution across iterations. |
| `--no-preload` | off | Skip up-front loading; slices load lazily on first view. |

## How class images are located

For each `run_itXXX_model.star` the viewer reads the `data_model_classes`
block and pulls the `_rlnReferenceImage` + `_rlnClassDistribution` columns.
The reference path is resolved against (in order):

1. Absolute path
2. The project root (parent of the `Class3D/` or `Class2D/` directory)
3. The job directory itself
4. The current working directory

Paths written by RELION are project-root-relative, so #2 normally hits.

## Display semantics

- **Slice taken from each Class3D volume:** `vol[nz//2, :, :]` (the central
  Z slice) — same as `img.getSlice(ZSIZE(img)/2, slice)` in
  `relion/src/displayer.cpp`.
- **Class2D stack indexing:** the RELION `idx@path.mrcs` notation in
  `_rlnReferenceImage` is mapped to `stack[idx-1, :, :]` (1-based RELION
  index → 0-based NumPy slice).
- **Contrast mapping:** display range is `[P, 100-P]` percentile of the
  panel's pixel values (computed per panel, per redraw). `P = 0` falls back
  to the raw `(min, max)` of the panel. The optional `--softness` tone map
  is then applied as an S-curve on the resulting LUT.
- **Class label:** each panel title shows the original class index and
  `rlnClassDistribution × 100`.

## Performance notes

- All redraws push image data through Qt's `QPainter` raster path. The
  framerate ceiling is mainly the X11 forwarding link.
- `percentile` mode calls `np.percentile` on every redraw. On the default
  256² panels it's ~1-2 ms per panel, so 50 panels ≈ 50-100 ms per slider
  tick. Setting `percentile` to `0` (raw min/max) skips this entirely.
- Increase `--downsample` for faster redraws: `4` is roughly 2× faster than
  the default `3`, `5` is ~3× faster, with progressively visible-but-
  tolerable quality loss. For a 720² source, factor 3 → 240², factor 5 →
  144², factor 8 → 90².
- If even `--downsample 6` feels sluggish, your X11 link is the bottleneck,
  not the renderer — try a remote-desktop alternative like NoMachine or
  TurboVNC instead of raw `ssh -X`.

## Files

```
relion_class_viewer/
├── README.md
├── requirements.txt
└── relion_class_viewer.py     # single-file CLI; star parsing + Qt GUI
```
