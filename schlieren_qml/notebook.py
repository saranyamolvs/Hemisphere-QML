"""Small Jupyter helpers that reuse the verified command-line pipeline.

Heavy processing runs in a child process of the active kernel interpreter. The
parent notebook keeps its inline plotting backend and does not hold the large
image cache open, which is important for reruns on Windows.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def resolve_notebook_config(config, project_dir):
    """Return a copy whose file paths resolve against the visible project root."""
    from .cli import _validate_config

    resolved = deepcopy(config)
    _validate_config(resolved)
    project = Path(project_dir).expanduser().resolve()
    for key in ("image_directory", "output_directory", "labels_csv"):
        if resolved.get(key) is not None:
            path = Path(resolved[key]).expanduser()
            resolved[key] = str((project/path).resolve() if not path.is_absolute() else path.resolve())
    image_dir, output_dir = (Path(resolved[key]) for key in ("image_directory", "output_directory"))
    if image_dir == output_dir or image_dir in output_dir.parents or output_dir in image_dir.parents:
        raise ValueError("Keep the output and image folders separate; neither may contain the other.")
    return resolved


def preview_notebook_roi(config):
    """Return a first-frame/crop figure and a bounded processing-size summary.

    The caller owns closing/displaying the returned Matplotlib figure. Axes on
    both panels use coordinates of the original sensor image.
    """
    from .data import list_images, read_image, preprocess_image
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    paths = list_images(config["image_directory"], config.get("pattern", "*"))
    selected = paths[config.get("start", 0)::config.get("stride", 1)]
    if config.get("limit", 200) is not None:
        selected = selected[:config.get("limit", 200)]
    if not selected:
        raise ValueError("No images remain after start, limit and stride selection.")
    raw = read_image(selected[0])
    factor = config.get("bin_factor", 1)
    processed = preprocess_image(raw, config.get("roi"), factor)
    height, width = raw.shape
    roi = config.get("roi") or [0, 0, width, height]
    x0, y0, x1, y1 = roi
    low, high = float(raw.min()), float(raw.max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].imshow(raw, cmap="gray", vmin=low, vmax=high, extent=(0, width, height, 0))
    axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#D55E00", linewidth=1.5))
    axes[0].set_title("First selected image and fixed ROI")
    axes[1].imshow(processed, cmap="gray", vmin=low, vmax=high, extent=(x0, x1, y1, y0))
    axes[1].set_title(f"Processed crop: {processed.shape[1]} × {processed.shape[0]} pixels")
    for ax in axes:
        ax.set(xlabel="Original sensor x (pixels)", ylabel="Original sensor y (pixels)")
    count = len(selected)
    summary = {
        "first_filename": selected[0].name, "last_filename": selected[-1].name,
        "selected_frames": count, "original_image_height_width": list(raw.shape),
        "processed_height_width": list(processed.shape), "roi_exclusive": list(roi),
        "cache_estimate_GB": count*processed.size*4/1e9,
        "first_to_last_span_ms": (count-1)*config.get("stride", 1)/config.get("fps", 51032)*1000,
        "timing_assumption": "Natural file order is complete and uniformly sampled; verify camera frame numbering.",
    }
    return fig, summary


def run_from_notebook(config, project_dir, overwrite=False, interpreter=None):
    """Run with the active kernel's Python and return the saved run manifest.

    Exit code 1 is accepted only for the pipeline's documented partial result
    with unavailable classification. Configuration/runtime failures raise an
    error with the tail of the child log. Interrupting the cell stops the child.
    """
    project = Path(project_dir).expanduser().resolve()
    if not (project/"schlieren_qml"/"cli.py").is_file():
        raise FileNotFoundError("PROJECT_DIR must point to the extracted Hemisphere-QML project folder.")
    resolved = resolve_notebook_config(config, project)
    output = Path(resolved["output_directory"])
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Results already exist at {output}. Change the output folder or enable the relevant notebook overwrite option to replace generated results.")
    config_folder = project/"work"/"notebook_configs"
    config_folder.mkdir(parents=True, exist_ok=True)
    config_path = config_folder/f"run_{uuid.uuid4().hex}.json"
    config_bytes = json.dumps(resolved, indent=2, allow_nan=False).encode("utf-8")
    config_path.write_bytes(config_bytes)
    command = [str(interpreter or sys.executable), "-u", "-m", "schlieren_qml", "run", str(config_path)]
    if overwrite:
        command.append("--overwrite")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    tail = deque(maxlen=20)
    process = subprocess.Popen(command, cwd=project, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line.rstrip())
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
    manifest_path = output/"run_manifest.json"
    if return_code not in (0, 1) or not manifest_path.is_file():
        raise RuntimeError(f"Processing failed (exit code {return_code}).\n"+"\n".join(tail))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != hashlib.sha256(config_bytes).hexdigest():
        raise RuntimeError("Saved results do not match this notebook run's configuration; inspect the child log.")
    if return_code == 1 and not (manifest.get("status") == "partial" and manifest.get("classification_status") == "unavailable"):
        raise RuntimeError("Processing did not finish successfully.\n"+"\n".join(tail))
    if return_code == 0 and manifest.get("status") != "complete":
        raise RuntimeError("The child process returned an inconsistent run status; inspect run_manifest.json.")
    if manifest.get("status") == "partial":
        print("POD results are available; classification could not be supported by these training data.")
    return manifest


def display_saved_figures(manifest, names=None, width=950):
    """Embed generated PNGs in notebook outputs without changing the backend."""
    from IPython.display import Image, Markdown, display

    wanted = set(names) if names is not None else None
    count = 0
    for entry in manifest.get("figures", []):
        path = Path(entry)
        if path.suffix.lower() == ".png" and (wanted is None or path.stem in wanted):
            if not path.is_file():
                raise FileNotFoundError(f"Saved figure is missing: {path}")
            display(Markdown(f"**{path.stem}**"))
            display(Image(filename=str(path), width=width))
            count += 1
    return count


def replot_spatial_modes(results_directory, *, vmax=None, percentile=None, n_modes=4, dpi=600):
    """Adjust a saved POD mode figure without reading images or refitting POD.

    Exports go into ``figures/mode_color_adjusted`` so the original figure and
    numerical results are preserved. Limits apply only to the displayed colors.
    """
    from .pod import load_pod
    from .figures import plot_spatial_modes

    results = Path(results_directory).expanduser().resolve()
    model = load_pod(results/"pod_model.npz")
    metadata = json.loads((results/"frames.json").read_text(encoding="utf-8"))
    saved_config = json.loads((results/"config_used.json").read_text(encoding="utf-8"))
    config = dict(saved_config.get("figures", {}))
    config.update(n_modes_plot=n_modes, dpi=dpi, bin_factor=metadata.get("bin_factor", 1),
                  mode_color_vmax=vmax, mode_color_percentile=percentile)
    destination = results/"figures"/"mode_color_adjusted"
    paths = plot_spatial_modes(model, tuple(metadata["image_shape"]), destination, config=config)
    scale = json.loads((destination/"04_spatial_modes_color_scale.json").read_text(encoding="utf-8"))
    return {"figures": paths, "color_scale": scale, "output_directory": str(destination)}
