"""Static, publication-oriented figures for measured schlieren sequences.

All plots describe image intensity, not kinetic energy, pressure, or density.
PDF and SVG preserve vector text; raster image panels retain their native
spatial information. No experimental data are synthesized in this module.
"""

from __future__ import annotations

import csv
import json
from numbers import Real
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.signal import welch


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "^", "D", "v", "P", "X"]
STYLE = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _save(fig, directory: Path, name: str, dpi: int) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for extension in ("pdf", "svg", "png"):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
            paths.append(str(path))
    finally:
        plt.close(fig)
    return paths


def _csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _extent(shape, config):
    h, w = shape
    factor = _positive_integer(config.get("bin_factor", 1), "bin_factor")
    scale = config.get("pixel_size_mm")
    if scale is None:
        scale = factor
        return [0, w*scale, h*scale, 0], "sensor pixels"
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("pixel_size_mm must be a positive finite number.")
    # Pixel calibration refers to the original image; binning changes pitch.
    scale *= factor
    return [0, w * scale, h * scale, 0], "mm"


def _image(ax, data, extent, unit, *, title=None, cmap="gray", vmin=None, vmax=None):
    handle = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest")
    display_unit = "sensor px" if unit == "sensor pixels" else unit
    ax.set_xlabel(f"x ({display_unit})")
    ax.set_ylabel(f"y ({display_unit})")
    if title:
        ax.set_title(title)
    return handle


def _stream_stats(X, batch_size=64):
    """Numerically stable population mean and RMS image fluctuations."""
    n_seen = 0
    mean = np.zeros(X.shape[1], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for start in range(0, len(X), batch_size):
        batch = np.asarray(X[start:start + batch_size], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("Image matrix contains non-finite intensity values.")
        n = len(batch)
        batch_mean = batch.mean(axis=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(axis=0)
        delta = batch_mean - mean
        total = n_seen + n
        m2 += batch_m2 + delta * delta * n_seen * n / total
        mean += delta * n / total
        n_seen = total
    return mean, np.sqrt(np.maximum(m2 / n_seen, 0))


def plot_roi(image, roi, output_dir, config=None) -> list[str]:
    """Show a user-specified exclusive crop on the unprocessed sensor image."""
    config = config or {}
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0 or not np.isfinite(image).all():
        raise ValueError("plot_roi expects a nonempty, finite two-dimensional grayscale image.")
    h, w = image.shape
    dpi = _positive_integer(config.get("dpi", config.get("figure_dpi", 600)), "dpi")
    if roi is not None and (len(roi) != 4 or any(isinstance(v, (bool, np.bool_)) or
                                                not isinstance(v, (int, np.integer)) for v in roi)):
        raise ValueError("ROI must contain four integer bounds [x0,y0,x1,y1].")
    x0, y0, x1, y1 = (0, 0, w, h) if roi is None else map(int, roi)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("ROI must lie inside the sensor image.")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
        ax.imshow(image, cmap="gray", interpolation="nearest", extent=[0, w, h, 0])
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                               edgecolor=COLORS[1], linewidth=1.5))
        ax.set(xlabel="Sensor image x (pixels)", ylabel="Sensor image y (pixels)",
               title=f"Selected region: x=[{x0}, {x1}), y=[{y0}, {y1})")
        return _save(fig, Path(output_dir), "01_selected_region", dpi)


def _mode_color_options(config):
    """Validate display-only color limits before rendering or writing files."""
    limit = config.get("mode_color_vmax")
    percentile = config.get("mode_color_percentile")
    for key, value in (("mode_color_vmax", limit), ("mode_color_percentile", percentile)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                                  or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be null or a positive finite number.")
    if percentile is not None and percentile > 100:
        raise ValueError("mode_color_percentile must be greater than zero and at most 100.")
    if limit is not None and percentile is not None:
        raise ValueError("Set only one of mode_color_vmax and mode_color_percentile.")
    return limit, percentile


def _mode_scale_note(scale):
    clipped = ", ".join(f"mode {row['mode']}: {100*row['fraction_clipped']:.3g}%"
                         for row in scale["per_mode"])
    return (
        f"04: Spatial POD modes share one symmetric, zero-centered diverging scale "
        f"[{scale['vmin']:.8g}, {scale['vmax']:.8g}], selected by {scale['source']}. "
        "Color saturation changes the display only; stored spatial modes are not rescaled or clipped. "
        f"Displayed pixels beyond the limits: {clipped}. Triangular colorbar extensions mark "
        "out-of-range values when present. Mode signs are arbitrary; near-degenerate modes may "
        "rotate within their subspace. Mode normalization depends on the ROI and spatial resolution, "
        "so numeric amplitudes should not be compared across different crops/binning without accounting "
        "for that normalization. Exact limits and saturation fractions are saved in "
        "04_spatial_modes_color_scale.json."
    )


def plot_spatial_modes(pod_model, image_shape, output_dir, config=None) -> list[str]:
    """Render saved POD modes with a shared symmetric display color scale.

    ``mode_color_vmax`` sets explicit limits [-vmax,+vmax]. Alternatively,
    ``mode_color_percentile`` chooses that limit from pooled absolute entries
    of all displayed modes. With both unset, the complete data range is used.
    Neither option modifies modes, coefficients, POD variance, or classifiers.
    """
    config = dict(config or {})
    manual_limit, percentile = _mode_color_options(config)
    dpi = _positive_integer(config.get("dpi", config.get("figure_dpi", 600)), "dpi")
    requested = _positive_integer(config.get("n_modes_plot", 4), "n_modes_plot")
    shape = tuple(image_shape)
    if len(shape) != 2 or any(isinstance(value, (bool, np.bool_)) or
                              not isinstance(value, (int, np.integer)) or value < 1 for value in shape):
        raise ValueError("image_shape must contain two positive integers.")
    all_modes = np.asarray(pod_model.modes)
    fractions = np.asarray(pod_model.energy_fraction, dtype=float)
    if all_modes.ndim != 2 or not len(all_modes) or all_modes.shape[1] != np.prod(shape):
        raise ValueError("POD modes must be a nonempty mode-by-pixel array matching image_shape.")
    if np.iscomplexobj(all_modes) or not np.isfinite(all_modes).all():
        raise ValueError("POD spatial modes must contain finite real values.")
    if (fractions.shape != (len(all_modes),) or not np.isfinite(fractions).all()
            or np.any(fractions < 0) or fractions.sum() > 1+1e-6):
        raise ValueError("POD variance fractions must be finite, nonnegative and total at most one.")
    count = min(requested, len(all_modes))
    # This is a view used only by plotting; no normalization or in-place clipping.
    modes = all_modes[:count].reshape(count, *shape)
    full_limit = float(np.max(np.abs(modes)))
    if manual_limit is not None:
        limit, source = float(manual_limit), "manual shared limit"
    elif percentile is not None:
        limit = float(np.percentile(np.abs(modes), percentile))
        source = f"pooled absolute-value percentile {float(percentile):g}"
        if limit <= 0 and full_limit > 0:
            raise ValueError("Selected mode_color_percentile gives a zero limit; choose a higher percentile or positive mode_color_vmax.")
    else:
        limit, source = full_limit, "full displayed-mode range"
    if limit == 0:
        limit = 1.0
        source += " (all-zero display fallback)"
    extent, unit = _extent(shape, config)
    below, above = bool(np.any(modes < -limit)), bool(np.any(modes > limit))
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    scale = {
        "vmin": -limit, "vmax": limit, "center": 0.0, "source": source,
        "requested_mode_color_vmax": None if manual_limit is None else float(manual_limit),
        "requested_mode_color_percentile": None if percentile is None else float(percentile),
        "shared_across_modes": True, "displayed_mode_count": count,
        "pooled_pixel_count": int(modes.size), "full_absolute_max": full_limit,
        "colorbar_extend": extend, "display_only": True,
        "stored_mode_values_modified": False,
        "per_mode": [{"mode": index+1, "data_min": float(mode.min()), "data_max": float(mode.max()),
                      "fraction_clipped": float(np.mean(np.abs(mode) > limit)),
                      "fraction_below_limit": float(np.mean(mode < -limit)),
                      "fraction_above_limit": float(np.mean(mode > limit))}
                     for index, mode in enumerate(modes)],
    }
    ncols = min(count, 2)
    nrows = int(np.ceil(count/ncols))
    width = 7.2 if ncols == 2 else 3.7
    # Match row height to the data aspect; compressed layout removes the large
    # empty bands produced by fixed-aspect images in a fixed-height subplot grid.
    panel_width = max((width-1.25-.35*(ncols-1))/ncols, .7)
    height = nrows*(panel_width*shape[0]/shape[1]+.62)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height),
                                 squeeze=False, layout="compressed")
        active = []
        for index, ax in enumerate(axes.flat):
            if index >= count:
                ax.set_visible(False)
                continue
            image = _image(ax, modes[index], extent, unit,
                           title=f"Mode {index+1} ({100*fractions[index]:.2f}%)",
                           cmap="RdBu_r", vmin=-limit, vmax=limit)
            active.append(ax)
        fig.colorbar(image, ax=active, shrink=.95, extend=extend,
                     ticks=np.linspace(-limit, limit, 5),
                     format=FuncFormatter(lambda value, position: f"{value:.3g}"),
                     label="Unit-normalized spatial mode (dimensionless)")
        paths = _save(fig, directory, "04_spatial_modes", dpi)
    (directory/"04_spatial_modes_color_scale.json").write_text(json.dumps(scale, indent=2, allow_nan=False), encoding="utf-8")
    (directory/"04_spatial_modes_notes.md").write_text(_mode_scale_note(scale)+"\n", encoding="utf-8")
    return paths


def make_figures(X, image_shape, pod_model, scores, times_s, train_indices,
                 output_dir, metadata, classification=None, config=None) -> list[str]:
    """Export measured-image, POD, spectrum, reconstruction, and classifier figures.

    The source sequence must be uniformly sampled for a Welch spectrum. Spatial
    calibration, if provided, is interpreted as mm per original sensor pixel.
    Training indices determine coefficient normalization in the POD portrait.
    Classification arrays use the schema returned by ``compare_classifiers``.
    """
    config = dict(config or {})
    for key in ("bin_factor", "stride"):
        config.setdefault(key, metadata.get(key, 1))
    _mode_color_options(config)
    directory = Path(output_dir)
    dpi = _positive_integer(config.get("dpi", config.get("figure_dpi", 600)), "dpi")
    batch_size = _positive_integer(config.get("batch_size", 64), "batch_size")
    requested_modes = _positive_integer(config.get("n_modes_plot", 4), "n_modes_plot")
    requested_segment = _positive_integer(config.get("psd_nperseg", 128), "psd_nperseg", minimum=4)
    requested_lag = _positive_integer(config.get("acf_max_lag", 1024), "acf_max_lag")
    image_shape = tuple(image_shape)
    if len(image_shape) != 2 or any(isinstance(v, (bool, np.bool_)) or
                                  not isinstance(v, (int, np.integer)) or v < 1 for v in image_shape):
        raise ValueError("image_shape must contain two positive integers.")
    if X.ndim != 2 or X.shape[1] != np.prod(image_shape):
        raise ValueError("Image matrix pixel count must match image_shape.")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("POD scores must be a finite two-dimensional array.")
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1:
        raise ValueError("Figure timestamps must be a one-dimensional sequence.")
    train = np.asarray(train_indices)
    if train.ndim != 1 or train.dtype.kind not in "iu" or len(train) < 2 or np.any(train < 0) or np.any(train >= len(X)) or np.any(train[1:] <= train[:-1]):
        raise ValueError("Training indices must be sorted, unique integer rows in range, with at least two rows.")
    if len(X) != len(times) or len(scores) != len(X):
        raise ValueError("Images, projected scores, and timestamps must have the same length.")
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Figure timestamps must be finite and strictly increasing.")
    modes_array = np.asarray(pod_model.modes)
    fractions = np.asarray(pod_model.energy_fraction, dtype=float)
    if modes_array.ndim != 2 or modes_array.shape != (scores.shape[1], X.shape[1]) or np.asarray(pod_model.mean).shape != (X.shape[1],):
        raise ValueError("POD mean/modes must match score and image matrix dimensions.")
    if fractions.shape != (len(modes_array),) or not np.isfinite(fractions).all() or np.any(fractions < 0) or fractions.sum() > 1 + 1e-6:
        raise ValueError("POD variance fractions must be finite, nonnegative and total at most one.")
    if not np.isfinite(modes_array).all() or not np.isfinite(pod_model.mean).all():
        raise ValueError("POD mean and modes must be finite.")
    if classification is not None:
        labels = np.asarray(classification.get("labels", []))
        if labels.shape != (len(X),):
            raise ValueError("Classification labels must align with every selected frame.")
        for name in ("train_indices", "validation_indices", "test_indices"):
            rows = np.asarray(classification.get(name, []))
            if rows.ndim != 1 or rows.dtype.kind not in "iu" or len(rows) == 0 or np.any(rows < 0) or np.any(rows >= len(X)) or np.any(rows[1:] <= rows[:-1]):
                raise ValueError(f"Classifier {name} must be nonempty, sorted, unique integer rows in range.")
    extent, unit = _extent(image_shape, config)
    directory.mkdir(parents=True, exist_ok=True)
    mean, rms = _stream_stats(X, batch_size)
    n_modes = min(requested_modes, scores.shape[1], len(pod_model.modes))
    if n_modes < 1:
        raise ValueError("At least one POD mode is required for plotting.")
    paths = []
    notes = []
    t_ms = (times - times[0]) * 1e3
    with plt.rc_context(STYLE):
        # Mean and RMS summarize the selected record, not exclusively training.
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
        for ax, values, title, cmap in zip(axes, (mean, rms),
                ("(a) Mean image", "(b) RMS image fluctuations"), ("gray", "magma")):
            im = _image(ax, values.reshape(image_shape), extent, unit, title=title,
                        cmap=cmap, vmin=0 if cmap == "magma" else None)
            fig.colorbar(im, ax=ax, shrink=0.78, label="Image intensity (uncalibrated)")
        paths += _save(fig, directory, "02_mean_rms", dpi)
        notes.append("02: Mean and population RMS of image-intensity fluctuations over all selected frames. "
                     "RMS is not velocity fluctuation or turbulent kinetic energy.")

        # The denominator includes unresolved variance, so truncated cumulative
        # energy remains below one unless all variation is actually captured.
        fraction = np.asarray(pod_model.energy_fraction, dtype=float)
        order = np.arange(1, len(fraction) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
        axes[0].bar(order, 100 * fraction, color=COLORS[0], width=0.78)
        axes[0].set(xlabel="POD mode", ylabel="Image-intensity variance (%)", title="(a) Modal variance")
        axes[0].set_ylim(bottom=0)
        axes[1].plot(order, 100 * np.cumsum(fraction), "o-", color=COLORS[0], markersize=3)
        axes[1].set(xlabel="Retained POD modes", ylabel="Cumulative variance (%)", ylim=(0, 101),
                    title="(b) Cumulative resolved variance")
        axes[1].axhline(95, color="0.5", linestyle="--", linewidth=0.7)
        for ax in axes:
            ax.set_xlim(0.4, len(fraction) + 0.6)
            ax.grid(axis="y", color="0.9", linewidth=0.5)
            ax.set_axisbelow(True)
        paths += _save(fig, directory, "03_pod_variance", dpi)
        _csv(directory / "03_pod_variance.csv", ["mode", "variance_fraction", "cumulative_fraction"],
             zip(order, fraction, np.cumsum(fraction)))
        notes.append("03: POD variance is normalized by the total centered training-image variance, "
                     "including unresolved modes. It is an image-intensity statistic, not flow kinetic energy.")

        paths += plot_spatial_modes(pod_model, image_shape, directory, config)
        scale = json.loads((directory/"04_spatial_modes_color_scale.json").read_text(encoding="utf-8"))
        notes.append(_mode_scale_note(scale))

        fig, axes = plt.subplots(n_modes, 1, figsize=(7.2, 1.45*n_modes+0.4),
                                 sharex=True, squeeze=False, layout="constrained")
        for j, ax in enumerate(axes.flat):
            ax.plot(t_ms, scores[:, j], color=COLORS[j % len(COLORS)])
            ax.set_ylabel(rf"$a_{{{j+1}}}$ (a.u.)")
            ax.axhline(0, color="0.7", linewidth=0.5)
            if classification is not None:
                for split_name, color in (("train_indices", "0.95"),
                                          ("validation_indices", "0.90"), ("test_indices", "0.83")):
                    indices = np.asarray(classification.get(split_name, []), dtype=int)
                    if len(indices):
                        ax.axvspan(t_ms[indices.min()], t_ms[indices.max()], color=color, zorder=0)
        axes[-1, 0].set_xlabel("Time from first selected frame (ms)")
        axes[0, 0].set_title("POD modal coefficients")
        paths += _save(fig, directory, "05_modal_coefficients", dpi)
        notes.append("05: Coefficients are projections onto training-fitted modes. If classification is "
                     "available, gray spans mark the ranges of capped classifier training, validation, "
                     "and test examples (light to dark); unshaded regions include excluded gaps.")

        dt = np.diff(times)
        if np.allclose(dt, dt[0], rtol=1e-6, atol=1e-12):
            fs = 1 / float(dt[0])
            nperseg = min(requested_segment, len(scores))
            if nperseg >= 4:
                noverlap = nperseg // 2
                n_segments = 1 + (len(scores)-nperseg)//(nperseg-noverlap)
                fig, ax = plt.subplots(figsize=(7.2, 3.0), layout="constrained")
                psds = []
                for j in range(n_modes):
                    freq, psd = welch(scores[:, j], fs=fs, window="hann", nperseg=nperseg,
                                      noverlap=noverlap, nfft=nperseg, detrend="constant",
                                      scaling="density")
                    psds.append(psd)
                    ax.semilogy(freq[1:]/1e3, np.maximum(psd[1:], np.finfo(float).tiny),
                                color=COLORS[j % len(COLORS)], linestyle=LINESTYLES[j%len(LINESTYLES)],
                                label=f"Mode {j+1}")
                ax.set(xlabel="Frequency (kHz)", ylabel=r"PSD (coefficient a.u.$^2$/Hz)",
                       xlim=(0, fs/2e3), title=f"Welch spectra; {n_segments} segments, bin spacing {fs/nperseg:.1f} Hz")
                ax.legend(frameon=False, ncol=min(n_modes, 4))
                ax.grid(axis="y", which="major", color="0.9", linewidth=0.5)
                paths += _save(fig, directory, "06_modal_spectra", dpi)
                _csv(directory / "06_modal_spectra.csv", ["frequency_hz"] + [f"mode_{j+1}_psd" for j in range(n_modes)],
                     zip(freq, *psds))
                notes.append(f"06: Welch PSD uses the complete uniformly sampled selected record, "
                             f"fs={fs:.8g} Hz, Hann windows, {nperseg} samples/window, {noverlap} overlapping samples "
                             f"({100*noverlap/nperseg:.4g}%), {n_segments} segments, "
                             f"constant detrending, and no zero padding. Record span is {times[-1]-times[0]:.8g} s; "
                             f"bin spacing is {fs/nperseg:.8g} Hz. Short records limit spectral inference. "
                             "Frame skipping changes effective fs and requires anti-alias treatment before interpreting spectra.")
        else:
            notes.append("06: A Welch spectrum was not exported because selected timestamps are irregular.")

        if scores.shape[1] >= 2:
            center = scores[train, :2].mean(axis=0)
            scale = scores[train, :2].std(axis=0, ddof=0)
            scale[scale == 0] = 1
            plane = (scores[:, :2] - center) / scale
            fig, ax = plt.subplots(figsize=(3.5, 3.2), layout="constrained")
            if classification is None:
                points = ax.scatter(plane[:, 0], plane[:, 1], c=t_ms, cmap="viridis", s=10,
                                    linewidths=0, alpha=0.8, rasterized=len(plane)>2000)
                fig.colorbar(points, ax=ax, label="Time (ms)")
            else:
                labels = np.asarray(classification["labels"])
                classes = np.unique(labels)
                train_mask = np.zeros(len(X), dtype=bool)
                test_mask = np.zeros(len(X), dtype=bool)
                train_mask[np.asarray(classification.get("train_indices", train), dtype=int)] = True
                test_mask[np.asarray(classification.get("test_indices", []), dtype=int)] = True
                for j, label in enumerate(classes):
                    mask = labels == label
                    color = COLORS[j % len(COLORS)]
                    ax.scatter(plane[mask & ~test_mask, 0], plane[mask & ~test_mask, 1],
                               color=color, marker=MARKERS[j%len(MARKERS)], s=13, linewidths=0,
                               alpha=0.55, label=str(label),
                               rasterized=len(plane)>2000)
                    ax.scatter(plane[mask & test_mask, 0], plane[mask & test_mask, 1],
                               facecolors="none", edgecolors=color, marker=MARKERS[j%len(MARKERS)],
                               s=26, linewidths=0.8)
                ax.legend(frameon=False, loc="best")
            ax.set(xlabel=r"$(a_1-\overline{a}_{1,\mathrm{train}})/s_{1,\mathrm{train}}$",
                   ylabel=r"$(a_2-\overline{a}_{2,\mathrm{train}})/s_{2,\mathrm{train}}$",
                   title="Training-standardized POD plane")
            paths += _save(fig, directory, "07_pod_plane", dpi)
            notes.append("07: First two POD coefficients standardized with training means and standard deviations. "
                         "Open symbols identify the classifier's held-out test examples when available; "
                         "symbol shape and color identify assigned states. "
                         "Cluster-derived state labels are algorithmic surrogates, not independently established physical regimes.")

        selected = int(np.asarray(classification["test_indices"])[-1]) if classification is not None else len(X)-1
        observed = np.asarray(X[selected], dtype=float)
        reconstructed = np.asarray(pod_model.mean) + scores[selected] @ np.asarray(pod_model.modes)
        residual = observed - reconstructed
        vmin, vmax_image = float(min(observed.min(), reconstructed.min())), float(max(observed.max(), reconstructed.max()))
        residual_lim = float(np.max(np.abs(residual))) or 1
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), layout="constrained")
        for ax, values, title in zip(axes[:2], (observed, reconstructed),
                                     ("(a) Measured", "(b) POD estimate")):
            im = _image(ax, values.reshape(image_shape), extent, unit, title=title,
                        vmin=vmin, vmax=vmax_image)
        intensity_bar = fig.colorbar(im, ax=list(axes[:2]), orientation="horizontal", shrink=0.85,
                                     pad=0.14, label="Intensity (a.u.)")
        intensity_bar.locator = MaxNLocator(nbins=4)
        intensity_bar.update_ticks()
        im = _image(axes[2], residual.reshape(image_shape), extent, unit, title="(c) Residual",
                    cmap="RdBu_r", vmin=-residual_lim, vmax=residual_lim)
        residual_bar = fig.colorbar(im, ax=axes[2], orientation="horizontal", shrink=0.85,
                                    pad=0.14, label="Residual intensity (a.u.)")
        residual_bar.locator = MaxNLocator(nbins=3)
        residual_exponent = int(np.floor(np.log10(residual_lim)))
        if residual_exponent < -2 or residual_exponent >= 3:
            residual_factor = 10.**residual_exponent
            residual_bar.formatter = FuncFormatter(lambda value, position: f"{value/residual_factor:g}")
            residual_bar.set_label(rf"Residual ($10^{{{residual_exponent}}}$ a.u.)")
        residual_bar.update_ticks()
        paths += _save(fig, directory, "08_reconstruction", dpi)
        source_record = metadata.get("frame_records", [])
        source_record = source_record[selected] if len(source_record) > selected else {}
        notes.append(f"08: Reconstruction of selected sequence row {selected}, "
                     f"source filename {source_record.get('filename', 'not supplied')}, "
                     f"original sequence index {source_record.get('frame_index', 'not supplied')}, "
                     f"time {times[selected]:.10g} s, with the training mean and "
                     f"{len(pod_model.modes)} modes. Observed and reconstructed panels share intensity limits; "
                     "the residual has its own symmetric scale.")

        if classification is not None:
            paths += _classification_figures(classification, directory, dpi, notes, metadata=metadata)
            paths += _state_figures(X, image_shape, extent, unit, train, t_ms,
                                    classification, directory, dpi, notes,
                                    batch_size)

        train_steps = np.diff(times[train])
        if len(train) >= 16 and np.allclose(train_steps, train_steps[0], rtol=1e-6, atol=1e-12):
            max_lag = min(requested_lag, len(train)//4)
            lags = np.arange(max_lag+1)
            acfs = []
            fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
            for j in range(n_modes):
                signal = scores[train, j] - scores[train, j].mean()
                nfft = 1 << (2*len(signal)-1).bit_length()
                spectrum = np.fft.rfft(signal, n=nfft)
                corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:max_lag+1]
                acf = corr/corr[0] if corr[0] > 0 else np.full(len(corr), np.nan)
                acfs.append(acf)
                ax.plot(lags*train_steps[0]*1e3, acf, color=COLORS[j%len(COLORS)],
                        linestyle=LINESTYLES[j%len(LINESTYLES)], label=f"Mode {j+1}")
            ax.axhline(0, color="0.6", linewidth=0.6)
            ax.axhline(1/np.e, color="0.7", linewidth=0.6, linestyle=":")
            ax.set(xlabel="Time lag (ms)", ylabel="Normalized autocorrelation", ylim=(-1.05,1.05),
                   title="Training-record modal autocorrelation")
            ax.legend(frameon=False, ncol=min(n_modes,4))
            paths += _save(fig, directory, "13_modal_autocorrelation", dpi)
            original_fps = metadata.get("fps", config.get("fps"))
            if original_fps is not None and np.isfinite(original_fps) and original_fps > 0:
                original_frame_lags = lags*train_steps[0]*float(original_fps)
                original_frame_lags = np.where(np.isclose(original_frame_lags, np.rint(original_frame_lags)),
                                               np.rint(original_frame_lags), original_frame_lags)
            else:
                original_frame_lags = lags*np.diff(train)[0]*config.get("stride", 1)
            _csv(directory / "13_modal_autocorrelation.csv", ["lag_selected_samples", "lag_original_frames", "lag_s"]+
                 [f"mode_{j+1}_acf" for j in range(n_modes)],
                 zip(lags, original_frame_lags, lags*train_steps[0], *acfs))
            notes.append("13: Mean-subtracted, biased sample autocorrelations use the full uniformly "
                         "sampled POD training partition and are normalized by their zero-lag value. "
                         "The dotted guide is 1/e; it is not a statistical significance bound. "
                         "Examine recurrence and slowly decaying modes before choosing temporal split gaps; "
                         "a single threshold crossing cannot establish independence. Source-table lags "
                         "distinguish selected sample intervals, original camera frames and seconds.")

        if metadata.get("reconstruction_metrics") is not None:
            paths += _reconstruction_error_figure(metadata["reconstruction_metrics"],
                                                   directory, dpi, notes)

    notes += ["Spatial coordinates are ROI-relative image coordinates; image y increases downward. "
              "Coordinates use pixels unless an explicit pixel_size_mm calibration is supplied. "
              "Sensor cropping and binning must be documented separately in the methods.",
              "All PDFs embed TrueType fonts; SVGs keep editable text; PNGs use the configured DPI. "
              "Journal-specific type size, line width, panel dimensions, and resolution requirements still apply."]
    (directory / "FIGURE_NOTES.md").write_text("# Figure interpretation and export notes\n\n" +
                                               "\n\n".join(notes) + "\n", encoding="utf-8")
    return paths


def _reconstruction_error_figure(metrics, directory, dpi, notes):
    """Render reviewed partition reconstruction errors without choosing a rank."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0), layout="constrained")
    rows = []
    has_points = False
    for j, (key, name) in enumerate((("train", "Training"), ("validation", "Validation"),
                                     ("test", "Test"))):
        values = metrics.get(key)
        if values is None:
            continue
        ranks = np.asarray(values["ranks"], dtype=int)
        errors = np.asarray([np.nan if value is None else value for value in
                             values["relative_rms_fluctuation_error"]], dtype=float)
        if ranks.ndim != 1 or len(ranks) != len(errors) or np.any(ranks < 1):
            raise ValueError("Reconstruction-error ranks and error arrays must align with positive ranks.")
        if np.any(errors[np.isfinite(errors)] < 0):
            raise ValueError("Relative RMS reconstruction errors cannot be negative.")
        finite = np.isfinite(errors)
        n = int(values["n_frames"])
        if np.any(finite):
            ax.plot(ranks, errors, color=COLORS[j], linestyle=LINESTYLES[j], marker=MARKERS[j],
                    markersize=3, label=f"{name} (n={n})")
            has_points = True
        centered_sum = values.get("centered_sum_squares")
        rows.extend((key, int(rank), None if not np.isfinite(error) else float(error), n, centered_sum)
                    for rank, error in zip(ranks, errors))
    ax.set(xlabel="Retained POD rank", ylabel="Relative RMS image-fluctuation error",
           title="Reconstruction error relative to training-mean-centered fluctuations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    if has_points:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Relative error is undefined for records with zero centered variance",
                transform=ax.transAxes, ha="center", va="center", wrap=True)
    paths = _save(fig, directory, "14_reconstruction_error_by_rank", dpi)
    _csv(directory / "14_reconstruction_error_by_rank.csv",
         ["partition", "rank", "relative_rms_fluctuation_error", "n_frames", "centered_sum_squares"], rows)
    notes.append("14: Each curve reports sqrt[sum ||I-I_reconstructed(r)||^2 / "
                 "sum ||I-training_mean||^2] within its stated partition. The denominator uses "
                 "that partition's image fluctuations about the fixed training mean. Training-fitted "
                 "modes and mean are held fixed for every partition; no confidence interval is asserted. "
                 "These curves are descriptive diagnostics and were not used to tune the model. "
                 "A zero denominator produces an undefined value rather than a fabricated zero error.")
    return paths


def _state_figures(X, shape, extent, unit, train, t_ms, result, directory, dpi, notes, batch_size):
    """Training-only state composites and descriptive full-record assignments."""
    labels = np.asarray(result["labels"])
    if len(labels) != len(X):
        raise ValueError("State labels must align with all selected snapshots.")
    classes = np.unique(labels)
    sums = np.zeros((len(classes), X.shape[1]), dtype=np.float64)
    count = np.zeros(len(classes), dtype=int)
    for start in range(0, len(train), batch_size):
        indices = train[start:start+batch_size]
        images = np.asarray(X[indices], dtype=np.float64)
        for j, state in enumerate(classes):
            mask = labels[indices] == state
            if np.any(mask):
                sums[j] += images[mask].sum(axis=0)
                count[j] += int(mask.sum())
    means = np.divide(sums, count[:, None], out=np.full_like(sums, np.nan), where=count[:,None]>0)
    valid = means[count>0]
    low, high = (float(valid.min()), float(valid.max())) if len(valid) else (0., 1.)
    ncols = min(len(classes), 3)
    nrows = int(np.ceil(len(classes)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.7*nrows),
                             squeeze=False, layout="constrained")
    im = None
    active = []
    for j, ax in enumerate(axes.flat):
        if j >= len(classes):
            ax.set_visible(False)
            continue
        if count[j] == 0:
            ax.text(0.5, 0.5, f"{classes[j]}\nNo training support", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        im = _image(ax, means[j].reshape(shape), extent, unit, vmin=low, vmax=high,
                    title=f"{classes[j]} (training n={count[j]})")
        active.append(ax)
    if im is not None:
        fig.colorbar(im, ax=active, shrink=0.8, label="Mean intensity (a.u.)")
    paths = _save(fig, directory, "11_state_conditional_means", dpi)
    _csv(directory / "11_state_training_support.csv", ["state", "training_frames"], zip(classes, count))
    np.savez_compressed(directory / "11_state_conditional_means.npz", states=classes,
                        training_counts=count, mean_images=means.reshape(len(classes), *shape))
    notes.append("11: Conditional mean images use only the full POD training partition; support is "
                 "reported for each state and all panels share intensity limits. Cluster-derived states "
                 "are candidate partitions whose physical interpretation requires independent evidence. "
                 "Mean images can obscure within-state variation.")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained", gridspec_kw={"width_ratios":[2,1]})
    state_number = np.empty(len(labels), dtype=int)
    total = np.zeros(len(classes), dtype=int)
    for j, state in enumerate(classes):
        mask = labels == state
        state_number[mask] = j
        total[j] = int(mask.sum())
    axes[0].step(t_ms, state_number, where="mid", color="0.55", linewidth=0.6)
    for j, state in enumerate(classes):
        mask = labels == state
        axes[0].scatter(t_ms[mask], state_number[mask], s=7, color=COLORS[j%len(COLORS)],
                        linewidths=0, rasterized=len(labels)>2000)
    axes[0].set_yticks(range(len(classes)), [str(c) for c in classes])
    axes[0].set(xlabel="Time from first selected frame (ms)", ylabel="Assigned state",
                title="(a) Full-record state assignments", ylim=(-0.5, len(classes)-0.5))
    occupancy = total / len(labels)
    axes[1].barh(range(len(classes)), 100*occupancy, color=[COLORS[j%len(COLORS)] for j in range(len(classes))])
    axes[1].set_yticks(range(len(classes)), [str(c) for c in classes])
    axes[1].set(xlabel="Selected frames (%)", title="(b) Record occupancy", xlim=(0, max(100*occupancy.max()*1.2, 5)))
    for j, value in enumerate(occupancy):
        axes[1].text(100*value+0.8, j, f"{100*value:.1f}%", va="center", fontsize=8)
    paths += _save(fig, directory, "12_state_timeline_occupancy", dpi)
    _csv(directory / "12_state_occupancy.csv", ["state", "selected_frame_count", "selected_record_fraction"],
         zip(classes, total, occupancy))
    _csv(directory / "12_state_timeline.csv", ["time_ms", "assigned_state"], zip(t_ms, labels))
    notes.append("12: State assignments and occupancy describe all selected frames, including training "
                 "and held-out rows. Occupancy is a descriptive frame fraction, not an independent "
                 "statistical prevalence estimate; nearby frames are correlated. Candidate-state "
                 "assignments come from a training-fitted clustering rule when no external labels exist.")
    return paths


def _classification_figures(result, directory, dpi, notes, metadata=None):
    paths = []
    names = [("rbf_svm", "RBF-SVM"), ("qsvm", "Fidelity-kernel SVM")]
    available = [(key, name) for key, name in names if key in result.get("confusion", {})]
    if available:
        source_rows = []
        sample_counts = []
        fig, axes = plt.subplots(1, len(available), figsize=(7.2, 3.9), squeeze=False, layout="constrained")
        for ax, (key, title) in zip(axes.flat, available):
            counts = np.asarray(result["confusion"][key], dtype=int)
            raw_counts = np.asarray(result["confusion"][key])
            if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.size == 0 or not np.isfinite(raw_counts).all() or np.any(raw_counts != counts) or np.any(counts < 0):
                plt.close(fig)
                raise ValueError("Confusion matrices must contain nonnegative integer counts in a square array.")
            denom = counts.sum(axis=1, keepdims=True)
            normal = np.divide(counts, denom, out=np.zeros_like(counts, dtype=float), where=denom>0)
            sample_counts.append(int(counts.sum()))
            observed = denom[:, 0] > 0
            balanced = float(np.mean(np.diag(normal)[observed])) if np.any(observed) else None
            im = ax.imshow(normal, vmin=0, vmax=1, cmap="Blues", aspect="equal")
            classes = result.get("metrics", {}).get("classes", np.unique(result["labels"]).tolist())
            if len(classes) != len(counts):
                plt.close(fig)
                raise ValueError("Confusion matrix dimensions must match the declared class names.")
            tick_labels = [
                ("Candidate "+str(c).removeprefix("candidate_")) if str(c).startswith("candidate_")
                else textwrap.fill(str(c).replace("_", " "), width=12)
                for c in classes
            ]
            ax.set_xticks(range(len(counts)), tick_labels)
            ax.set_yticks(range(len(counts)), tick_labels)
            for i in range(len(counts)):
                for j in range(len(counts)):
                    label = f"{100*normal[i,j]:.1f}%\n(n={counts[i,j]})" if denom[i,0]>0 else "No test\nsamples"
                    ax.text(j, i, label, ha="center", va="center", color="white" if normal[i,j]>.55 else "0.15", fontsize=8)
                    source_rows.append((key, str(classes[i]), str(classes[j]), int(counts[i,j]),
                                        int(denom[i,0]), float(normal[i,j]) if denom[i,0] else None))
            performance = f"Balanced accuracy: {100*balanced:.1f}%" if balanced is not None else "Balanced accuracy: undefined"
            ax.set(xlabel="Predicted state", ylabel="Reference state", title=f"{title}\n{performance}")
        label_mode = result.get("label_mode", result.get("metrics", {}).get("label_mode"))
        if label_mode == "cluster-surrogate":
            label_source = "training-defined K-means assignments"
        elif label_mode == "external":
            label_source = "external labels"
        else:
            label_source = "source not specified"
        sample_description = (f"N = {sample_counts[0]} held-out test examples" if len(set(sample_counts)) == 1
                              else "; ".join(f"{title}: N = {count}" for (_, title), count in zip(available, sample_counts)))
        fig.suptitle(f"{sample_description}\nReference labels: {label_source}", fontsize=9)
        fig.colorbar(im, ax=list(axes.flat), shrink=0.8, label="Fraction within reference state")
        paths += _save(fig, directory, "09_confusion_matrices", dpi)
        _csv(directory / "09_confusion_matrices.csv", ["model", "reference_state", "predicted_state",
             "count", "reference_support", "fraction_within_reference"], source_rows)
        notes.append(f"09: {sample_description}. Reference labels: {label_source}. "
                     "Held-out confusion matrices are row-normalized; raw counts are printed in every cell. "
                     "Percentages are rounded to one decimal place and may not sum to exactly 100%. "
                     "Balanced accuracy is mean recall over reference states with test support. "
                     "Absent reference states have no test evidence. Cluster assignments are surrogate "
                     "targets rather than independent physical flow-state ground truth.")
    kernels = result.get("kernels", {})
    if "train" in kernels:
        matrix = np.asarray(kernels["train"])
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0 or not np.isfinite(matrix).all() or np.any(matrix < -1e-9) or np.any(matrix > 1+1e-9):
            raise ValueError("Training fidelity kernel must be a finite square array with values in [0,1].")
        if matrix.ndim == 2:
            test_matrix = np.asarray(kernels.get("test", []))
            has_test = (test_matrix.ndim == 2 and len(test_matrix) > 0 and
                        test_matrix.shape[1] == matrix.shape[1])
            if "test" in kernels and (not has_test or not np.isfinite(test_matrix).all() or
                                      np.any(test_matrix < -1e-9) or np.any(test_matrix > 1+1e-9)):
                raise ValueError("Test fidelity kernel must have one column per training example and values in [0,1].")
            count = 2 if has_test else 1
            fig, axes = plt.subplots(1, count, figsize=(7.2 if has_test else 3.5, 3.2),
                                     squeeze=False, layout="constrained")
            im = axes[0, 0].imshow(matrix, cmap="viridis", norm=Normalize(0, 1),
                                    aspect="auto", interpolation="nearest")
            axes[0, 0].set(xlabel="Training example index", ylabel="Training example index",
                            title="(a) Training fidelity kernel" if has_test else "Training fidelity kernel")
            if has_test:
                axes[0, 1].imshow(test_matrix, cmap="viridis", norm=Normalize(0, 1),
                                  aspect="auto", interpolation="nearest")
                axes[0, 1].set(xlabel="Training example index", ylabel="Test example index",
                                title="(b) Test-versus-training kernel")
            for ax in axes.flat:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
            fig.colorbar(im, ax=list(axes.flat), shrink=0.85, label="State fidelity")
            paths += _save(fig, directory, "10_quantum_kernel", dpi)
            notes.append("10: Exact statevector fidelity kernels on capped classifier examples in time order. "
                         "Training and test-versus-training panels, when present, share the fidelity range [0,1]. "
                         "This is a classical simulation of a quantum feature map; no quantum hardware execution is implied.")
            records = (metadata or {}).get("frame_records", [])
            sample_rows = []
            for partition in ("train", "test"):
                for position, selected_row in enumerate(result.get(f"{partition}_indices", [])):
                    record = records[selected_row] if len(records) > selected_row else {}
                    sample_rows.append((partition, position, int(selected_row), record.get("filename"),
                                        record.get("frame_index"), record.get("time_s")))
            _csv(directory / "10_quantum_kernel_samples.csv", ["partition", "kernel_example_index",
                 "selected_sequence_row", "filename", "original_frame_index", "time_s"], sample_rows)
            np.savez_compressed(directory / "10_quantum_kernel.npz", train=matrix,
                                **({"test": test_matrix} if has_test else {}))
    return paths
