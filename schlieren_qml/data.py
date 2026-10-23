"""Image ingest with explicit, reproducible spatial and temporal sampling.

Intensities retain their camera numeric scale. No framewise contrast adjustment,
normalization, thresholding, or registration is performed implicitly.
"""
from __future__ import annotations

import json
from numbers import Real
import re
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _same_file(left: Path, right: Path) -> bool:
    """Compare paths including symlinks and existing hard-link aliases."""
    return left.resolve() == right.resolve() or (left.exists() and right.exists() and left.samefile(right))


def _natural_key(path: Path):
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                 for part in re.split(r"(\d+)", str(path)))


def list_images(directory: str | Path, pattern: str = "*") -> list[Path]:
    """Naturally sort image files, e.g. frame2 precedes frame10.

    The sort order must be checked against the acquisition sequence. Timestamps
    below assume no dropped files and a constant acquisition frame rate.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    paths = sorted((p for p in directory.glob(pattern)
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS),
                   key=_natural_key)
    if not paths:
        raise ValueError(f"No supported single-frame images found in {directory}")
    return paths


def read_image(path: str | Path) -> np.ndarray:
    """Read a single grayscale/RGB image as float32, retaining numeric depth."""
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Multi-page/animated input is unsupported: {path}")
        if image.mode in {"P", "CMYK", "YCbCr", "HSV"}:
            image = image.convert("RGB")
        elif image.mode not in {"1", "L", "I", "F", "I;16", "I;16L", "I;16B", "I;16N", "RGB", "RGBA"}:
            raise ValueError(f"Unsupported image color mode {image.mode!r}: {path}")
        array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        # RGB is uncommon for schlieren cameras. Alpha is ignored explicitly.
        array = np.asarray(array[..., :3], dtype=np.float32)
        array = array @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    elif array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale or RGB image; got {array.shape}: {path}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Image contains non-finite pixels: {path}")
    return array


def preprocess_image(image: np.ndarray, roi: Sequence[int] | None = None,
                     bin_factor: int = 1) -> np.ndarray:
    """Crop [x0,y0,x1,y1), then area-average square bins.

    Binning requires exact divisibility. This avoids silently discarding edge
    pixels or changing the requested field of view.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("preprocess_image requires finite 2D grayscale data")
    if roi is not None:
        if not isinstance(roi, Sequence) and not isinstance(roi, np.ndarray):
            raise ValueError("ROI must be four integer coordinates [x0,y0,x1,y1]")
        if len(roi) != 4:
            raise ValueError("ROI must be four integer coordinates [x0,y0,x1,y1]")
        x0, y0, x1, y1 = (_integer(v, "ROI coordinate") for v in roi)
        height, width = image.shape
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"ROI {list(roi)} lies outside image width={width}, height={height}")
        image = image[y0:y1, x0:x1]
    factor = _integer(bin_factor, "bin_factor", 1)
    height, width = image.shape
    if height % factor or width % factor:
        raise ValueError(f"Cropped shape {image.shape} must be divisible by bin_factor={factor}")
    if factor > 1:
        image = image.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))
    return np.ascontiguousarray(image, dtype=np.float32)


def build_image_matrix(paths: Sequence[str | Path], cache_path: str | Path,
                       fps: float = 51032.0, roi: Sequence[int] | None = None,
                       bin_factor: int = 1, start: int = 0, limit: int | None = 200,
                       stride: int = 1, manifest_path: str | Path | None = None
                       ) -> tuple[np.memmap, dict]:
    """Write a memory-mapped (frame,pixel) matrix and return its provenance.

    ``frame_index`` is the zero-based index in the complete sorted input list;
    ``time_s`` equals this index divided by fps. Sampling stride changes the
    effective sampling frequency, and is never hidden by renumbering frames.
    """
    if isinstance(fps, (bool, np.bool_)) or not isinstance(fps, Real) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    start = _integer(start, "start")
    stride = _integer(stride, "stride", 1)
    if limit is not None:
        limit = _integer(limit, "limit", 1)
    paths = [Path(p) for p in paths]
    indices = list(range(start, len(paths), stride))
    if limit is not None:
        indices = indices[:limit]
    if not indices:
        raise ValueError("No frames remain after start/limit/stride selection")
    cache_path = Path(cache_path)
    if cache_path.suffix != ".npy":
        raise ValueError("Image matrix cache path must end in .npy")
    manifest_path = Path(manifest_path) if manifest_path is not None else None
    # Validate both destinations before creating or truncating anything. Protect
    # every supplied raw image, including frames omitted by limit or stride.
    destinations = [("Cache", cache_path)]
    if manifest_path is not None:
        if _same_file(cache_path, manifest_path):
            raise ValueError("Manifest path and cache path must be different files")
        destinations.append(("Manifest", manifest_path))
    resolved_sources = {path.resolve() for path in paths}
    for name, destination in destinations:
        if destination.is_dir():
            raise ValueError(f"{name} path is a directory: {destination}")
        if destination.resolve() in resolved_sources or (destination.exists() and any(_same_file(destination, source) for source in paths)):
            raise ValueError(f"{name} path cannot overwrite a source image")
    first_raw = read_image(paths[indices[0]])
    original_shape = first_raw.shape
    first = preprocess_image(first_raw, roi, bin_factor)
    processed_shape = first.shape
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = len(indices) * first.size * np.dtype(np.float32).itemsize + 4096
    if shutil.disk_usage(cache_path.parent).free < required_bytes:
        raise ValueError(f"Insufficient cache disk space; need approximately {required_bytes / 1e9:.2f} GB")
    matrix = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32,
                                       shape=(len(indices), first.size))
    records = []
    for row, index in enumerate(indices):
        raw = first_raw if row == 0 else read_image(paths[index])
        if raw.shape != original_shape:
            raise ValueError(f"Image dimensions changed at {paths[index]}: {raw.shape} != {original_shape}")
        image = first if row == 0 else preprocess_image(raw, roi, bin_factor)
        matrix[row] = image.ravel()
        source_stat = paths[index].stat()
        records.append({"row": row, "frame_index": index, "filename": paths[index].name,
                        "file_size_bytes": source_stat.st_size, "file_mtime_ns": source_stat.st_mtime_ns,
                        "path": str(paths[index].resolve()), "time_s": index / float(fps),
                        "mean_intensity": float(image.mean(dtype=np.float64)),
                        "std_intensity": float(image.std(dtype=np.float64)),
                        "min_intensity": float(image.min()), "max_intensity": float(image.max())})
    matrix.flush()
    metadata = {"fps": float(fps), "effective_fps": float(fps) / int(stride),
                "start": int(start), "stride": int(stride), "requested_limit": limit,
                "n_frames": len(indices), "n_input_files": len(paths),
                "original_shape": list(original_shape), "image_shape": list(processed_shape),
                "roi": [int(value) for value in roi] if roi is not None else [0, 0, original_shape[1], original_shape[0]],
                "bin_factor": int(bin_factor), "dtype": "float32",
                "cache_path": str(cache_path.resolve()),
                "sampling_assumption": "Sorted source list is a complete, constant-fps sequence without dropped files.",
                "preprocessing": "Fixed crop and area binning; native numeric intensity; no per-frame normalization.",
                "frame_records": records}
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return matrix, metadata
