"""Inspect saved image-to-candidate assignments without rerunning an analysis.

``state`` labels every processed frame using the saved clustering rule. SVM
prediction columns label only evaluated test examples; their other cells are
intentionally blank. The default gallery uses training images so inspecting
candidate structure need not consume a held-out evaluation set.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from numbers import Integral
from pathlib import Path, PureWindowsPath
import re
import tempfile

import numpy as np

from .data import read_image, preprocess_image, _same_file


_REQUIRED_COLUMNS = {"filename", "sequence_frame_index", "time_s", "partition", "state"}
_LABEL_COLUMNS = {"state", "rbf_test_prediction", "qsvm_test_prediction"}
_PARTITIONS = {"train", "validation", "test", "excluded"}


def _load_results(results_directory):
    directory = Path(results_directory).expanduser().resolve()
    config = json.loads((directory/"config_used.json").read_text(encoding="utf-8-sig"))
    metadata = json.loads((directory/"frames.json").read_text(encoding="utf-8-sig"))
    with (directory/"state_assignments.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if len(columns) != len(set(columns)) or not _REQUIRED_COLUMNS.issubset(columns):
            raise ValueError("state_assignments.csv must have unique filename, sequence_frame_index, time_s, partition, and state columns.")
        records = list(reader)
    if not records:
        raise ValueError("The saved candidate assignments are empty.")
    by_index = {}
    for record in metadata.get("frame_records", []):
        try:
            index = int(record["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frames.json has a malformed frame index.") from exc
        if index in by_index:
            raise ValueError("frames.json contains duplicate frame indices.")
        by_index[index] = record
    if len(by_index) != len(records):
        raise ValueError("frames.json and state_assignments.csv contain different frame counts.")
    seen = set()
    for row in records:
        try:
            if None in row or any(row.get(column) is None for column in columns):
                raise ValueError("Missing or extra CSV fields")
            index = int(row["sequence_frame_index"])
            time = float(row["time_s"])
            original = by_index[index]
            name = row["filename"]
            if (not name or name in {".", ".."} or Path(name).name != name
                    or PureWindowsPath(name).name != name):
                raise ValueError("Filename must be a basename")
            if index < 0 or index in seen or not np.isfinite(time):
                raise ValueError("Duplicate/invalid frame identity or timestamp")
            if original["filename"] != name or PureWindowsPath(original["path"]).name != name:
                raise ValueError("Filename identity differs")
            if not np.isclose(time, float(original["time_s"]), rtol=1e-10, atol=1e-12):
                raise ValueError("Frame timestamps differ")
            if row["partition"] not in _PARTITIONS or not row["state"].strip():
                raise ValueError("Missing candidate or invalid partition")
            if row["partition"] != "test" and any(row.get(column, "").strip()
                                                      for column in _LABEL_COLUMNS-{"state"}):
                raise ValueError("SVM predictions occur outside the test partition")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Saved candidate assignments do not match frames.json; select files from the same completed run.") from exc
        seen.add(index)
    if config.get("bin_factor", 1) != metadata.get("bin_factor", 1):
        raise ValueError("Saved configuration and frame metadata disagree about spatial binning.")
    if "roi" in metadata:
        original_shape = metadata.get("original_shape")

        def normalized_roi(value):
            if value is None:
                return ([0, 0, original_shape[1], original_shape[0]]
                        if original_shape is not None else None)
            return list(value)

        if normalized_roi(config.get("roi")) != normalized_roi(metadata["roi"]):
            raise ValueError("Saved configuration and frame metadata disagree about the crop.")
    return directory, config, metadata, columns, records, by_index


def _image_for_row(row, config, metadata, by_index, processed, image_directory_override):
    original = by_index[int(row["sequence_frame_index"])]
    if image_directory_override is not None and sum(
            record["filename"] == row["filename"] for record in by_index.values()) > 1:
        raise ValueError("The image-directory override is ambiguous for duplicate basenames; use the recorded source paths.")
    path = (Path(image_directory_override).expanduser()/row["filename"]
            if image_directory_override is not None else Path(original["path"]).expanduser())
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find {path}. If the TIFF folder moved, set image_directory_override.")
    image = read_image(path)
    expected_raw = metadata.get("original_shape")
    if expected_raw is not None and tuple(image.shape) != tuple(expected_raw):
        raise ValueError(f"Image dimensions changed since the saved run: {row['filename']}.")
    if processed:
        image = preprocess_image(image, config.get("roi"), config.get("bin_factor", 1))
        expected_shape = metadata.get("image_shape")
        if expected_shape is not None and tuple(image.shape) != tuple(expected_shape):
            raise ValueError("The saved crop/binning does not reproduce the recorded image shape.")
    return image


def _limits(images):
    low = min(float(image.min()) for image in images)
    high = max(float(image.max()) for image in images)
    return low, high if high > low else low+1.0


def candidate_gallery(results_directory, *, examples_per_candidate=4, seed=42,
                      label_column="state", partition="train", processed=True,
                      image_directory_override=None, only_candidate=None):
    """Return a gallery figure and exact selected rows from a saved run.

    ``counts`` describes the eligible population after partition/prediction
    filtering, rather than the small displayed sample. Set ``partition=None``
    for the full record, or ``partition='test'`` for SVM prediction galleries.
    All displayed images share a grayscale range; original TIFFs retain their
    native numeric intensity scale. The caller owns displaying/closing figure.
    """
    if (isinstance(examples_per_candidate, (bool, np.bool_))
            or not isinstance(examples_per_candidate, Integral)
            or not 1 <= examples_per_candidate <= 12):
        raise ValueError("examples_per_candidate must be an integer from 1 to 12.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not isinstance(processed, (bool, np.bool_)):
        raise ValueError("processed must be a boolean.")
    if partition is not None and partition not in _PARTITIONS:
        raise ValueError("partition must be train, validation, test, excluded, or None.")
    _, config, metadata, columns, records, by_index = _load_results(results_directory)
    if label_column not in _LABEL_COLUMNS or label_column not in columns:
        raise ValueError("label_column must be state, rbf_test_prediction, or qsvm_test_prediction.")
    eligible = [row for row in records if row[label_column].strip()
                and (partition is None or row["partition"] == partition)]
    if only_candidate is not None:
        eligible = [row for row in eligible if row[label_column] == only_candidate]
    counts = dict(sorted(Counter(row[label_column] for row in eligible).items()))
    if not counts:
        raise ValueError("No images match the selection. SVM prediction columns are populated only for evaluated test examples; use partition='test'.")
    rng = np.random.default_rng(int(seed))
    groups, selected = [], []
    for label in counts:
        pool = [row for row in eligible if row[label_column] == label]
        positions = np.sort(rng.choice(len(pool), min(int(examples_per_candidate), len(pool)), replace=False))
        group = []
        for position in positions:
            row = pool[int(position)]
            image = _image_for_row(row, config, metadata, by_index, processed, image_directory_override)
            group.append((row, image))
            selected.append(dict(row))
        groups.append(group)
    low, high = _limits([image for group in groups for _, image in group])
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(groups), int(examples_per_candidate), squeeze=False,
                                figsize=(3.6*int(examples_per_candidate), 3.1*len(groups)),
                                layout="constrained")
    for i, group in enumerate(groups):
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j >= len(group):
                continue
            row, image = group[j]
            displayed = ax.imshow(image, cmap="gray", vmin=low, vmax=high, origin="upper")
            labels = (row["state"] if label_column == "state" else
                      f"assigned {row['state']}\npredicted {row[label_column]}")
            ax.set_title(f"{labels}\n{row['filename']}\n"
                         f"{float(row['time_s'])*1000:.3f} ms | {row['partition']}", fontsize=9)
    view = "saved crop and binning" if processed else "original TIFF pixels"
    population = partition if partition is not None else "all partitions"
    figure.suptitle(f"Candidate images — {label_column} — {population}\n{view}", fontsize=12)
    figure.colorbar(displayed, ax=axes.ravel().tolist(), shrink=.75, label="Intensity (a.u.)")
    return {"figure": figure, "selected_rows": selected, "counts": counts,
            "label_column": label_column, "partition": partition,
            "display_limits": [low, high], "processed": bool(processed)}


def candidate_image(results_directory, filename, *, processed=False, image_directory_override=None):
    """Return one original or processed image together with its saved labels."""
    if not isinstance(processed, (bool, np.bool_)):
        raise ValueError("processed must be a boolean.")
    _, config, metadata, _, records, by_index = _load_results(results_directory)
    matches = [row for row in records if row["filename"] == filename]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching filename; inspect state_assignments.csv for the exact name or duplicate basenames.")
    row = matches[0]
    image = _image_for_row(row, config, metadata, by_index, processed, image_directory_override)
    low, high = _limits([image])
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    shown = ax.imshow(image, cmap="gray", vmin=low, vmax=high, origin="upper")
    view = "processed pixels" if processed else "original sensor pixels"
    ax.set(title=f"{row['filename']} — assigned {row['state']}\n"
                 f"{float(row['time_s'])*1000:.3f} ms | {row['partition']}",
           xlabel=f"x ({view})", ylabel=f"y ({view})")
    figure.colorbar(shown, ax=ax, shrink=.8, label="Intensity (a.u.)")
    return {"figure": figure, "row": dict(row), "processed": bool(processed),
            "display_limits": [low, high]}


def export_candidate_lists(results_directory, output_directory, *, overwrite=False):
    """Write one complete CSV per saved candidate without copying any TIFFs.

    Existing outputs require ``overwrite=True``. Only this export's target
    files are replaced; unrelated files are retained. Source/alias checks run
    for every destination before any output is written.
    """
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean.")
    directory, config, _, columns, records, by_index = _load_results(results_directory)
    output = Path(output_directory).expanduser().resolve()
    if output == directory or output in directory.parents:
        raise ValueError("Choose a separate candidate-list output subfolder or sibling folder.")
    source_images = Path(config["image_directory"]).expanduser().resolve()
    if output == source_images or source_images in output.parents or output in source_images.parents:
        raise ValueError("Keep candidate-list outputs separate from the source image folder.")
    if output.exists() and not output.is_dir():
        raise ValueError("The candidate-list output path is not a directory.")
    if output.is_dir() and any(output.iterdir()) and not overwrite:
        raise FileExistsError("Candidate-list output is not empty; choose a new folder or explicitly enable overwrite.")
    counts = dict(sorted(Counter(row["state"] for row in records).items()))
    files, used = {}, set()
    for number, label in enumerate(counts, start=1):
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")[:80] or "candidate"
        if name.casefold() in used:
            name = f"{name}_{number}"
        used.add(name.casefold())
        files[label] = output/f"{name}.csv"
    manifest_path = output/"candidate_lists_manifest.json"
    destinations = [*files.values(), manifest_path]
    if len({path.name.casefold() for path in destinations}) != len(destinations):
        raise ValueError("Candidate labels produce conflicting output filenames.")
    protected = [path for path in directory.iterdir() if path.is_file()]
    protected += [Path(record["path"]).expanduser() for record in by_index.values()]
    protected += [source_images/row["filename"] for row in records]
    for destination in destinations:
        if any(_same_file(destination, source) for source in protected):
            raise ValueError("Candidate-list output must not overwrite a source image, saved result, or file alias.")
        if destination.exists() and (not destination.is_file() or not overwrite):
            raise FileExistsError(f"Candidate-list destination already exists: {destination}.")
    output.mkdir(parents=True, exist_ok=True)
    for label, destination in files.items():
        # A temporary neighbor plus replace avoids truncating a previous CSV if
        # serialization fails and does not follow a destination symlink.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig",
                                             dir=output, delete=False, suffix=".tmp") as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(row for row in records if row["state"] == label)
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    summary = {"output_directory": str(output), "files": {label:str(path) for label,path in files.items()},
               "counts": counts, "total_frames": len(records),
               "assignment_column": "state", "images_copied": False}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         delete=False, suffix=".tmp") as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(summary, indent=2, allow_nan=False))
        temporary.replace(manifest_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary
