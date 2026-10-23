"""Memory-bounded randomized POD fitted exclusively to designated frames.

The randomized range is spatial, requiring O(p * (r+oversampling)) memory, rather
than a dense n-by-n snapshot covariance or a full in-RAM image sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass
class PODModel:
    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    energy_fraction: np.ndarray
    total_variance_sum: float
    training_count: int
    method: str = "randomized spatial-range SVD"

    def transform(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.mean.size:
            raise ValueError("Image matrix has incompatible pixel dimensions")
        if np.iscomplexobj(X):
            raise ValueError("Image matrix must contain real intensities")
        batch_size = _integer(batch_size, "batch_size", 1)
        scores = np.empty((X.shape[0], self.modes.shape[0]), dtype=np.float64)
        for start in range(0, len(X), batch_size):
            block = np.asarray(X[start:start + batch_size], dtype=np.float64) - self.mean
            if not np.isfinite(block).all():
                raise ValueError("Non-finite image data encountered during POD projection")
            scores[start:start + len(block)] = block @ self.modes.T
        return scores

    def reconstruct(self, scores: np.ndarray, rank: int | None = None) -> np.ndarray:
        rank = self.modes.shape[0] if rank is None else _integer(rank, "rank", 1)
        if not 1 <= rank <= self.modes.shape[0]:
            raise ValueError("Requested reconstruction rank is unavailable")
        scores = np.asarray(scores)
        if scores.ndim not in (1, 2) or scores.shape[-1] < rank or np.iscomplexobj(scores) or not np.isfinite(scores).all():
            raise ValueError("Reconstruction scores must be finite real vectors with at least rank coefficients")
        return self.mean + scores[..., :rank] @ self.modes[:rank]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, mean=self.mean, modes=self.modes,
                            singular_values=self.singular_values,
                            energy_fraction=self.energy_fraction,
                            total_variance_sum=self.total_variance_sum,
                            training_count=self.training_count, method=self.method)


def load_pod(path: str | Path) -> PODModel:
    with np.load(path, allow_pickle=False) as saved:
        return PODModel(mean=saved["mean"], modes=saved["modes"],
                        singular_values=saved["singular_values"],
                        energy_fraction=saved["energy_fraction"],
                        total_variance_sum=float(saved["total_variance_sum"]),
                        training_count=int(saved["training_count"]),
                        method=str(saved["method"]))


def fit_pod(X: np.ndarray, train_indices: np.ndarray, rank: int = 12,
            batch_size: int = 64, oversampling: int = 8,
            power_iterations: int = 1, seed: int = 42) -> PODModel:
    """Fit a Euclidean/intensity POD; retained variance is divided by exact total.

    ``energy_fraction`` quantifies variance of schlieren image intensity, not
    kinetic energy. Pixel areas are equal after fixed square binning. Spatial
    mode signs are arbitrary but fixed deterministically for reproducibility.
    No test frame contributes to centering, the spatial basis, or mode ranking.
    """
    if X.ndim != 2:
        raise ValueError("POD expects an (n_frames,n_pixels) matrix")
    if np.iscomplexobj(X):
        raise ValueError("POD expects real image intensities")
    indices = np.asarray(train_indices)
    if indices.ndim != 1 or indices.size < 2 or indices.dtype.kind not in "iu":
        raise ValueError("train_indices must contain at least two integer frame indices")
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(X):
        raise ValueError("Training indices must be unique and in bounds")
    max_rank = min(len(indices) - 1, X.shape[1])
    rank = _integer(rank, "rank", 1)
    if rank > max_rank:
        raise ValueError(f"rank must be an integer from 1 to {max_rank}")
    batch_size = _integer(batch_size, "batch_size", 1)
    oversampling = _integer(oversampling, "oversampling")
    power_iterations = _integer(power_iterations, "power_iterations")
    n_train, n_pixels = len(indices), X.shape[1]
    sketch_size = min(rank + oversampling, max_rank)
    mean = np.zeros(n_pixels, dtype=np.float64)
    for start in range(0, n_train, batch_size):
        block = np.asarray(X[indices[start:start + batch_size]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Training images contain non-finite data")
        mean += block.sum(axis=0)
    mean /= n_train

    generator = np.random.default_rng(seed)
    omega = generator.standard_normal((n_train, sketch_size))
    spatial_range = np.zeros((n_pixels, sketch_size), dtype=np.float64)
    total_variance_sum = 0.0
    for start in range(0, n_train, batch_size):
        stop = min(start + batch_size, n_train)
        block = np.asarray(X[indices[start:stop]], dtype=np.float64) - mean
        spatial_range += block.T @ omega[start:stop]
        total_variance_sum += float(np.einsum("ij,ij->", block, block))
    del omega
    if total_variance_sum <= 0 or not np.isfinite(total_variance_sum):
        raise ValueError("Training images contain no finite temporal intensity variation")
    q, _ = np.linalg.qr(spatial_range, mode="reduced")
    del spatial_range
    for _ in range(power_iterations):
        # Reorthogonalize both sides of the power iteration. Forming A.T A Q
        # directly squares the spectral conditioning and can destroy weak
        # resolved directions before the following spatial QR.
        temporal_range = np.empty((n_train, sketch_size), dtype=np.float64)
        for start in range(0, n_train, batch_size):
            stop = min(start + batch_size, n_train)
            block = np.asarray(X[indices[start:stop]], dtype=np.float64) - mean
            temporal_range[start:stop] = block @ q
        temporal_basis, _ = np.linalg.qr(temporal_range, mode="reduced")
        del temporal_range
        spatial_range = np.zeros_like(q)
        for start in range(0, n_train, batch_size):
            stop = min(start + batch_size, n_train)
            block = np.asarray(X[indices[start:stop]], dtype=np.float64) - mean
            spatial_range += block.T @ temporal_basis[start:stop]
        del temporal_basis
        q, _ = np.linalg.qr(spatial_range, mode="reduced")
        del spatial_range

    compressed = np.empty((sketch_size, n_train), dtype=np.float64)
    for start in range(0, n_train, batch_size):
        stop = min(start + batch_size, n_train)
        block = np.asarray(X[indices[start:stop]], dtype=np.float64) - mean
        compressed[:, start:stop] = q.T @ block.T
    spatial_rotation, singular_values, _ = np.linalg.svd(compressed, full_matrices=False)
    modes = (q @ spatial_rotation[:, :rank]).T
    singular_values = singular_values[:rank]
    # Fix signs to place each mode's largest-magnitude entry on its positive side.
    pivots = np.argmax(np.abs(modes), axis=1)
    signs = np.sign(modes[np.arange(rank), pivots])
    modes *= np.where(signs == 0, 1, signs)[:, None]
    return PODModel(mean=mean, modes=modes, singular_values=singular_values,
                    energy_fraction=(singular_values / np.sqrt(total_variance_sum)) ** 2,
                    total_variance_sum=total_variance_sum, training_count=n_train,
                    method=f"randomized spatial-range SVD; stabilized two-sided QR; oversampling={oversampling}; power_iterations={power_iterations}; seed={seed}")
