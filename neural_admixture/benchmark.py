"""
Benchmarking utilities for Neural ADMIXTURE.

Provides:
  - Wall-clock runtime measurement for training and inference
  - Peak memory tracking (GPU via torch.cuda, CPU via tracemalloc)
  - Formatted comparison tables
"""

import time
import tracemalloc
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

@contextmanager
def timer():
    """
    Context manager that measures wall-clock time.

    Usage::

        with timer() as t:
            train(...)
        print(t.elapsed)   # seconds as float
        print(t.formatted)  # "HH:MM:SS"
    """
    result = _TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed = time.perf_counter() - start
        result.formatted = _format_time(result.elapsed)


class _TimerResult:
    elapsed: float = 0.0
    formatted: str = "00:00:00"


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

@contextmanager
def track_memory(device: str = "cpu"):
    """
    Context manager that tracks peak memory usage.

    For CUDA: uses torch.cuda.max_memory_allocated.
    For CPU: uses tracemalloc.

    Usage::

        with track_memory("cuda") as mem:
            train(...)
        print(mem.peak_mb)
    """
    result = _MemoryResult()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        try:
            yield result
        finally:
            torch.cuda.synchronize()
            result.peak_bytes = torch.cuda.max_memory_allocated()
            result.peak_mb = result.peak_bytes / (1024 ** 2)
    else:
        tracemalloc.start()
        try:
            yield result
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result.peak_bytes = peak
            result.peak_mb = peak / (1024 ** 2)


class _MemoryResult:
    peak_bytes: int = 0
    peak_mb: float = 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_training(
    trainer_factory: Callable,
    X_train: np.ndarray,
    n_epochs: int = 50,
    devices: Optional[List[str]] = None,
    X_val: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Run training on each device and collect timing + memory stats.

    Parameters
    ----------
    trainer_factory : callable(device: str) -> Trainer.
        Should create a fresh model + Trainer on the given device.
    X_train : training data.
    n_epochs : epochs to train.
    devices : list of device strings to benchmark. Auto-detected if None.
    X_val : optional validation data.

    Returns
    -------
    results : list of dicts with keys:
        device, train_time_s, train_time_fmt, peak_memory_mb
    """
    if devices is None:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        elif torch.backends.mps.is_available():
            devices.append("mps")

    results = []
    for device in devices:
        trainer = trainer_factory(device)
        trainer.initialize_decoders(X_train)

        with timer() as t, track_memory(device) as mem:
            trainer.fit(X_train, n_epochs=n_epochs, X_val=X_val, verbose=False)

        results.append({
            "device": device,
            "train_time_s": t.elapsed,
            "train_time_fmt": t.formatted,
            "peak_memory_mb": mem.peak_mb,
        })

    return results


def benchmark_inference(
    trainer,
    X: np.ndarray,
    temperature: float = 1.0,
    n_runs: int = 5,
) -> Dict[str, float]:
    """
    Benchmark inference time (average over n_runs).

    Returns
    -------
    dict with keys: avg_time_s, avg_time_fmt, std_time_s
    """
    times = []
    for _ in range(n_runs):
        with timer() as t:
            trainer.predict(X, temperature=temperature)
        times.append(t.elapsed)

    avg = np.mean(times)
    std = np.std(times)
    return {
        "avg_time_s": avg,
        "avg_time_fmt": _format_time(avg),
        "std_time_s": std,
    }


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_results_table(
    results: List[Dict[str, Any]],
    dataset_name: str = "",
) -> str:
    """
    Format benchmark results as a readable table.

    Parameters
    ----------
    results : list of dicts from benchmark_training.
    dataset_name : label for the table header.

    Returns
    -------
    table : formatted string.
    """
    header = f"{'Dataset':<20} {'Device':<10} {'Train Time':<14} {'Peak Mem (MB)':<15}"
    sep = "-" * len(header)
    lines = [header, sep]

    for r in results:
        lines.append(
            f"{dataset_name:<20} {r['device']:<10} "
            f"{r['train_time_fmt']:<14} {r['peak_memory_mb']:<15.1f}"
        )

    return "\n".join(lines)
