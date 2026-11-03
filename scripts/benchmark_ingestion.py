"""Compare ingestion time, traced allocations, and peak resident memory."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import gc
import json
import os
import sys
import tempfile
import threading
import time
import tracemalloc
from typing import Callable, Dict, Tuple

import awkward as ak
import h5py
import numpy as np
import pandas as pd
import uproot

try:
    import resource
except ImportError:
    resource = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phasespace_onnx.data.columnar_loader import ColumnarPhaseSpaceLoader
from phasespace_onnx.data.generator import OBSERVABLE_NAMES, generate_synthetic_phasespace

MIB = 1024.0 * 1024.0


def _current_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        success = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not success:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)

    try:
        with open("/proc/self/statm", "r", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, IndexError, ValueError):
        if resource is None:
            raise RuntimeError("RSS measurement is unavailable on this system")
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024


class PeakRSS:
    """Sample current process RSS during a measured operation."""

    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = interval_seconds
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.baseline = _current_rss_bytes()
        self.peak = self.baseline

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak = max(self.peak, _current_rss_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak = max(self.peak, _current_rss_bytes())
        return max(0, self.peak - self.baseline) / MIB


def _measure(operation: Callable[[], int]) -> Dict[str, float | int]:
    gc.collect()
    tracemalloc.start()
    rss = PeakRSS()
    rss.start()
    started = time.perf_counter()
    try:
        events = int(operation())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_delta = rss.stop()
    return {
        "num_events": events,
        "execution_time_ms": elapsed_ms,
        "peak_tracemalloc_mb": traced_peak / MIB,
        "peak_rss_delta_mb": rss_delta,
        "throughput_eps": events / max(elapsed_ms / 1000.0, 1e-12),
    }


def benchmark_csv_ingestion(csv_path: str) -> Dict[str, float | int]:
    def consume() -> int:
        frame = pd.read_csv(csv_path)
        jets = frame["jets"].str.strip("[]").str.split(",", expand=True).astype(np.float32)
        frame["lead_jet_pt"] = jets[0].fillna(0.0)
        frame["sublead_jet_pt"] = jets[1].fillna(0.0)
        columns = [
            "lead_lepton_pt",
            "sublead_lepton_pt",
            "lead_lepton_eta",
            "invariant_mass_ll",
            "missing_et",
            "lead_jet_pt",
            "sublead_jet_pt",
            "delta_r_ll",
            "scalar_ht",
            "pt_balance_ratio",
        ]
        features = frame[columns].to_numpy(dtype=np.float32)
        labels = frame["is_variation"].to_numpy(dtype=np.int32)
        return int(len(features) if len(features) == len(labels) else 0)

    return _measure(consume)


def benchmark_monolithic_root(root_path: str) -> Dict[str, float | int]:
    def consume() -> int:
        with uproot.open(root_path) as root_file:
            data = root_file["PhaseSpaceTree"].arrays(library="ak")
        sorted_jets = data["jet_pt"][ak.argsort(data["jet_pt"], axis=1, ascending=False)]
        padded = ak.fill_none(ak.pad_none(sorted_jets, 2, axis=1), 0.0)
        columns = []
        for name in OBSERVABLE_NAMES:
            if name == "lead_jet_pt":
                columns.append(ak.to_numpy(padded[:, 0]).astype(np.float32))
            elif name == "sublead_jet_pt":
                columns.append(ak.to_numpy(padded[:, 1]).astype(np.float32))
            else:
                columns.append(ak.to_numpy(data[name]).astype(np.float32))
        features = np.column_stack(columns)
        labels = ak.to_numpy(data["is_variation"])
        return int(len(features) if len(features) == len(labels) else 0)

    return _measure(consume)


def benchmark_root_streaming(root_path: str, step_size: str = "2 MB") -> Dict[str, float | int]:
    loader = ColumnarPhaseSpaceLoader(observable_names=OBSERVABLE_NAMES)

    def consume() -> int:
        events = 0
        for features, labels in loader.stream_root_chunks(root_path, step_size=step_size):
            events += len(labels)
            _ = float(np.sum(features[:, 0]))
        return events

    return _measure(consume)


def benchmark_hdf5_streaming(h5_path: str, chunk_size: int = 10000) -> Dict[str, float | int]:
    loader = ColumnarPhaseSpaceLoader(observable_names=OBSERVABLE_NAMES)

    def consume() -> int:
        events = 0
        for features, labels in loader.stream_hdf5_chunks(h5_path, chunk_size=chunk_size):
            events += len(labels)
            _ = float(np.sum(features[:, 0]))
        return events

    return _measure(consume)


def _write_csv(root_path: str, csv_path: str) -> int:
    with uproot.open(root_path) as root_file:
        data = root_file["PhaseSpaceTree"].arrays(library="ak")
    count = len(data)
    with open(csv_path, "w", encoding="utf-8") as output:
        output.write(
            "lead_lepton_pt,sublead_lepton_pt,lead_lepton_eta,invariant_mass_ll,"
            "missing_et,jets,delta_r_ll,scalar_ht,pt_balance_ratio,is_variation\n"
        )
        for index in range(count):
            jets = "[" + ",".join(f"{value:.1f}" for value in data["jet_pt"][index].to_list()) + "]"
            values = [
                f"{float(data['lead_lepton_pt'][index]):.3f}",
                f"{float(data['sublead_lepton_pt'][index]):.3f}",
                f"{float(data['lead_lepton_eta'][index]):.3f}",
                f"{float(data['invariant_mass_ll'][index]):.3f}",
                f"{float(data['missing_et'][index]):.3f}",
                f'"{jets}"',
                f"{float(data['delta_r_ll'][index]):.3f}",
                f"{float(data['scalar_ht'][index]):.3f}",
                f"{float(data['pt_balance_ratio'][index]):.3f}",
                str(int(data["is_variation"][index])),
            ]
            output.write(",".join(values) + "\n")
    return count


def _ensure_samples(root_path: str, h5_path: str) -> None:
    hdf5_is_ready = False
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, "r") as sample:
                group = sample["kinematics"]
                hdf5_is_ready = "jet_pt_flat" in group and "jet_pt_offsets" in group
        except (OSError, KeyError):
            hdf5_is_ready = False
    if not os.path.exists(root_path) or not hdf5_is_ready:
        print("Generating 100,000 synthetic events for the ingestion comparison...")
        generate_synthetic_phasespace(root_path, h5_path, n_per_class=50000, seed=42)


def main() -> None:
    print("=" * 75)
    print("PHASESPACE-ONNX: DATA INGESTION MEMORY & THROUGHPUT COMPARISON")
    print("=" * 75)
    data_dir = os.path.join(PROJECT_ROOT, "data")
    assets_dir = os.path.join(PROJECT_ROOT, "assets")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)
    root_path = os.path.join(data_dir, "mc_phasespace.root")
    h5_path = os.path.join(data_dir, "mc_phasespace.h5")
    _ensure_samples(root_path, h5_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temporary_csv:
        csv_path = temporary_csv.name
    try:
        events_written = _write_csv(root_path, csv_path)
        print(f"Created temporary CSV for {events_written:,} events.")
        results = {
            "num_events": events_written,
            "csv_flattened": benchmark_csv_ingestion(csv_path),
            "monolithic_root": benchmark_monolithic_root(root_path),
            "awkward_root_streaming": {
                **benchmark_root_streaming(root_path, step_size="2 MB"),
                "chunk_step_size": "2 MB",
            },
            "hdf5_jagged_streaming": {
                **benchmark_hdf5_streaming(h5_path, chunk_size=10000),
                "chunk_size": 10000,
            },
        }
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    baseline = float(results["csv_flattened"]["peak_rss_delta_mb"])
    for name, values in results.items():
        if isinstance(values, dict) and "peak_rss_delta_mb" in values:
            print(
                f"{name:<27} {values['execution_time_ms']:>9.2f} ms | "
                f"tracemalloc {values['peak_tracemalloc_mb']:>8.2f} MB | "
                f"RSS delta {values['peak_rss_delta_mb']:>8.2f} MB"
            )
    results["rss_reduction_vs_csv_percent"] = {
        key: (1.0 - float(results[key]["peak_rss_delta_mb"]) / max(baseline, 1e-9)) * 100.0
        for key in ("monolithic_root", "awkward_root_streaming", "hdf5_jagged_streaming")
    }

    result_path = os.path.join(assets_dir, "ingestion_memory_comparison.json")
    with open(result_path, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print(f"Saved ingestion measurements: {result_path}")


if __name__ == "__main__":
    main()
