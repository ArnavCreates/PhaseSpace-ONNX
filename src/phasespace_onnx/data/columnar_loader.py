"""Read ROOT and HDF5 phase-space data in bounded chunks."""

from __future__ import annotations

from typing import Generator, List, Tuple

import awkward as ak
import h5py
import numpy as np
import uproot

from .generator import OBSERVABLE_NAMES


class ColumnarPhaseSpaceLoader:
    """Load event observables and variable-length jets from columnar files."""

    def __init__(self, observable_names: List[str] | None = None) -> None:
        self.observable_names = observable_names or list(OBSERVABLE_NAMES)

    @staticmethod
    def _numpy_view(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Return a view when the input already has the requested dtype."""
        try:
            return np.asarray(array, dtype=dtype, copy=False)
        except TypeError:
            return np.asarray(array, dtype=dtype)

    @staticmethod
    def _leading_jet_features(jet_pt: ak.Array) -> Tuple[np.ndarray, np.ndarray]:
        sorted_jets = jet_pt[ak.argsort(jet_pt, axis=1, ascending=False)]
        padded_jets = ak.fill_none(ak.pad_none(sorted_jets, 2, axis=1), 0.0)
        lead = np.asarray(ak.to_numpy(padded_jets[:, 0]), dtype=np.float32)
        sublead = np.asarray(ak.to_numpy(padded_jets[:, 1]), dtype=np.float32)
        return lead, sublead

    def stream_root_chunks(
        self,
        root_path: str,
        tree_name: str = "PhaseSpaceTree",
        step_size: str = "2 MB",
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield contiguous feature and label arrays from successive ROOT chunks."""
        query = f"{root_path}:{tree_name}"
        branches = [
            "lead_lepton_pt",
            "sublead_lepton_pt",
            "lead_lepton_eta",
            "invariant_mass_ll",
            "missing_et",
            "jet_pt",
            "delta_r_ll",
            "scalar_ht",
            "pt_balance_ratio",
            "is_variation",
        ]

        for chunk in uproot.iterate(query, expressions=branches, library="ak", step_size=step_size):
            lead_jet_pt, sublead_jet_pt = self._leading_jet_features(chunk["jet_pt"])
            event_columns = [
                ak.to_numpy(chunk["lead_lepton_pt"]).astype(np.float32),
                ak.to_numpy(chunk["sublead_lepton_pt"]).astype(np.float32),
                ak.to_numpy(chunk["lead_lepton_eta"]).astype(np.float32),
                ak.to_numpy(chunk["invariant_mass_ll"]).astype(np.float32),
                ak.to_numpy(chunk["missing_et"]).astype(np.float32),
                lead_jet_pt,
                sublead_jet_pt,
                ak.to_numpy(chunk["delta_r_ll"]).astype(np.float32),
                ak.to_numpy(chunk["scalar_ht"]).astype(np.float32),
                ak.to_numpy(chunk["pt_balance_ratio"]).astype(np.float32),
            ]
            features = np.ascontiguousarray(np.column_stack(event_columns), dtype=np.float32)
            labels = np.asarray(ak.to_numpy(chunk["is_variation"]), dtype=np.int32)
            yield features, labels

    def load_from_root(
        self,
        root_path: str,
        tree_name: str = "PhaseSpaceTree",
        step_size: str = "2 MB",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read all ROOT chunks and concatenate them into one matrix."""
        x_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []
        for x_chunk, y_chunk in self.stream_root_chunks(root_path, tree_name, step_size):
            x_chunks.append(x_chunk)
            y_chunks.append(y_chunk)
        if not x_chunks:
            return np.empty((0, len(self.observable_names)), dtype=np.float32), np.empty(0, dtype=np.int32)
        return (
            np.ascontiguousarray(np.concatenate(x_chunks, axis=0), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(y_chunks, axis=0), dtype=np.int32),
        )

    def stream_hdf5_chunks(
        self,
        h5_path: str,
        chunk_size: int = 10000,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yield feature chunks while rebuilding jets from flattened values and offsets."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        with h5py.File(h5_path, "r") as h5_file:
            group = h5_file["kinematics"]
            feat_dset = group["features"]
            lbl_dset = group["labels"]
            if "jet_pt_flat" not in group or "jet_pt_offsets" not in group:
                raise ValueError("HDF5 input must contain kinematics/jet_pt_flat and kinematics/jet_pt_offsets")
            jet_flat_dset = group["jet_pt_flat"]
            jet_offsets_dset = group["jet_pt_offsets"]

            total_events = int(feat_dset.shape[0])
            if int(jet_offsets_dset.shape[0]) != total_events + 1:
                raise ValueError("jet_pt_offsets must contain one more entry than the event count")

            for start in range(0, total_events, chunk_size):
                end = min(start + chunk_size, total_events)
                global_offsets = self._numpy_view(
                    jet_offsets_dset[start : end + 1], np.dtype(np.int64)
                )
                flat_start = int(global_offsets[0])
                flat_end = int(global_offsets[-1])
                local_offsets = np.ascontiguousarray(global_offsets - flat_start, dtype=np.int64)
                local_flat = self._numpy_view(
                    jet_flat_dset[flat_start:flat_end], np.dtype(np.float32)
                )

                jagged_jets = ak.Array(
                    ak.contents.ListOffsetArray(
                        ak.index.Index64(local_offsets),
                        ak.contents.NumpyArray(local_flat),
                    )
                )
                lead_jet_pt, sublead_jet_pt = self._leading_jet_features(jagged_jets)

                features = self._numpy_view(feat_dset[start:end], np.dtype(np.float32))
                features[:, 5] = lead_jet_pt
                features[:, 6] = sublead_jet_pt
                labels = self._numpy_view(lbl_dset[start:end], np.dtype(np.int32))
                yield features, labels

    def load_from_hdf5(
        self,
        h5_path: str,
        chunk_size: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read all HDF5 chunks and concatenate them into one matrix."""
        x_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []
        for x_chunk, y_chunk in self.stream_hdf5_chunks(h5_path, chunk_size):
            x_chunks.append(x_chunk)
            y_chunks.append(y_chunk)
        if not x_chunks:
            return np.empty((0, len(self.observable_names)), dtype=np.float32), np.empty(0, dtype=np.int32)
        return (
            np.ascontiguousarray(np.concatenate(x_chunks, axis=0), dtype=np.float32),
            np.ascontiguousarray(np.concatenate(y_chunks, axis=0), dtype=np.int32),
        )
