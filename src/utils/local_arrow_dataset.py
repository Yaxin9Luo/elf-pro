"""Lightweight local Arrow-shard dataset.

The HF Hub copy of `embedded-language-flows/openwebtext-t5` ships
`dataset_info.json` with `features._type == "List"`, which only the newer
`datasets` (>=3.0) understands. On Python 3.9 we are stuck with `datasets<3`,
so reading via `load_from_disk` blows up. We bypass that by reading the IPC
streaming-format `*.arrow` shards directly with pyarrow, exposing just the
`__len__` / `__getitem__` / `column_names` / `set_format` surface that
`get_dataloader` and friends actually use.
"""

from __future__ import annotations

import bisect
import glob
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from torch.utils.data import Dataset


def _read_shard(path: str) -> pa.Table:
    with pa.memory_map(path, "r") as src:
        rd = ipc.open_stream(src)
        schema = rd.schema
        batches = list(rd)
    return pa.Table.from_batches(batches, schema=schema)


class LocalArrowDataset(Dataset):
    """A torch Dataset over a directory of HF-saved `.arrow` shards."""

    def __init__(self, root: str, columns: Optional[Sequence[str]] = None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"LocalArrowDataset root not found: {root}")
        self.root = root
        self.shard_paths = sorted(glob.glob(os.path.join(root, "*.arrow")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No .arrow shards under {root}")

        # Memory-mapped Tables: opening each shard is O(metadata), zero-copy.
        self._tables: List[pa.Table] = []
        self._row_offsets: List[int] = [0]
        for path in self.shard_paths:
            t = _read_shard(path)
            self._tables.append(t)
            self._row_offsets.append(self._row_offsets[-1] + t.num_rows)
        self._schema = self._tables[0].schema
        self._total = self._row_offsets[-1]

        all_cols = list(self._schema.names)
        self._columns = list(columns) if columns is not None else all_cols
        for c in self._columns:
            if c not in all_cols:
                raise KeyError(f"column {c!r} missing from dataset schema {all_cols}")

    @property
    def column_names(self) -> List[str]:
        return list(self._columns)

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int):
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        # bisect_right returns the offset boundary > idx; subtract one for shard idx
        sh = bisect.bisect_right(self._row_offsets, idx) - 1
        local = idx - self._row_offsets[sh]
        return sh, local

    def __getitem__(self, idx) -> Dict:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._total))]
        sh, local = self._locate(int(idx))
        table = self._tables[sh]
        out = {}
        for col in self._columns:
            arr = table.column(col)
            val = arr[local]
            if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
                out[col] = np.asarray(val.as_py(), dtype=np.int32)
            else:
                out[col] = val.as_py()
        return out

    # ------------------------------------------------------------------
    # `datasets`-compatible API surface needed by ELF's data_utils.
    # ------------------------------------------------------------------
    def set_format(self, type=None, columns=None, **kwargs):
        if columns is not None:
            self._columns = list(columns)
        # Type is always numpy/list — we already return native numpy below.
        return self

    def __repr__(self) -> str:
        return (
            f"LocalArrowDataset(root={self.root!r}, n={self._total}, "
            f"columns={self._columns})"
        )
