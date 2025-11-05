"""Unified interface for chord caches."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import abs_mask_bigint_of, struct_id_of


@dataclass
class StoredChord:
    chord_id: int
    midi: Tuple[int, ...]
    meta: Dict[str, object]


class _BaseBackend:
    def get_or_add_abs(self, midi_list: Sequence[int], meta: Dict[str, object]) -> StoredChord:
        raise NotImplementedError

    def has_abs_mask(self, mask: int) -> bool:
        raise NotImplementedError

    def get_by_struct_id(self, struct_id: Tuple[int, Tuple[int, ...]]) -> List[StoredChord]:
        raise NotImplementedError


class _MemoryBackend(_BaseBackend):
    def __init__(self) -> None:
        self._records: Dict[int, StoredChord] = {}
        self._by_abs: Dict[int, int] = {}
        self._by_struct: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        self._next_id = 1

    def get_or_add_abs(self, midi_list: Sequence[int], meta: Dict[str, object]) -> StoredChord:
        mask = abs_mask_bigint_of(midi_list)
        struct = struct_id_of(midi_list).as_key()
        if mask in self._by_abs:
            chord_id = self._by_abs[mask]
            return self._records[chord_id]
        chord_id = self._next_id
        self._next_id += 1
        merged_meta = dict(meta)
        merged_meta.setdefault("abs_mask_bigint", mask)
        merged_meta.setdefault("struct_id", struct)
        record = StoredChord(chord_id, tuple(int(m) for m in midi_list), merged_meta)
        self._records[chord_id] = record
        self._by_abs[mask] = chord_id
        self._by_struct.setdefault(struct, []).append(chord_id)
        return record

    def has_abs_mask(self, mask: int) -> bool:
        return mask in self._by_abs

    def get_by_struct_id(self, struct_id: Tuple[int, Tuple[int, ...]]) -> List[StoredChord]:
        ids = self._by_struct.get(struct_id, [])
        return [self._records[i] for i in ids]


class _SqliteBackend(_BaseBackend):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                abs_mask_bigint TEXT UNIQUE,
                struct_id TEXT,
                payload TEXT
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chords_struct ON chords(struct_id)")
        self._conn.commit()

    def get_or_add_abs(self, midi_list: Sequence[int], meta: Dict[str, object]) -> StoredChord:
        mask = abs_mask_bigint_of(midi_list)
        mask_str = str(mask)
        struct = struct_id_of(midi_list).as_key()
        cur = self._conn.cursor()
        cur.execute("SELECT id, payload FROM chords WHERE abs_mask_bigint = ?", (mask_str,))
        row = cur.fetchone()
        if row:
            chord_id, payload = row
            stored = json.loads(payload)
            meta = _load_meta(stored["meta"])
            return StoredChord(chord_id, tuple(stored["midi"]), meta)
        merged_meta = dict(meta)
        merged_meta.setdefault("abs_mask_bigint", mask)
        merged_meta.setdefault("struct_id", struct)
        payload = json.dumps(
            {"midi": list(midi_list), "meta": _dumpable_meta(merged_meta)},
            default=_json_default,
        )
        cur.execute(
            "INSERT INTO chords(abs_mask_bigint, struct_id, payload) VALUES (?,?,?)",
            (mask_str, json.dumps(struct), payload),
        )
        chord_id = cur.lastrowid
        self._conn.commit()
        return StoredChord(chord_id, tuple(int(m) for m in midi_list), merged_meta)

    def has_abs_mask(self, mask: int) -> bool:
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM chords WHERE abs_mask_bigint = ?", (str(mask),))
        return cur.fetchone() is not None

    def get_by_struct_id(self, struct_id: Tuple[int, Tuple[int, ...]]) -> List[StoredChord]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, payload FROM chords WHERE struct_id = ?",
            (json.dumps(struct_id),),
        )
        records: List[StoredChord] = []
        for chord_id, payload in cur.fetchall():
            stored = json.loads(payload)
            meta = _load_meta(stored["meta"])
            records.append(StoredChord(chord_id, tuple(stored["midi"]), meta))
        return records


class ChordStore:
    """Facade to work with the supported chord cache backends."""

    def __init__(self, backend: str = "memory", *, path: Optional[str] = None) -> None:
        backend = backend.lower()
        if backend == "memory":
            self._backend: _BaseBackend = _MemoryBackend()
        elif backend == "sqlite":
            if path is None:
                raise ValueError("SQLite backend requires a path")
            self._backend = _SqliteBackend(Path(path))
        elif backend == "parquet":
            raise NotImplementedError("Parquet backend is not implemented yet")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def get_or_add_abs(self, midi_list: Sequence[int], meta: Dict[str, object]) -> StoredChord:
        return self._backend.get_or_add_abs(midi_list, meta)

    def has_abs_mask(self, mask: int) -> bool:
        return self._backend.has_abs_mask(mask)

    def get_by_struct_id(self, struct_id: Tuple[int, Tuple[int, ...]]) -> List[StoredChord]:
        return self._backend.get_by_struct_id(struct_id)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")


def _dumpable_meta(meta: Dict[str, object]) -> Dict[str, object]:
    return {k: _json_default(v) if isinstance(v, (np.ndarray, set, tuple)) else v for k, v in meta.items()}


def _load_meta(meta: Dict[str, object]) -> Dict[str, object]:
    loaded: Dict[str, object] = {}
    for key, value in meta.items():
        if key in {"struct_id", "octave_vector", "pc_tuple_canon0"} and isinstance(value, list):
            if key == "struct_id" and len(value) == 2:
                loaded[key] = (int(value[0]), tuple(int(v) for v in value[1]))
            else:
                loaded[key] = tuple(value)
        else:
            loaded[key] = value
    return loaded
