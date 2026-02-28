"""
Herramientas sintéticas para generar acordes compatibles con la DB (inversiones, transposiciones)
y preparar datos para experimentos (vectores Sethares + MDS) sin persistir en la base.

Principios:
- Mantener exactamente las columnas/formatos usados por la tabla `chords`.
- Reutilizar el cálculo canónico (`calculate_row`) de chordcodex si está disponible.
- Fallback embebido si chordcodex no está accesible en el entorno actual.

Salidas principales (registros sintéticos):
    id, n, interval, notes, bass, octave, frequencies, chroma, tag, code,
    span_semitones, abs_mask_int, abs_mask_hex, notes_abs_json
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Cargar utilidades DB/cálculo canónico
# ------------------------------------------------------------

try:
    # Intentar usar el paquete instalado
    from chordcodex.scripts.db_fill_v2 import calculate_row, TAG as DB_TAG  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback si el paquete no expone scripts
    import importlib.util

    DB_TAG = "ABS_V2"

    def _load_db_fill_v2() -> Tuple[Any, str]:
        base_dir = Path(__file__).resolve().parent
        venv_dir = Path(sys.prefix)
        candidates = [
            base_dir / "chordcodex" / "scripts" / "db_fill_v2.py",
            base_dir / "venv" / "Lib" / "site-packages" / "chordcodex" / "scripts" / "db_fill_v2.py",
            venv_dir / "Lib" / "site-packages" / "chordcodex" / "scripts" / "db_fill_v2.py",
            venv_dir / "site-packages" / "chordcodex" / "scripts" / "db_fill_v2.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("_db_fill_v2", candidate)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore[attr-defined]
                    return module.calculate_row, getattr(module, "TAG", DB_TAG)

        # Fallback embebido (réplica lógica mínima y fiel)
        NOTE_FREQUENCIES = {
            0: 261.63, 1: 277.18, 2: 293.66, 3: 311.13, 4: 329.63, 5: 349.23,
            6: 369.99, 7: 391.99, 8: 415.30, 9: 440.00, 10: 466.16, 11: 493.88,
        }
        HEX12 = "0123456789AB"

        def _embedded_calculate_row(notes_abs: Sequence[int]):
            notes_abs = list(notes_abs)
            notes_abs.sort()
            n = len(notes_abs)
            span_semitones = notes_abs[-1] - notes_abs[0] if n > 1 else 0
            code = ''.join(HEX12[note % 12] for note in notes_abs)
            abs_mask_int = 0
            for note in notes_abs:
                abs_mask_int |= (1 << note)
            abs_mask_hex = format(abs_mask_int, '07X')
            notes_abs_json = json.dumps(notes_abs)
            pitch_classes = [str(note % 12) for note in notes_abs]
            bass = str(notes_abs[0] % 12) if notes_abs else '0'
            intervals = [(notes_abs[i + 1] - notes_abs[i]) for i in range(n - 1)] if n > 1 else []
            octaves = [4 + (note // 12) for note in notes_abs]
            frequencies = [NOTE_FREQUENCIES[note % 12] * (2 ** ((4 + (note // 12)) - 4)) for note in notes_abs]
            chroma = [0] * 12
            for note in notes_abs:
                chroma[note % 12] = 1
            return (
                n,
                intervals,
                pitch_classes,
                bass,
                octaves[0] if octaves else 4,
                frequencies,
                chroma,
                DB_TAG,
                code,
                span_semitones,
                abs_mask_int,
                abs_mask_hex,
                notes_abs_json,
            )

        return _embedded_calculate_row, DB_TAG

    calculate_row, DB_TAG = _load_db_fill_v2()

# QueryExecutor: usar el del paquete si está; sino, fallback liviano
try:
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover

    class QueryExecutor:  # type: ignore
        """Fallback liviano: importa psycopg2 solo al ejecutar consultas."""
        def __init__(self, **config: Any):
            self.config = config

        def as_pandas(self, query: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame:
            try:
                import psycopg2  # type: ignore
                from psycopg2.extras import RealDictCursor  # type: ignore
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "psycopg2 no está instalado. Instálalo con "
                    "'pip install psycopg2-binary' si necesitas acceso a la DB."
                ) from None
            params = params or ()
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall() if cursor.description else []
            return pd.DataFrame(rows)

    model_module = types.ModuleType("chordcodex.model")
    model_module.QueryExecutor = QueryExecutor
    package_module = types.ModuleType("chordcodex")
    package_module.model = model_module
    sys.modules.setdefault("chordcodex", package_module)
    sys.modules.setdefault("chordcodex.model", model_module)


# ------------------------------------------------------------
# Utilidades de parsing y construcción de registro DB-like
# ------------------------------------------------------------

def _parse_pg_array(value: Any) -> List[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip("{}[]")
        if not stripped:
            return []
        return [part.strip().strip('"') for part in stripped.split(',') if part.strip()]
    raise TypeError(f"Unsupported array format: {type(value)!r}")


def _to_int_list(value: Any) -> List[int]:
    return [int(x) for x in _parse_pg_array(value)]


def _extract_notes_abs(row: Dict[str, Any]) -> List[int]:
    raw_json = row.get("notes_abs_json")
    if isinstance(raw_json, str) and raw_json:
        try:
            parsed = json.loads(raw_json)
            return [int(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    # Anclados en 0: reconstruir por intervalos
    intervals = _to_int_list(row.get("interval", []))
    notes = [0]
    for step in intervals:
        notes.append(notes[-1] + int(step))
    return notes


def _build_record_from_notes(notes_abs: Sequence[int], tag: Optional[str] = None) -> Dict[str, Any]:
    (
        n,
        intervals,
        pitch_classes,
        bass,
        octave,
        frequencies,
        chroma,
        default_tag,
        code,
        span_semitones,
        abs_mask_int,
        abs_mask_hex,
        notes_abs_json,
    ) = calculate_row(tuple(sorted(int(n) for n in notes_abs)))
    return {
        "id": None,
        "n": n,
        "interval": intervals,
        "notes": pitch_classes,
        "bass": bass,
        "octave": octave,
        "frequencies": frequencies,
        "chroma": chroma,
        "tag": tag if tag is not None else default_tag,
        "code": code,
        "span_semitones": span_semitones,
        "abs_mask_int": abs_mask_int,
        "abs_mask_hex": abs_mask_hex,
        "notes_abs_json": notes_abs_json,
    }


def _rotate_once(notes_abs: Sequence[int], k: int) -> List[int]:
    """Mueve k notas del frente al final, sumando +12 a las movidas."""
    if k <= 0:
        return list(notes_abs)
    head = [note + 12 for note in notes_abs[:k]]
    tail = list(notes_abs[k:])
    return tail + head


def invert_row(
    row: Dict[str, Any],
    tag: Optional[str] = None,
    *,
    max_abs: Optional[int] = 24,
    include_rotation: bool = False,
    warn_on_skip: bool = True,
) -> List[Dict[str, Any]]:
    """Genera registros para todas las inversiones de ``row``.

    Parameters
    ----------
    row:
        Fila con columnas compatibles con la tabla ``chords``.
    tag:
        Etiqueta opcional para sobrescribir el tag de ``calculate_row``.
    max_abs:
        Límite superior para las alturas absolutas permitidas. Usa ``None`` para
        desactivar la restricción (permite alturas > 24).
    include_rotation:
        Si es ``True`` añade ``__inv_rotation`` al registro generado.
    warn_on_skip:
        Controla si se imprime una advertencia cuando una inversión queda fuera
        del rango permitido y se omite.
    """

    notes_abs = _extract_notes_abs(row)
    out: List[Dict[str, Any]] = []
    for k in range(1, len(notes_abs)):
        rotated = _rotate_once(notes_abs, k)
        if max_abs is not None and ((min(rotated) < 0) or (max(rotated) > max_abs)):
            if warn_on_skip:
                print(
                    "Advertencia: se omite inversión "
                    f"k={k} (id={row.get('id')}, code={row.get('code')}) "
                    f"por salir de [0,{max_abs}]: {rotated}"
                )
            continue
        record = _build_record_from_notes(rotated, tag)
        if include_rotation:
            record["__inv_rotation"] = k
        out.append(record)
    return out


def transpose_row(row: Dict[str, Any], semitones: int, tag: Optional[str] = None) -> Dict[str, Any]:
    notes_abs = _extract_notes_abs(row)
    transposed = [int(n) + int(semitones) for n in notes_abs]
    if (min(transposed) < 0) or (max(transposed) > 24):
        raise ValueError(
            f"Transposición fuera de rango [0,24] (id={row.get('id')}, code={row.get('code')}): {transposed}"
        )
    return _build_record_from_notes(transposed, tag)


def make_inversions_df(
    df_source: pd.DataFrame,
    tag: Optional[str] = None,
    include_original: bool = True,
    *,
    allow_out_of_range: bool = False,
) -> pd.DataFrame:
    """Genera un ``DataFrame`` con inversiones sintéticas.

    Además de las columnas básicas compatibles con ``chords`` añade metadatos
    que permiten rastrear familias de inversiones (``__family_id`` y
    ``__family_size``) y marcar qué filas son generadas (``__inv_flag``).
    """

    records: List[Dict[str, Any]] = []
    family_tracker: List[Optional[int]] = []
    fallback_family_id = -1

    base_cols = [
        "id",
        "n",
        "interval",
        "notes",
        "bass",
        "octave",
        "frequencies",
        "chroma",
        "tag",
        "code",
        "span_semitones",
        "abs_mask_int",
        "abs_mask_hex",
        "notes_abs_json",
    ]
    meta_cols = [
        "__source__",
        "__inv_flag",
        "__inv_rotation",
        "__inv_source_id",
        "__family_id",
        "__family_size",
    ]

    for idx, row in df_source.iterrows():
        rowd = row.to_dict()

        raw_id = rowd.get("id")
        source_id: Optional[int]
        if raw_id is not None and not pd.isna(raw_id):
            try:
                source_id = int(raw_id)
            except Exception:
                source_id = None
        else:
            source_id = None

        family_id: Optional[int] = source_id
        if family_id is None:
            mask_val = rowd.get("abs_mask_int")
            if mask_val is not None and not pd.isna(mask_val):
                try:
                    family_id = int(mask_val)
                except Exception:
                    family_id = None
        if family_id is None:
            family_id = fallback_family_id
            fallback_family_id -= 1

        source_label = rowd.get("__source__")
        generated_label = "C:generated"
        if isinstance(source_label, str) and source_label.strip():
            generated_label = source_label

        inv_source_id = source_id if source_id is not None else np.nan

        if include_original:
            base = {key: rowd.get(key) for key in base_cols}
            if isinstance(base.get("interval"), str):
                base["interval"] = _to_int_list(base["interval"])
            if isinstance(base.get("notes"), str):
                base["notes"] = [str(x) for x in _parse_pg_array(base.get("notes"))]
            base["id"] = source_id
            base["__source__"] = source_label
            base["__inv_flag"] = False
            base["__inv_rotation"] = np.nan
            base["__inv_source_id"] = inv_source_id
            base["__family_id"] = family_id
            records.append(base)
            family_tracker.append(family_id)

        inv_records = invert_row(
            rowd,
            tag,
            max_abs=None if allow_out_of_range else 24,
            include_rotation=True,
            warn_on_skip=not allow_out_of_range,
        )

        for rec in inv_records:
            rec.setdefault("tag", tag if tag is not None else rec.get("tag"))
            rec.setdefault("__inv_rotation", rec.get("__inv_rotation"))
            rec["__source__"] = generated_label
            rec["__inv_flag"] = True
            rec["__inv_source_id"] = inv_source_id
            rec["__family_id"] = family_id
            records.append(rec)
            family_tracker.append(family_id)

    if not records:
        return pd.DataFrame(columns=base_cols + meta_cols)

    family_counts: Dict[Optional[int], int] = {}
    for fid in family_tracker:
        if fid is None or (isinstance(fid, float) and np.isnan(fid)):
            continue
        family_counts[fid] = family_counts.get(fid, 0) + 1

    for rec, fid in zip(records, family_tracker):
        if fid is None or (isinstance(fid, float) and np.isnan(fid)):
            rec["__family_size"] = np.nan
        else:
            rec["__family_size"] = family_counts.get(fid, 1)

        if "__inv_flag" not in rec:
            rec["__inv_flag"] = False
        if "__inv_rotation" not in rec:
            rec["__inv_rotation"] = np.nan
        if "__inv_source_id" not in rec:
            rec["__inv_source_id"] = np.nan
        if "__family_id" not in rec:
            rec["__family_id"] = np.nan
        if "__source__" not in rec:
            rec["__source__"] = None

    df_out = pd.DataFrame.from_records(records)

    for col in base_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan
    for col in meta_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan

    if "__inv_flag" in df_out.columns:
        df_out["__inv_flag"] = df_out["__inv_flag"].apply(lambda v: bool(v) if not pd.isna(v) else False)
    if "__inv_rotation" in df_out.columns:
        df_out["__inv_rotation"] = pd.to_numeric(df_out["__inv_rotation"], errors="coerce")
    if "__inv_source_id" in df_out.columns:
        df_out["__inv_source_id"] = pd.to_numeric(df_out["__inv_source_id"], errors="coerce")
    if "__family_id" in df_out.columns:
        try:
            df_out["__family_id"] = pd.to_numeric(df_out["__family_id"])
        except Exception:
            pass
    if "__family_size" in df_out.columns:
        df_out["__family_size"] = pd.to_numeric(df_out["__family_size"], errors="coerce")

    ordered_cols = base_cols + [c for c in meta_cols if c not in base_cols]
    extra_cols = [c for c in df_out.columns if c not in ordered_cols]
    return df_out[ordered_cols + extra_cols]


def make_transpositions_df(df_source: pd.DataFrame, semitones: int, tag: Optional[str] = None, include_original: bool = True) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    cols = [
        "id", "n", "interval", "notes", "bass", "octave", "frequencies", "chroma",
        "tag", "code", "span_semitones", "abs_mask_int", "abs_mask_hex", "notes_abs_json",
    ]
    for _, row in df_source.iterrows():
        rowd = row.to_dict()
        if include_original:
            base = {key: rowd.get(key) for key in cols}
            base["interval"] = _to_int_list(base["interval"]) if isinstance(base.get("interval"), str) else base.get("interval")
            base["notes"] = [str(x) for x in _parse_pg_array(base.get("notes"))] if isinstance(base.get("notes"), str) else base.get("notes")
            base["id"] = rowd.get("id")
            records.append(base)
        records.append(transpose_row(rowd, semitones, tag))
    return pd.DataFrame.from_records(records, columns=cols)


# ------------------------------------------------------------
# Demo: cargar una muestra, generar inversiones y graficar MDS
# ------------------------------------------------------------

from config import config_db  # noqa: E402



def fetch_sample(min_notes: int = 3, max_notes: int = 3, limit: int = 24) -> pd.DataFrame:
    """Retirado: usa QueryExecutor(**config_db) + consultas en config.py directamente."""
    raise NotImplementedError(
        "fetch_sample fue retirado. Usa QueryExecutor + constantes SQL de config.py"
    )


def compute_sethares_vectors(df: pd.DataFrame) -> Tuple[np.ndarray, List[Any]]:
    """Retirado: usa LaboratorioAcordes + visualization en lugar de helpers locales."""
    raise NotImplementedError(
        "compute_sethares_vectors fue retirado. Usa LaboratorioAcordes + visualization."
    )


def mds_embed(vectors: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Retirado: usa LaboratorioAcordes + visualization en lugar de helpers locales."""
    raise NotImplementedError("mds_embed fue retirado. Usa LaboratorioAcordes + visualization.")


def demo_mds(limit: int = 24, min_notes: int = 3, max_notes: int = 3, include_original: bool = True, output_html: Optional[Path] = None):
    """Retirado: usa el pipeline principal (lab.py/visualization.py)."""
    raise NotImplementedError("demo_mds fue retirado. Usa LaboratorioAcordes + visualization.")


if __name__ == '__main__':
    print('Este modulo ya no ejecuta demos directamente. Usa tools/run_lab.py o el notebook.')
