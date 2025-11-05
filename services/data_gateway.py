"""Experiment data access layer for CLI/GUI tools.

This module centralises how experiment runners access chord populations.
It provides a registry-driven gateway interface that abstracts the
underlying data source (database, CSV exports, etc.), ensures consistent
SQL resolution, handles deduplication policies and exposes the shared
chord template catalogue.  The gateway registry now also exposes a
``generator`` backend capable of streaming populations produced by the
procedural engine in :mod:`gen.generate`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

from config import (
    CHORD_TEMPLATES_METADATA,
    QUERY_DYADS_REFERENCE,
    QUERY_TRIADS_CORE,
    config_db,
)
from tools.population_utils import dedupe_population
from tools.query_registry import get_all_queries, resolve_query_sql

try:  # pragma: no cover - prefer packaged executor when available
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from synth_tools import QueryExecutor  # type: ignore

from gen.adapters import iter_chord_records
from gen.generate import gen_struct, gen_total


# Keep references to ensure the canonical config constants are loaded.
_ = QUERY_DYADS_REFERENCE, QUERY_TRIADS_CORE

DEFAULT_DYADS_QUERY = "QUERY_DYADS_REFERENCE"
DEFAULT_TRIADS_QUERY = "QUERY_TRIADS_CORE"
DEFAULT_GATEWAY_NAME = "database"


@dataclass
class PopulationResult:
    """Container with population data returned by a gateway."""

    dataframe: pd.DataFrame
    dedupe_key: Optional[str] = None
    sources: Tuple[str, ...] = ()
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataframe = self.dataframe.reset_index(drop=True)


_NAMED_ALPHABETS: Dict[str, Tuple[int, ...]] = {
    "chromatic": tuple(range(12)),
    "cromatica": tuple(range(12)),
    "chromática": tuple(range(12)),
    "diatonic": (0, 2, 4, 5, 7, 9, 11),
    "diatonica": (0, 2, 4, 5, 7, 9, 11),
    "pentatonic": (0, 2, 4, 7, 9),
    "pentatonica": (0, 2, 4, 7, 9),
}


def _parse_generator_spec_string(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"No se pudo interpretar el spec JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("El spec JSON debe ser un objeto {…}.")
        return data
    result: Dict[str, Any] = {}
    for segment in text.split(";"):
        if not segment.strip():
            continue
        if "=" not in segment:
            raise ValueError(
                "Las especificaciones textuales deben usar 'clave=valor' separadas por ';'.",
            )
        key, value = segment.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _parse_int_list(raw: Any, *, allow_range: bool = False) -> Tuple[int, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, str):
        text = raw.strip().strip("[]()")
        if not text:
            return tuple()
        tokens = [tok.strip() for tok in re.split(r"[;,]", text) if tok.strip()]
        values: List[int] = []
        for token in tokens:
            if allow_range and "-" in token:
                try:
                    start_str, end_str = token.split("-", 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                except ValueError as exc:
                    raise ValueError(f"Rango inválido '{token}'. Usa formato inicio-fin.") from exc
                if start <= end:
                    values.extend(range(start, end + 1))
                else:
                    values.extend(range(end, start + 1))
            else:
                values.append(int(token))
        return tuple(values)
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, str)):
        return tuple(int(x) for x in raw)
    return (int(raw),)


def _parse_octave_range(raw: Any) -> Tuple[Optional[int], Optional[int]]:
    if raw is None:
        return (None, None)
    if isinstance(raw, str):
        text = raw.strip().strip("[]()")
        if not text:
            return (None, None)
        if "-" in text and "," not in text:
            parts = text.split("-", 1)
        else:
            parts = [tok.strip() for tok in re.split(r"[;,]", text) if tok.strip()]
        if not parts:
            return (None, None)
        if len(parts) == 1:
            value = int(parts[0])
            return (value, value)
        return (int(parts[0]), int(parts[1]))
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, str)):
        values = [int(v) for v in raw]
        if not values:
            return (None, None)
        if len(values) == 1:
            return (values[0], values[0])
        return (values[0], values[1])
    value = int(raw)
    return (value, value)


def _parse_alphabet(raw: Any) -> Tuple[int, ...]:
    if raw is None:
        raise ValueError("El alfabeto (pitch classes) es obligatorio para el generador.")
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in _NAMED_ALPHABETS:
            return _NAMED_ALPHABETS[key]
    values = _parse_int_list(raw)
    pcs = sorted({int(v) % 12 for v in values})
    if not pcs:
        raise ValueError("El alfabeto del generador no puede quedar vacío.")
    return tuple(pcs)


def _parse_interval_pattern(raw: Any) -> Tuple[int, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, str):
        text = raw.strip().strip("[]()")
        if not text:
            return tuple()
        # Permitir múltiples patrones separados por ';' y escoger el primero
        if ";" in text:
            text = text.split(";", 1)[0]
        tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
        return tuple(int(tok) for tok in tokens)
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, str)):
        values = [int(v) for v in raw]
        return tuple(values)
    return (int(raw),)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "si", "sí"}
    return bool(value)


@dataclass
class GeneratorRequest:
    """Parameters used by the streaming generator backend."""

    mode: str = "total"
    alphabet: Tuple[int, ...] = tuple(range(12))
    octave_min: Optional[int] = None
    octave_max: Optional[int] = None
    cardinalities: Tuple[int, ...] = (3,)
    edge_pc0: bool = False
    max_span: Optional[int] = None
    must_have_pcs: Tuple[int, ...] = tuple()
    must_avoid_pcs: Tuple[int, ...] = tuple()
    interval_pattern: Tuple[int, ...] = tuple()
    max_struct_span: int = 12
    label: Optional[str] = None
    tag: Optional[str] = None
    limit: Optional[int] = None
    batch_size: Optional[int] = None

    @classmethod
    def from_any(
        cls,
        raw: Union["GeneratorRequest", Mapping[str, Any], str],
        *,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> "GeneratorRequest":
        if isinstance(raw, GeneratorRequest):
            return raw
        if isinstance(raw, str):
            mapping = _parse_generator_spec_string(raw)
            return cls.from_mapping(mapping, defaults=defaults)
        if isinstance(raw, Mapping):
            return cls.from_mapping(raw, defaults=defaults)
        raise TypeError(f"Spec del generador no soportado: {type(raw)!r}")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        defaults: Optional[Mapping[str, Any]] = None,
    ) -> "GeneratorRequest":
        data: Dict[str, Any] = dict(defaults or {})
        data.update(dict(mapping))

        mode = str(data.get("mode", data.get("tipo", "total"))).strip().lower() or "total"
        alphabet = _parse_alphabet(
            data.get("alphabet")
            or data.get("pitch_classes")
            or data.get("pcs")
            or data.get("scale")
        )
        oct_min, oct_max = _parse_octave_range(data.get("octaves") or data.get("octava"))
        if mode == "total" and (oct_min is None or oct_max is None):
            raise ValueError("El modo 'total' requiere un rango de octavas (por ejemplo '4-5').")
        cardinalities = _parse_int_list(
            data.get("cardinalities")
            or data.get("sizes")
            or data.get("n")
            or data.get("N"),
        )
        if not cardinalities:
            raise ValueError("Debes indicar al menos una cardinalidad (N).")
        cardinalities = tuple(sorted({int(abs(n)) for n in cardinalities if int(abs(n)) >= 2}))
        if not cardinalities:
            raise ValueError("Las cardinalidades deben ser enteros ≥ 2.")

        max_span = data.get("max_span")
        max_span_val = int(max_span) if max_span not in (None, "") else None
        must_have = _parse_int_list(data.get("must_have") or data.get("must_have_pcs"))
        must_have = tuple(sorted({int(pc) % 12 for pc in must_have}))
        must_avoid = _parse_int_list(data.get("must_avoid") or data.get("must_avoid_pcs"))
        must_avoid = tuple(sorted({int(pc) % 12 for pc in must_avoid}))
        interval_pattern = _parse_interval_pattern(
            data.get("interval_pattern")
            or data.get("interval")
            or data.get("pattern"),
        )
        max_struct_span = data.get("max_struct_span")
        max_struct_span_val = int(max_struct_span) if max_struct_span not in (None, "") else 12
        limit = data.get("limit")
        limit_val = int(limit) if limit not in (None, "") else None
        batch = data.get("batch_size")
        batch_val = int(batch) if batch not in (None, "") else None

        return cls(
            mode=mode,
            alphabet=alphabet,
            octave_min=oct_min,
            octave_max=oct_max,
            cardinalities=cardinalities,
            edge_pc0=_coerce_bool(data.get("edge_pc0", False)),
            max_span=max_span_val,
            must_have_pcs=must_have,
            must_avoid_pcs=must_avoid,
            interval_pattern=interval_pattern,
            max_struct_span=max_struct_span_val,
            label=(str(data.get("label")) if data.get("label") not in (None, "") else None),
            tag=(str(data.get("tag")) if data.get("tag") not in (None, "") else None),
            limit=limit_val,
            batch_size=batch_val,
        )

    def ensure_defaults(self, *, fallback_tag: str, index: int) -> None:
        if not self.label:
            self.label = self.default_label(index)
        if not self.tag:
            self.tag = fallback_tag

    def default_label(self, index: int) -> str:
        pcs = "-".join(str(pc) for pc in self.alphabet)
        cards = "-".join(str(n) for n in self.cardinalities)
        if self.mode == "struct":
            label = f"GEN_STRUCT[{pcs}]_N[{cards}]"
        else:
            label = (
                f"GEN_TOTAL[{pcs}]_O[{self.octave_min},{self.octave_max}]_N[{cards}]"
            )
        if index > 0:
            return f"{label}#{index+1}"
        return label

    def filters(self) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if self.max_span is not None:
            filters["max_span"] = int(self.max_span)
        if self.must_have_pcs:
            filters["must_have_pcs"] = {int(pc) % 12 for pc in self.must_have_pcs}
        if self.must_avoid_pcs:
            filters["must_avoid_pcs"] = {int(pc) % 12 for pc in self.must_avoid_pcs}
        if self.interval_pattern:
            filters["interval_pattern"] = tuple(int(i) for i in self.interval_pattern)
        return filters

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "alphabet": list(self.alphabet),
            "octaves": [self.octave_min, self.octave_max],
            "cardinalities": list(self.cardinalities),
            "edge_pc0": bool(self.edge_pc0),
            "max_span": self.max_span,
            "must_have_pcs": list(self.must_have_pcs),
            "must_avoid_pcs": list(self.must_avoid_pcs),
            "interval_pattern": list(self.interval_pattern),
            "max_struct_span": self.max_struct_span,
            "label": self.label,
            "tag": self.tag,
            "limit": self.limit,
            "batch_size": self.batch_size,
        }


class ExperimentDataGateway(Protocol):
    """Protocol implemented by experiment data gateways."""

    def resolve_sql(self, query_or_alias: str) -> str:
        """Resolve a query reference to SQL using the shared registry."""

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:
        """Fetch and combine populations identified by ``sources``."""

    def ingest_population(
        self,
        frame: pd.DataFrame,
        *,
        dedupe: bool = True,
        source: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PopulationResult:
        """Ingest an already materialised population frame."""

    def get_templates(self) -> Sequence[Mapping[str, Any]]:
        """Return chord template metadata for consumers that need it."""

    def available_queries(self) -> Mapping[str, Dict[str, str]]:
        """Expose the discoverable SQL registry (name -> {sql, source})."""


class BaseExperimentDataGateway(ExperimentDataGateway):
    """Base implementation with shared helpers for concrete gateways."""

    def resolve_sql(self, query_or_alias: str) -> str:
        trimmed = (query_or_alias or "").strip()
        if not trimmed:
            raise ValueError("No se proporcionó una referencia de consulta válida.")

        lower = trimmed.lower()

        if lower.startswith("sql:"):
            remainder = trimmed.split(":", 1)[1].strip()
            if not remainder:
                raise ValueError("El prefijo sql: requiere una sentencia SQL después de los dos puntos.")
            return remainder

        if lower.startswith("config:") or lower.startswith("custom:"):
            remainder = trimmed.split(":", 1)[1].strip()
            if not remainder:
                raise ValueError(
                    "El prefijo config:/custom: requiere el nombre de la consulta (por ejemplo QUERY_DYADS_REFERENCE)."
                )
            candidates = [remainder]
            upper = remainder.upper()
            if upper not in candidates:
                candidates.append(upper)
            if not upper.startswith("QUERY_"):
                prefixed = f"QUERY_{upper}"
                if prefixed not in candidates:
                    candidates.append(prefixed)

            for candidate in candidates:
                try:
                    return resolve_query_sql(candidate)
                except KeyError:
                    continue
            raise KeyError(f"No se encontró la consulta referenciada '{remainder}'.")

        try:
            return resolve_query_sql(trimmed)
        except KeyError as exc:
            raise KeyError(f"No se encontró la consulta referenciada '{query_or_alias}'.") from exc

    def available_queries(self) -> Mapping[str, Dict[str, str]]:
        return get_all_queries()

    def get_templates(self) -> Sequence[Mapping[str, Any]]:
        # Return shallow copies to avoid accidental mutation of globals.
        return tuple(dict(template) for template in CHORD_TEMPLATES_METADATA)

    def ingest_population(
        self,
        frame: pd.DataFrame,
        *,
        dedupe: bool = True,
        source: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PopulationResult:
        df = frame.copy()
        stats: Dict[str, Any] = dict(metadata or {})

        if source and "__source__" not in df.columns:
            df = df.copy()
            df["__source__"] = source

        raw_count = len(df)
        stats.setdefault("raw_count", raw_count)

        dedupe_key: Optional[str] = None
        if dedupe:
            df, dedupe_key = dedupe_population(df)
            stats["removed"] = raw_count - len(df)
        else:
            df = df.copy()
            stats["removed"] = 0

        final_count = len(df)
        stats["final_count"] = final_count

        if "__source__" in df.columns:
            source_series = df["__source__"].astype(str).fillna("")
            stats["source_counts_after"] = source_series.value_counts().to_dict()
            sources = tuple(source_series.unique())
        else:
            sources = tuple()

        return PopulationResult(df, dedupe_key, sources, stats)

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:  # pragma: no cover - abstract
        raise NotImplementedError


class GeneratorPopulationGateway(BaseExperimentDataGateway):
    """Gateway that streams populations from the procedural generator."""

    def __init__(
        self,
        *,
        default_spec: Optional[Mapping[str, Any]] = None,
        batch_size: int = 4096,
        tag: str = "GEN",
    ) -> None:
        self._defaults: Dict[str, Any] = dict(default_spec or {})
        self._batch_size = max(1, int(batch_size))
        self._tag = tag or "GEN"

    def _materialise_spec(
        self,
        spec: GeneratorRequest,
        *,
        start_id: int,
        batch_size: int,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        filters = spec.filters()
        if spec.mode == "struct":
            chord_iter = gen_struct(
                set(spec.alphabet),
                spec.cardinalities,
                max_span_struct=int(spec.max_struct_span),
            )
        elif spec.mode == "total":
            if spec.octave_min is None or spec.octave_max is None:
                raise ValueError("El modo 'total' requiere un rango de octavas definido.")
            chord_iter = gen_total(
                set(spec.alphabet),
                int(spec.octave_min),
                int(spec.octave_max),
                spec.cardinalities,
                edge_pc0=bool(spec.edge_pc0),
                early_filters=filters or None,
            )
        else:
            raise ValueError(f"Modo de generador desconocido '{spec.mode}'.")

        frames: List[pd.DataFrame] = []
        chunk: List[Dict[str, Any]] = []
        produced = 0

        for record in iter_chord_records(chord_iter, start_id=start_id, tag=spec.tag):
            record["__source__"] = spec.label
            chunk.append(record)
            produced += 1
            if len(chunk) >= batch_size:
                frames.append(pd.DataFrame.from_records(chunk))
                chunk = []
            if spec.limit is not None and produced >= int(spec.limit):
                break

        if chunk:
            frames.append(pd.DataFrame.from_records(chunk))

        if frames:
            df = pd.concat(frames, ignore_index=True, sort=False)
        else:
            df = pd.DataFrame()

        metadata = {
            "spec": spec.describe(),
            "raw_count": produced,
            "start_id": start_id,
        }
        return df, metadata

    def fetch_population(self, sources: Sequence[Any], *, dedupe: bool = True) -> PopulationResult:
        raw_specs = [spec for spec in sources if spec]
        if not raw_specs:
            if self._defaults:
                raw_specs = [self._defaults]
            else:
                raise ValueError("No se proporcionaron especificaciones para el generador.")

        specs = [GeneratorRequest.from_any(spec, defaults=self._defaults) for spec in raw_specs]

        frames: List[pd.DataFrame] = []
        counts_before: Dict[str, int] = {}
        generator_meta: List[Dict[str, Any]] = []
        next_start_id = 1

        for idx, spec in enumerate(specs):
            spec.ensure_defaults(fallback_tag=self._tag, index=idx)
            batch_size = spec.batch_size or self._batch_size
            if batch_size <= 0:
                batch_size = self._batch_size
            df, meta = self._materialise_spec(spec, start_id=next_start_id, batch_size=batch_size)
            produced = int(meta.get("raw_count", len(df)))
            if produced:
                next_start_id += produced
            label = spec.label or f"GEN#{idx}"
            counts_before[label] = produced
            generator_meta.append({**meta["spec"], "raw_count": produced})
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        metadata = {
            "source_counts_before": counts_before,
            "generator_specs": tuple(generator_meta),
        }
        return self.ingest_population(combined, dedupe=dedupe, metadata=metadata)


class DatabaseQueryGateway(BaseExperimentDataGateway):
    """Gateway that loads chord populations from the configured database."""

    def __init__(
        self,
        *,
        db_config: Optional[Mapping[str, Any]] = None,
        executor: Optional[QueryExecutor] = None,
    ) -> None:
        self._db_config: MutableMapping[str, Any] = dict(db_config or config_db)
        self._executor: Optional[QueryExecutor] = executor

    @property
    def executor(self) -> QueryExecutor:
        if self._executor is None:
            self._executor = QueryExecutor(**self._db_config)
        return self._executor

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:
        frames = []
        counts_before: Dict[str, int] = {}
        used_sources: Tuple[str, ...] = tuple()
        ordered_sources = []

        for reference in sources:
            if not reference:
                continue
            sql = self.resolve_sql(reference)
            df = self.executor.as_pandas(sql)
            counts_before[reference] = len(df)
            df = df.copy()
            if "__source__" not in df.columns:
                df["__source__"] = reference
            else:
                df["__source__"] = df["__source__"].fillna(reference)
            frames.append(df)
            ordered_sources.append(reference)

        if not frames:
            raise ValueError("No se proporcionaron consultas válidas para construir la población.")

        combined = pd.concat(frames, ignore_index=True)
        metadata = {
            "source_counts_before": counts_before,
            "query_order": tuple(ordered_sources),
        }
        result = self.ingest_population(combined, dedupe=dedupe, metadata=metadata)
        return result


class CSVPopulationGateway(BaseExperimentDataGateway):
    """Gateway that reads pre-exported populations from CSV/JSON files."""

    def __init__(
        self,
        *,
        default_path: Optional[str] = None,
        file_format: Optional[str] = None,
    ) -> None:
        self._default_path = Path(default_path).expanduser() if default_path else None
        self._explicit_format = file_format.lower() if file_format else None

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute() and self._default_path is not None:
            return (self._default_path.parent / path).resolve()
        return path

    def _detect_format(self, path: Path) -> str:
        if self._explicit_format:
            return self._explicit_format
        suffix = path.suffix.lower().lstrip(".")
        return suffix or "csv"

    def _load_file(self, path: Path) -> pd.DataFrame:
        fmt = self._detect_format(path)
        if fmt in {"json", "ndjson", "jsonl"}:
            return pd.read_json(path, orient="records", lines=True)
        if fmt in {"parquet"}:
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def fetch_population(self, sources: Sequence[str], *, dedupe: bool = True) -> PopulationResult:
        paths = [self._resolve_path(src) for src in sources if src]
        if not paths and self._default_path is not None:
            paths = [self._default_path]

        if not paths:
            raise ValueError("No se proporcionaron archivos de población para el gateway CSV.")

        frames = []
        counts_before: Dict[str, int] = {}
        ordered_paths = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"No existe el archivo de población: {path}")
            df = self._load_file(path)
            counts_before[str(path)] = len(df)
            df = df.copy()
            if "__source__" not in df.columns:
                df["__source__"] = str(path)
            frames.append(df)
            ordered_paths.append(str(path))

        combined = pd.concat(frames, ignore_index=True)
        metadata = {
            "source_counts_before": counts_before,
            "file_paths": tuple(ordered_paths),
        }
        return self.ingest_population(combined, dedupe=dedupe, metadata=metadata)


GatewayFactory = Callable[..., ExperimentDataGateway]
_GATEWAY_REGISTRY: Dict[str, GatewayFactory] = {}


def register_data_gateway(name: str, factory: GatewayFactory) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("El nombre del gateway no puede estar vacío.")
    _GATEWAY_REGISTRY[normalized] = factory


def get_registered_gateways() -> Tuple[str, ...]:
    return tuple(sorted(_GATEWAY_REGISTRY.keys()))


def create_data_gateway(name: str, **kwargs: Any) -> ExperimentDataGateway:
    normalized = name.strip().lower()
    if normalized not in _GATEWAY_REGISTRY:
        available = ", ".join(sorted(_GATEWAY_REGISTRY.keys()))
        raise KeyError(f"Gateway desconocido '{name}'. Disponibles: {available}")
    factory = _GATEWAY_REGISTRY[normalized]
    return factory(**kwargs)


register_data_gateway("database", DatabaseQueryGateway)
register_data_gateway("csv", CSVPopulationGateway)
register_data_gateway("generator", GeneratorPopulationGateway)

__all__ = [
    "DEFAULT_DYADS_QUERY",
    "DEFAULT_TRIADS_QUERY",
    "DEFAULT_GATEWAY_NAME",
    "PopulationResult",
    "ExperimentDataGateway",
    "GeneratorRequest",
    "GeneratorPopulationGateway",
    "DatabaseQueryGateway",
    "CSVPopulationGateway",
    "register_data_gateway",
    "get_registered_gateways",
    "create_data_gateway",
]
