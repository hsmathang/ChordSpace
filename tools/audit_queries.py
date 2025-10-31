"""
Herramienta de auditoría de consultas SQL usadas por la GUI y utilidades CLI.

Características:
  * Descubre las consultas registradas en config.py y en custom_queries.json.
  * Evalúa patrones SQL problemáticos frecuentes (ORDER BY RANDOM, casts sobre notes[1], etc.).
  * Ejecuta EXPLAIN (y opcionalmente EXPLAIN ANALYZE) con límites de seguridad.
  * Recupera una muestra de filas para validar tipos y detectar columnas ausentes.
  * Emite un reporte CSV/JSON y un resumen por consola para priorizar refactors.

Ejemplo:
    python -m tools.audit_queries --with-analyze --limit 150 --output-dir outputs/audit_queries/latest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - entorno sin pandas
    pd = None

from config import config_db
from tools.query_registry import get_all_queries


SQL_WARNING_RULES: Tuple[Tuple[str, re.Pattern, str], ...] = (
    (
        "order_by_random",
        re.compile(r"ORDER\s+BY\s+RANDOM", re.IGNORECASE),
        "Uso de ORDER BY RANDOM() → forza sort global en PostgreSQL; considerar TABLESAMPLE o filtros deterministas.",
    ),
    (
        "root_note_bitcast",
        re.compile(r"\('x'\s*\|\|\s*notes\[1\]\)\s*::\s*bit\(32\)::\s*int", re.IGNORECASE),
        "Cast ('x'||notes[1])::bit(32)::int evita el uso de índices; conviene columna root_note persistida.",
    ),
    (
        "select_star",
        re.compile(r"SELECT\s+\*\s+FROM\s+chords", re.IGNORECASE),
        "SELECT * → revisa si podemos proyectar columnas necesarias para reducir ancho de fila.",
    ),
)


DYNAMIC_QUERY_TEMPLATES: Tuple[Dict[str, str], ...] = (
    {
        "name": "INLINE_SELECTED_IDS",
        "sql": "SELECT * FROM chords WHERE id IN (<id_list>) ORDER BY id;",
        "usage": "Construida por GUI (compare y reductions) cuando se seleccionan acordes manualmente.",
    },
    {
        "name": "INLINE_ABS_MASK_LOOKUP",
        "sql": "SELECT * FROM chords WHERE abs_mask_int = ANY(%s);",
        "usage": "Usada por experiment_inversions para rehidratar inversiones existentes.",
    },
    {
        "name": "INLINE_IDS_ANY_LOOKUP",
        "sql": "SELECT * FROM chords WHERE id = ANY(%s);",
        "usage": "Rehidrata acordes base con información completa antes de unir inversiones.",
    },
    {
        "name": "INLINE_COUNT_CHORDS",
        "sql": "SELECT COUNT(*) AS total FROM chords;",
        "usage": "Validación post-carga en populate_db.",
    },
)


@dataclass
class AuditResult:
    name: str
    source: str
    category: str
    status: str
    error: Optional[str]
    warnings: List[str]
    has_limit: bool
    sample_rows: Optional[int]
    sample_elapsed_s: Optional[float]
    plan_node: Optional[str]
    plan_total_cost: Optional[float]
    analyze_elapsed_s: Optional[float]
    output_path: Optional[str]


def collect_registered_queries() -> List[Tuple[str, str, str]]:
    """Return list of (name, sql, source)."""
    registry = get_all_queries()
    collected: List[Tuple[str, str, str]] = []
    for name, payload in registry.items():
        collected.append((name, payload["sql"], payload["source"]))
    return collected


def collect_dynamic_templates() -> List[Tuple[str, str, str]]:
    entries: List[Tuple[str, str, str]] = []
    for item in DYNAMIC_QUERY_TEMPLATES:
        entries.append((item["name"], item["sql"], f"template:{item['usage']}"))
    return entries


def has_limit_clause(sql: str) -> bool:
    return bool(re.search(r"\bLIMIT\b|\bFETCH\s+FIRST\b", sql, re.IGNORECASE))


def wrap_with_limit(sql: str, limit: Optional[int]) -> Tuple[str, bool]:
    body = sql.strip().rstrip(";")
    if not limit or limit <= 0 or has_limit_clause(body):
        return body, False
    return f"SELECT * FROM ({body}) AS audit_sub LIMIT {limit}", True


def detect_warnings(sql: str) -> List[str]:
    warnings: List[str] = []
    for _, pattern, message in SQL_WARNING_RULES:
        if pattern.search(sql):
            warnings.append(message)
    return warnings


def _extract_cell(row: object) -> Optional[str]:
    if row is None:
        return None
    if isinstance(row, dict):
        if not row:
            return None
        return next(iter(row.values()))
    if isinstance(row, (list, tuple)):
        return row[0] if row else None
    return str(row)


def extract_plan_summary(plan_rows: object) -> Tuple[Optional[str], Optional[float]]:
    if plan_rows is None:
        return None, None
    if pd is not None and isinstance(plan_rows, pd.DataFrame):
        if plan_rows.empty:
            return None, None
        cell = plan_rows.iloc[0, 0]
    else:
        cell = None
        if isinstance(plan_rows, list) and plan_rows:
            cell = _extract_cell(plan_rows[0])
    if cell is None:
        return None, None
    try:
        plan_json = json.loads(cell)[0]["Plan"]
        node = plan_json.get("Node Type")
        total_cost = plan_json.get("Total Cost")
        if node and plan_json.get("Relation Name"):
            node = f"{node} on {plan_json['Relation Name']}"
        return node, float(total_cost) if total_cost is not None else None
    except Exception:
        return None, None


def run_query(executor: Any, sql: str) -> Tuple[object, float]:
    start = time.perf_counter()
    if pd is not None:
        result = executor.as_pandas(sql)
    else:
        result = executor.as_raw(sql)
    elapsed = time.perf_counter() - start
    return result, elapsed


def audit_query(
    executor: Any,
    name: str,
    sql: str,
    source: str,
    *,
    limit: Optional[int],
    with_analyze: bool,
    dry_run: bool,
    output_dir: Optional[Path],
) -> AuditResult:
    trimmed = sql.strip()
    warnings = detect_warnings(trimmed)
    limited_sql, wrapped = wrap_with_limit(trimmed, limit)

    if dry_run:
        return AuditResult(
            name=name,
            source=source,
            category="dry-run",
            status="skipped",
            error=None,
            warnings=warnings,
            has_limit=wrapped or has_limit_clause(trimmed),
            sample_rows=None,
            sample_elapsed_s=None,
            plan_node=None,
            plan_total_cost=None,
            analyze_elapsed_s=None,
            output_path=None,
        )

    plan_node = None
    plan_total_cost = None
    analyze_elapsed = None
    sample_rows = None
    sample_elapsed = None
    out_path: Optional[Path] = None

    try:
        plan_rows, _ = run_query(executor, f"EXPLAIN (FORMAT JSON) {trimmed}")
        plan_node, plan_total_cost = extract_plan_summary(plan_rows)
    except Exception as exc:  # pragma: no cover - EXPLAIN failure
        warnings.append(f"EXPLAIN falló: {exc}")

    try:
        sample_result, sample_elapsed = run_query(executor, limited_sql)
        if pd is not None and isinstance(sample_result, pd.DataFrame):
            sample_rows = int(sample_result.shape[0])
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{name.lower()}_sample.csv"
            sample_result.to_csv(out_path, index=False)
        else:
            sample_rows = len(sample_result) if isinstance(sample_result, list) else None
            if output_dir and isinstance(sample_result, list) and sample_result and isinstance(sample_result[0], dict):
                output_dir.mkdir(parents=True, exist_ok=True)
                out_path = output_dir / f"{name.lower()}_sample.csv"
                import csv  # local import to evitar carga si no se usa

                with out_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(sample_result[0].keys()))
                    writer.writeheader()
                    writer.writerows(sample_result)
    except Exception as exc:
        return AuditResult(
            name=name,
            source=source,
            category="query",
            status="error",
            error=str(exc),
            warnings=warnings,
            has_limit=wrapped or has_limit_clause(trimmed),
            sample_rows=None,
            sample_elapsed_s=None,
            plan_node=plan_node,
            plan_total_cost=plan_total_cost,
            analyze_elapsed_s=analyze_elapsed,
            output_path=str(out_path) if out_path else None,
        )

    if with_analyze:
        try:
            _, analyze_elapsed = run_query(executor, f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {limited_sql}")
        except Exception as exc:  # pragma: no cover
            warnings.append(f"EXPLAIN ANALYZE falló: {exc}")

    return AuditResult(
        name=name,
        source=source,
        category="query",
        status="ok",
        error=None,
        warnings=warnings,
        has_limit=wrapped or has_limit_clause(trimmed),
        sample_rows=sample_rows,
        sample_elapsed_s=sample_elapsed,
        plan_node=plan_node,
        plan_total_cost=plan_total_cost,
        analyze_elapsed_s=analyze_elapsed,
        output_path=str(out_path) if out_path else None,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoría de consultas SQL de ChordSpace.")
    parser.add_argument(
        "--scope",
        choices=["all", "config", "dynamic"],
        default="all",
        help="Conjunto de consultas a auditar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No ejecuta SQL; solo lista consultas y advertencias estáticas.",
    )
    parser.add_argument(
        "--with-analyze",
        action="store_true",
        help="Ejecuta también EXPLAIN ANALYZE (limitado).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="Límite para muestreos y EXPLAIN ANALYZE cuando la consulta no tenga LIMIT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio para guardar muestras CSV y el reporte consolidado.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    entries: List[Tuple[str, str, str]] = []
    if args.scope in {"all", "config"}:
        entries.extend(collect_registered_queries())
    if args.scope in {"all", "dynamic"}:
        entries.extend(collect_dynamic_templates())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None and not args.dry_run:
        output_dir = Path("outputs") / "audit_queries" / timestamp

    executor: Any = None
    if not args.dry_run:
        import_error: Optional[Exception] = None
        executor_cls: Optional[Any] = None
        try:  # pragma: no cover - dependencia externa
            from chordcodex.model import QueryExecutor as executor_cls  # type: ignore
        except ModuleNotFoundError as exc:
            import_error = exc
            try:
                from synth_tools import QueryExecutor as executor_cls  # type: ignore
            except Exception as exc_inner:  # numpy/pandas ausentes, etc.
                import_error = exc_inner
                executor_cls = None
        except Exception as exc:
            import_error = exc
        if executor_cls is None:
            parser.error(
                "No se pudo importar QueryExecutor (chordcodex/synth_tools). "
                "Instala las dependencias requeridas antes de ejecutar la auditoría."
                f" Detalle: {import_error}"
            )
        executor = executor_cls(**config_db)

    results: List[AuditResult] = []
    for name, sql, source in entries:
        result = audit_query(
            executor,
            name,
            sql,
            source,
            limit=args.limit,
            with_analyze=args.with_analyze,
            dry_run=args.dry_run,
            output_dir=output_dir,
        )
        results.append(result)

    rows = [asdict(r) for r in results]

    print("=== Auditoría de consultas ===")
    header = ["name", "source", "status", "sample_rows", "sample_elapsed_s", "plan_node"]
    print("\t".join(header))
    for item in rows:
        print(
            "\t".join(
                str(item.get(col, "")) if item.get(col) is not None else ""
                for col in header
            )
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if pd is not None:
            pd.DataFrame(rows).to_csv(output_dir / "audit_results.csv", index=False)
            pd.DataFrame(rows).to_json(
                output_dir / "audit_results.json", orient="records", indent=2, force_ascii=False
            )
        else:
            import csv

            with (output_dir / "audit_results.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys() if rows else [])
                if rows:
                    writer.writeheader()
                    writer.writerows(rows)
            (output_dir / "audit_results.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(f"\nReportes almacenados en: {output_dir}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
