"""
Pequeño runner para probar consultas SQL definidas en config.py desde terminal.

Uso
  python -m tools.run_sql --query QUERY_TRIADS_WITH_INVERSIONS_21 --limit 30

Parámetros
- --query: nombre de la constante en config.py que contiene la consulta (string SQL).
- --limit: muestra primeros N registros (no altera la consulta, solo la impresión).
"""

from __future__ import annotations

import argparse
from pprint import pprint

from config import config_db
from chordcodex.model import QueryExecutor
import config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Nombre de la constante SQL en config.py")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    sql = getattr(cfg, args.query)
    qe = QueryExecutor(**config_db)
    rows = qe.as_raw(sql)
    print(f"Filas devueltas: {len(rows)}")
    for r in rows[: args.limit]:
        pprint(r)


if __name__ == "__main__":
    main()

