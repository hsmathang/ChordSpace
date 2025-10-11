"""
Importa acordes a la tabla 'chords' desde un CSV, con upsert por (n, notes, interval).

Requisitos del CSV (columnas esperadas):
  id (opcional, ignorado), n, interval, notes, bass, octave, frequencies, chroma, tag, code
Los campos de listas pueden venir como literales Python (p.ej., "[4,3]", "['0','4','7']").

Uso
  python -m tools.import_chords_csv --csv data/chords_to_import.csv

Notas
- Crea un índice único si no existe: (n, notes, interval)
- Inserta con ON CONFLICT DO NOTHING para evitar duplicados.
"""

from __future__ import annotations

import argparse
import csv
from ast import literal_eval
from pathlib import Path
from typing import Any, List, Optional

import psycopg2
from psycopg2.extras import execute_values

from config import config_db


def parse_field(val: Any, default=None):
    if val is None:
        return default
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() == "none":
            return default
        try:
            return literal_eval(s)
        except Exception:
            return val
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    args = ap.parse_args()

    rows: List[List[Any]] = []
    with args.csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = int(r.get("n")) if r.get("n") not in (None, "") else None
            interval = parse_field(r.get("interval"), [])
            notes = parse_field(r.get("notes"), [])
            bass = r.get("bass")
            octave = r.get("octave")
            frequencies = parse_field(r.get("frequencies"), None)
            chroma = parse_field(r.get("chroma"), None)
            tag = r.get("tag")
            code = r.get("code")

            if n is None or not interval or not notes:
                continue
            rows.append([n, interval, notes, bass, octave, frequencies, chroma, tag, code])

    if not rows:
        print("No hay filas válidas para importar.")
        return

    conn = psycopg2.connect(
        host=config_db["host"],
        port=config_db["port"],
        user=config_db["user"],
        password=config_db["password"],
        dbname=config_db["dbname"],
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        # Índice único por (n, notes, interval)
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relkind = 'i'
                      AND c.relname = 'idx_chords_unique_n_notes_interval'
                ) THEN
                    CREATE UNIQUE INDEX idx_chords_unique_n_notes_interval
                    ON chords (n, notes, interval);
                END IF;
            END$$;
            """
        )

        insert_sql = (
            "INSERT INTO chords (n, interval, notes, bass, octave, frequencies, chroma, tag, code) VALUES %s "
            "ON CONFLICT (n, notes, interval) DO NOTHING"
        )
        execute_values(cur, insert_sql, rows, page_size=1000)

    conn.close()
    print(f"Importadas {len(rows)} filas (o ignoradas si ya existían).")


if __name__ == "__main__":
    main()

