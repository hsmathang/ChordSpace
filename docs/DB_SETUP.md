# DB Setup with ChordCodex (fork/tag)

The database is central. This project pins your fork/tag of ChordCodex so reviewers can rebuild the DB exactly.

## 1) Install dependencies

- Windows PowerShell:
  - `py -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt`
- macOS/Linux:
  - `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## 2) Start PostgreSQL

- Docker Compose (recommended):
  - `cp db/.env.db.example db/.env.db`
  - `docker compose -f docker-compose.db.yml up -d`
- Native PostgreSQL: create a database `ChordCodex` and (if needed) apply `db/init/01-schema.sql`.

## 3) App configuration

- Copy `example.env` to `.env` and set:
  - `DB_HOST=127.0.0.1`
  - `DB_PORT=5432`
  - `DB_USER` / `DB_PASSWORD`
  - `DB_NAME=ChordCodex`

## 4) Populate DB with your ChordCodex pipelines

- Quick insert benchmark (limited):
  - `python -m chordcodex.scripts.db_fill_v2 --mode benchmark-insert --limit 200000 --batch-size 20000`
- Full load (batched):
  - `python -m chordcodex.scripts.db_fill_v2 --mode full-run --batch-size 20000`
- PK migration to BIGSERIAL (if starting from older schema):
  - `python -m chordcodex.scripts.db_migrate_v2`

Validation (optional):
- `python -c "from chordcodex.model import QueryExecutor; print(QueryExecutor().from_env().as_pandas('select count(*) as n from chords'))"`

## 5) Use DB from ChordSpace

- Notebooks and tools pick credentials from `.env`.
- Run Jupyter or the GUI and execute your SQL-backed experiments.
