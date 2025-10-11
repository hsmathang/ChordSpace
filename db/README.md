# Database Setup

This folder contains a ready-to-run PostgreSQL setup for local experiments and notebooks.

## Quick Start (Docker Compose)

1) Copy env and start DB
- `cp db/.env.db.example db/.env.db`
- `docker compose -f docker-compose.db.yml up -d`

2) Verify connection
- Host: `127.0.0.1`, Port: `5432`
- DB: `ChordCodex` (or the value in `db/.env.db`)
- User/Password: from `db/.env.db`

3) Load sample data
- `python -m tools.import_chords_csv --csv data/chords_sample.csv`

Notes
- The schema is created automatically at first container start from `db/init/01-schema.sql`.
- Do not commit `db/.env.db` (ignored by .gitignore).

## Without Docker

- Create a local PostgreSQL database named `ChordCodex` (or choose your own and reflect it in `.env`).
- Run `db/init/01-schema.sql` with `psql` or your GUI client.
- Import `data/chords_sample.csv` via `python -m tools.import_chords_csv --csv data/chords_sample.csv`.

## App Configuration

- Copy `example.env` to `.env` and set:
  - `DB_HOST=127.0.0.1`
  - `DB_PORT=5432`
  - `DB_USER=postgres` (or your user)
  - `DB_PASSWORD=...`
  - `DB_NAME=ChordCodex`

Then notebooks and tools that use the DB will work.
