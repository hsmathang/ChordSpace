# Database quick reference

This directory stores the assets required to bootstrap PostgreSQL for ChordSpace. For the full walkthrough see `docs/DB_SETUP.md`; the notes below highlight the essentials.

---

## 1. Environment files
- `db/.env.db.example`: copy to `db/.env.db` to configure the Docker service.
- `.env`: copy from `example.env` and keep database credentials in sync with the running instance.

---

## 2. Schema
- `init/01-schema.sql` defines the canonical table expected by `chordcodex.scripts.db_fill_v2`. It upgrades older installations by:
  - Casting `interval` to `INTEGER[]`.
  - Casting `notes` to `TEXT[]`.
  - Adding `span_semitones`, `abs_mask_int`, `abs_mask_hex`, `notes_abs_json`.
  - Creating the unique indices and a GIN index on `interval`.

Apply it with:
```bash
psql -d ChordCodex -f db/init/01-schema.sql
```

---

## 3. Running PostgreSQL
### Docker
```bash
cp db/.env.db.example db/.env.db
docker compose -f docker-compose.db.yml up -d
```
The schema file is mounted into `/docker-entrypoint-initdb.d/`. Re-run the command above if you update `01-schema.sql`.

### Native installation
1. Install PostgreSQL 16 (Postgres.app on macOS, installer on Windows/Linux).
2. Create the database `ChordCodex` and apply `init/01-schema.sql`.

---

## 4. Populating data
Use the wrapper provided in `tools/populate_db.py`:
```bash
python -m tools.populate_db --mode quick   # smoke test (~100k rows)
python -m tools.populate_db --mode full    # full dataset (~2.6M rows)
```
Check the schema afterwards:
```bash
psql -d ChordCodex -f tools/sql/check_schema.sql
psql -d ChordCodex -c "SELECT COUNT(*) FROM chords;"
```
The CSV located at `data/chords_sample.csv` is only for smoke tests and does not replace the full `ChordCodex` population.

---

## 5. Troubleshooting
- **Type mismatch (`smallint[] vs integer[]`)**: rerun `db/init/01-schema.sql`.
- **Docker resets**: `docker compose -f docker-compose.db.yml down -v` wipes data; run the population script again afterwards.
- **Credentials**: keep `.env` (application) and `db/.env.db` (Docker) aligned.

For additional context and platform-specific notes, refer to `docs/DB_SETUP.md`.
