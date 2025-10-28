# ChordCodex: guia de preparacion

Esta guia resume el proceso completo para reconstruir la base de datos `ChordCodex` que alimenta ChordSpace, incorporando las lecciones aprendidas durante la puesta a punto en macOS 12, Windows y Linux.

---

## 1. Requisitos previos
- Python **3.11.x** (las dependencias fijadas no compilan con 3.9 ni 3.10).
- PostgreSQL 16:
  - macOS: Postgres.app (evita Homebrew en Monterey).
  - Windows: instalador oficial con las Command Line Tools.
  - Linux: paquetes oficiales o contenedores.
- Git, `virtualenv` y al menos 8 GB libres en disco (la carga completa genera ~2.6M filas).
- Para Apple Silicon usando Python x86_64 (Rosetta): fuerza `pip install --no-binary polars polars-lts-cpu==1.32.3`.

---

## 2. Crear el entorno Python
```bash
python3.11 -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp example.env .env
```
Completa `.env` con las credenciales del servidor:
```
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=ChordCodex
```

---

## 3. Instalar y preparar PostgreSQL (modo nativo)
1. Instala PostgreSQL 16 en tu sistema (Postgres.app en macOS, instalador oficial en Windows, paquetes de la distro en Linux) y confirma que `psql` está disponible.
2. Crea la base y aplica el esquema canónico:
   ```bash
   createdb ChordCodex
   psql -d ChordCodex -f db/init/01-schema.sql
   ```
   Si `createdb` no está en el PATH, puedes lanzar directamente la instrucción SQL:
   ```bash
   psql -U postgres -h 127.0.0.1 -c "CREATE DATABASE \"ChordCodex\";"
   psql -d ChordCodex -f db/init/01-schema.sql
   ```

---

## 4. Poblar la base con chordcodex
`tools/populate_db.py` envuelve `chordcodex.scripts.db_fill_v2` y estandariza los parametros recomendados.

```bash
python -m tools.populate_db --mode quick   # ~100k filas, valida conexion y esquema
python -m tools.populate_db --mode full    # ~2.6M filas, tarda varios minutos
```

Parametros utiles:
- `--mode quick|full`
- `--limit 500000` (solo para modo benchmark)
- `--batch-size 15000`
- `--extra --resume` para reanudar un proceso parcial (opcion de `db_fill_v2`)
- `--dry-run` para imprimir el comando sin ejecutarlo

El script imprime el comando final y, al concluir, reporta el conteo actual de filas.

---

## 5. Validar esquema e indices
```bash
psql -d ChordCodex -f tools/sql/check_schema.sql
psql -d ChordCodex -c "SELECT COUNT(*) FROM chords;"
```

Espera:
- Columna `interval` como `integer[]` y `notes` como `text[]`.
- Columnas adicionales: `span_semitones`, `abs_mask_int`, `abs_mask_hex`, `notes_abs_json`.
- Indices presentes: `idx_chords_unique_abs_mask_int`, `idx_chords_unique_n_notes_interval`, `idx_chords_interval`.
- Conteo total aproximado: **2 579 129** filas tras la carga completa.

---

## 6. Uso desde ChordSpace
- Las credenciales se leen desde `.env`; no necesitas variables de entorno adicionales.
- Prueba rapida:
  ```bash
  python -m tools.run_sql --query QUERY_CHORDS_WITH_NAME --limit 5
  ```
- GUI:
  ```bash
  python -m tools.gui_experiment_launcher
  ```
  Consulta `docs/GUI.md` para el flujo de trabajo y rutas de salida.

---

## 7. Problemas comunes
- `operator does not exist: smallint[] = integer[]`: vuelve a ejecutar `psql -d ChordCodex -f db/init/01-schema.sql`.
- `Illegal instruction (core dumped)`: reinstala `polars-lts-cpu` o usa un entorno Python ARM nativo (`arch -arm64` en macOS).
- Fallos de openssl/libpq en macOS 12: evita Homebrew; usa Postgres.app.
- Docker sin recursos: asigna al menos 6 GB de RAM y espacio libre en disco >10 GB antes de la carga completa.

---

## 8. Mantenimiento
- Respaldos: `pg_dump -d ChordCodex > backups/chordcodex_$(date +%Y%m%d).sql`.
- Restauración rápida: `psql -d ChordCodex < backups/archivo.sql`.

Documenta en issues cualquier variacion adicional que encuentres para mantener el flujo actualizado entre plataformas.
