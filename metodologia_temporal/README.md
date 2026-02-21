# ChordSpace

Herramientas, notebooks y GUI para explorar rugosidad y consonancia en acordes usando la base masiva de ChordCodex.

---

## TL;DR
- Python 3.11.x (las dependencias fallan con 3.9 o 3.10).
- PostgreSQL 16 con la tabla `public.chords` completa (interval como `integer[]`).
- Popular la base con `python -m tools.populate_db --mode full` y validar ~2 579 129 filas.
- Lanzar la GUI con `python -m tools.gui_experiment_launcher`.

---

## Requisitos clave
- Python 3.11.x (recomendado 3.11.3). En macOS evita usar Homebrew para Python; descarga el instalador oficial.
- PostgreSQL 16 (Postgres.app en macOS, instalador oficial en Windows/Linux) o Docker.
- Git y virtualenv.
- Opcional: `polars-lts-cpu` si ejecutas Python x86_64 sobre Apple Silicon (Rosetta).

> Nota macOS ARM: si `python -c "import platform; print(platform.machine())"` devuelve `x86_64`, estas bajo Rosetta. Instala `pip install --no-binary polars polars-lts-cpu==1.32.3` o crea un entorno Python ARM nativo.

---

## 1. Preparar el entorno Python
```bash
git clone https://github.com/hsmathang/ChordSpace.git
cd ChordSpace
python3.11 -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp example.env .env
```
Edita `.env` con los datos de tu servidor PostgreSQL. Valores por defecto esperados:
```
DB_HOST=127.0.0.1
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=postgres
DB_NAME=ChordCodex
```

---

## 2. Preparar PostgreSQL y la base ChordCodex
### Opcion A (recomendada): Postgres nativo
1. Instala Postgres 16 (Postgres.app en macOS o instalador oficial en Windows/Linux).
2. Crea la base y aplica el esquema completo:
   ```bash
   createdb ChordCodex          # macOS/Linux
   # Windows: createdb.exe -h 127.0.0.1 -U postgres ChordCodex
   psql -d ChordCodex -f db/init/01-schema.sql
   ```
3. Poblacion guiada:
   ```bash
   python -m tools.populate_db --mode quick   # ~100k filas para validar
   python -m tools.populate_db --mode full    # ~2.6M filas, varios minutos
   ```
4. Verifica el esquema e indices:
   ```bash
   psql -d ChordCodex -f tools/sql/check_schema.sql
   psql -d ChordCodex -c "SELECT COUNT(*) FROM chords;"
   ```
   `COUNT` esperado: ~2,579,129 filas.

### Opcion B: Docker Compose
1. Copia las variables de entorno del servicio:
   ```bash
   cp db/.env.db.example db/.env.db
   docker compose -f docker-compose.db.yml up -d
   ```
2. Espera a que Postgres 16 este listo y aplica el esquema (solo la primera vez):
   ```bash
   docker compose -f docker-compose.db.yml exec db psql -d "$env:POSTGRES_DB" -f /docker-entrypoint-initdb.d/01-schema.sql
   ```
3. Desde tu host, carga los datos con el mismo comando del modo nativo:
   ```bash
   python -m tools.populate_db --mode full
   ```
4. Verifica desde el host (asegura que `psql` apunte al puerto mapeado):
   ```bash
   psql -h 127.0.0.1 -p 5432 -d ChordCodex -f tools/sql/check_schema.sql
   psql -h 127.0.0.1 -p 5432 -d ChordCodex -c "SELECT COUNT(*) FROM chords;"
   ```

> El archivo `data/chords_sample.csv` solo sirve para pruebas rapidas. No reemplaza al dataset completo.

---

## 3. Ejecutar herramientas y experimentos
- Notebooks: `python -m notebook` y abre cualquiera de los `.ipynb`.
- Consultas rapidas: `python -m tools.run_sql --query QUERY_CHORDS_WITH_NAME --limit 10`.
- GUI: `python -m tools.gui_experiment_launcher` (lee [docs/GUI.md](docs/GUI.md)).
- Pipelines CLI: revisa `tools/populate_db.py` y `tools/run_sql.py` para ejemplos reproducibles.

Los componentes leyeran credenciales desde `.env`. Si necesitas usar variables del sistema, exportalas antes de lanzar el proceso.

---

## Troubleshooting recurrente
- **Python 3.9 o 3.10**: veras errores de build con `contourpy` o `matplotlib`. Instala Python 3.11.x.
- **Illegal instruction (AVX2) con polars**: ocurre en macOS ARM bajo Rosetta. Ejecuta:
  ```bash
  pip uninstall -y polars polars-lts-cpu
  pip install --no-binary polars polars-lts-cpu==1.32.3
  ```
  o crea un entorno ARM nativo (`arch -arm64 python3.11 -m venv .venv`).
- **Homebrew en macOS 12 falla con openssl**: usa Postgres.app en lugar de `brew install postgresql`.
- **Docker lenta**: la carga completa supera los 2.5M registros. Asegura al menos 6 GB de RAM disponibles y disco libre >10 GB.
- **Indices faltantes**: ejecuta `psql -d ChordCodex -f tools/sql/check_schema.sql` y recrea con `db/init/01-schema.sql` si es necesario.

---

## Referencias adicionales
- [docs/DB_SETUP.md](docs/DB_SETUP.md): guia extendida con capturas y tiempos estimados.
- [docs/GUI.md](docs/GUI.md): flujo del lanzador de experimentos y estructura de resultados.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): configuraciones para replicar los analisis.

Para soporte adicional, abre un issue describiendo tu sistema operativo, version de Python, logs relevantes y resultado de `psql -d ChordCodex -f tools/sql/check_schema.sql`.
