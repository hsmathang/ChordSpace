"""
config.py
---------
Este módulo centraliza todas las constantes y parámetros de configuración utilizados en el laboratorio de acordes.
La configuración incluye parámetros para:
  1. Conexión a la base de datos y consultas SQL.
     - Utilizado en la carga de datos (CELDA 11 del notebook) y en pre_process.py para inicializar QueryExecutor.
  2. Evaluación y reducción dimensional.
     - Constantes como EVAL_N_NEIGHBORS, UMAP_N_COMPONENTS y KERNEL_MDS_N_COMPONENTS se usan en metrics.py y funciones de reducción dimensional (CELDA 10).
  3. Ponderaciones de rugosidad.
     - Los valores por defecto se usan en pre_process.py, en las clases PonderacionConsonancia, PonderacionImportanciaPerceptual y PonderacionCombinada.
  4. Configuración de visualización.
     - Se utilizan en visualization.py para definir escalas de color, parámetros de densidad, tamaños de marcadores, layouts, etc.
  5. Otros parámetros globales.
     - Ejemplo: SETHARES_BASE_FREQ, usado en funciones de preprocesamiento (derive_fr) y en modelos de rugosidad.
  
El objetivo es centralizar todas las configuraciones en un solo lugar para evitar duplicaciones y facilitar el mantenimiento.
"""

import os
import sys
import types
from pathlib import Path
try:
    # Preferir python-dotenv si está disponible
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    _load_dotenv = None

_BASE_DIR = Path(__file__).resolve().parent

def _fallback_load_env(env_path: Path) -> None:
    try:
        text = env_path.read_text(encoding='utf-8')
    except Exception:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('#') or '=' not in s:
            continue
        k, v = s.split('=', 1)
        k = k.strip()
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        if k and (os.getenv(k) is None):
            os.environ[k] = v

# Cargar .env con python-dotenv si existe; si no, usar fallback ligero
_env_file = _BASE_DIR / '.env'


def _shim_load_dotenv(path=None, **_kwargs) -> bool:
    target = Path(path) if path is not None else _env_file
    _fallback_load_env(target)
    return True


if _load_dotenv is None:
    module = sys.modules.get('dotenv')
    if module is None:
        module = types.ModuleType('dotenv')
        sys.modules['dotenv'] = module
    module.load_dotenv = _shim_load_dotenv  # type: ignore[attr-defined]
    _load_dotenv = module.load_dotenv  # type: ignore[assignment]
else:
    module = sys.modules.get('dotenv')
    if module is not None and not hasattr(module, 'load_dotenv'):
        module.load_dotenv = _load_dotenv  # type: ignore[attr-defined]

if _load_dotenv is not None:
    try:
        _load_dotenv(_env_file)
    except Exception:
        _fallback_load_env(_env_file)
else:
    _fallback_load_env(_env_file)

def _get_env(key, default=None):
    # Retrieve configuration from environment variables or fall back to defaults.
    value = os.getenv(key)
    if value is None or value == '':
        value = default
    if value is None:
        raise RuntimeError(f'Missing required configuration for {key}')
    value = value.strip()
    if value[:1] == '"' and value[-1:] == '"':
        value = value[1:-1]
    if value[:1] == "'" and value[-1:] == "'":
        value = value[1:-1]
    return value


# -------------------------------
# 1. Configuración de Conexión a la Base de Datos
# -------------------------------
# --- CONSTANTES ---

# --- Parámetros del Modelo Sethares ---
SETHARES_N_HARMONICS = 6
SETHARES_DECAY = 0.88
SETHARES_D_STAR = 0.24
SETHARES_S1 = 0.0207
SETHARES_S2 = 18.96
SETHARES_C1 = 5
SETHARES_C2 = -5
SETHARES_A1 = -3.51
SETHARES_A2 = -5.75
SETHARES_BASE_WEIGHTS = {'bajo': 1.5, 'consonante': 1, 'final': 1.5, 'disonante': 2}
SETHARES_CONSONANT_INTERVALS = {0, 3, 4, 7}
SETHARES_BASE_FREQ = 440

# --- Parámetros del Modelo Euler ---
EULER_CONSONANCE_WEIGHTS = {
    0: 0.0, 3: 0.25, 4: 0.35, 7: 0.15,
    1: 1.1, 2: 1.0, 5: 0.1, 6: 0.9, 8: 0.6, 9: 1.3, 10: 1.2, 11: 1.4
}

# --- Parámetros del Modelo Armónicos Críticos ---
HARMONICS_BASE_FREQ = 440

# --- Parámetros de Ponderación de Importancia Perceptual ---
IMPORTANCE_BAJO_WEIGHT = 1.8
IMPORTANCE_FINAL_WEIGHT = 1.5
IMPORTANCE_DECAY_FACTOR = 0.3


EULER_PRIME_FACTORS_CACHE = {i: {2:0, 3:0, 5:0} for i in range(13)} # Placeholder, se llena en la clase si es necesario
EULER_CONSONANCIA_WEIGHTS_DEFAULT = { # Pesos de consonancia por defecto para Euler (si no se proveen en config)
    0: 0.0,  1: 1.3,  2: 1.0,  3: 0.4,
    4: 0.5,  5: 0.2,  6: 1.4,  7: 0.1,
    8: 0.6,  9: 0.7,  10: 1.1, 11: 1.2
}

ARMONICOS_CRITICOS_BASE_FREQ_DEFAULT = 440.0 # Frecuencia base para Armónicos Críticos (Hz)

PONDERACION_CONSONANCIA_DEFAULT_WEIGHTS = { # Pesos de consonancia por defecto para PonderacionConsonancia
    0: 0.0,  1: 1.3,  2: 1.0,  3: 0.4,
    4: 0.5,  5: 0.2,  6: 1.4,  7: 0.1,
    8: 0.6,  9: 0.7,  10: 1.1, 11: 1.2
}
PONDERACION_IMPORTANCIA_BAJO_WEIGHT_DEFAULT = 1.8
PONDERACION_IMPORTANCIA_FINAL_WEIGHT_DEFAULT = 1.5
PONDERACION_IMPORTANCIA_DECAY_FACTOR_DEFAULT = 0.3

DERIVE_FR_BASE_FREQ_DEFAULT = 440.0 # Frecuencia base por defecto para derive_fr
DERIVE_FR_MAX_DENOMINATOR_DEFAULT = 20

# --- Marcadores predeterminados por población ---
PLOT_MARKERS = {
    "dyads": {"label": "Diadas", "symbol": "circle", "size": 16, "line_width": 1, "line_color": "black"},
    "triads": {"label": "Triadas", "symbol": "star", "size": 20, "line_width": 2, "line_color": "black"},
    "other": {"label": "Otros", "symbol": "diamond", "size": 14, "line_width": 1, "line_color": "black"},
}


# --- Parámetros de Reducción Dimensional ---
UMAP_N_COMPONENTS = 2
UMAP_RANDOM_STATE = 42
MDS_N_COMPONENTS = 2
MDS_DISSIMILARITY = 'precomputed'
MDS_RANDOM_STATE = 42
MDS_NORMALIZED_STRESS = 'auto'
KERNEL_MDS_N_COMPONENTS = 2

# --- Parámetro de Métricas de Evaluación ---
EVAL_N_NEIGHBORS = 3

# --- Valores Predeterminados de Visualización ---
DEFAULT_COLORSCALE = 'Viridis'
HEATMAP_COLORSCALE = 'Viridis'
UMAP_GRAPH_NODE_COLORSCALE = 'YlGnBu'
UMAP_GRAPH_EDGE_COLOR = '#888'
# --- Parámetros de Visualización de Grafos UMAP ---
UMAP_GRAPH_NODE_COLORSCALE = 'Viridis'


# --- Credenciales de la Base de Datos ---
DB_HOST = _get_env("DB_HOST", "localhost")
DB_PORT = _get_env("DB_PORT", "5432")
DB_USER = _get_env("DB_USER", "postgres")
DB_PASSWORD = _get_env("DB_PASSWORD", "postgres")
DB_NAME = _get_env("DB_NAME", "ChordCodex")
config_db = {
    "host": DB_HOST,
    "port": int(DB_PORT),  # Convertir a entero
    "user": DB_USER,
    "password": DB_PASSWORD,
    "dbname": DB_NAME
}


# --- Opciones para Dropdown Widgets de la Interfaz de Usuario ---
MODELO_OPTIONS_LIST = [  # Opciones para el Dropdown de Modelos
    ("Sethares", "Sethares"),
    ("Euler", "Euler"),
    ("Armónicos Criticos", "ArmonicosCriticos")
]

PONDERACION_OPTIONS_LIST = [ # Opciones para el Dropdown de Ponderaciones
    ("Sin ponderación", "ninguna"),
    ("Consonancia", "consonancia"),
    ("Importancia Perceptual", "importancia"),
    ("Combinada (Cons. + Imp.)", "combinada")
]

METRICA_OPTIONS_LIST = [ # Opciones para el Dropdown de Métricas
    ("Euclidean", "euclidean"),
    ("Cosine", "cosine"),
    ("Cityblock", "cityblock"),
    ("Chebyshev", "chebyshev"),
    ("Custom (1-correlación)", "custom")
]

REDUCCION_OPTIONS_LIST = [
    
    ("MDS", "MDS"),
    ("UMAP", "UMAP"),
    ("Geometric MDS", "Geometric MDS"),
    ("Kernel MDS", "Kernel MDS"),
    ("Representación Sp–Sf", "RepresentacionSpSf")  # Nueva opción
]


def _intervals_to_semitone_set(intervals):
    total = 0
    semitones = {0}
    for step in intervals:
        total = (total + int(step)) % 12
        semitones.add(total)
    return frozenset(semitones)

_CHORD_TEMPLATE_DEFINITIONS = [
    # Dyads (2-note intervals)
    ("Unison", (0,), ["P1", "Unisono", "Unison"]),
    ("m2", (1,), ["minor second", "b2", "2m"]),
    ("M2", (2,), ["major second", "2", "2M"]),
    ("m3", (3,), ["minor third", "b3", "3m"]),
    ("M3", (4,), ["major third", "3", "3M"]),
    ("P4", (5,), ["perfect fourth", "4"]),
    ("TT", (6,), ["tritone", "A4", "d5"]),
    ("P5", (7,), ["perfect fifth", "5"]),
    ("m6", (8,), ["minor sixth", "b6", "6m"]),
    ("M6", (9,), ["major sixth", "6", "6M"]),
    ("m7", (10,), ["minor seventh", "b7", "7m"]),
    ("M7", (11,), ["major seventh", "7", "7M"]),
    ("Octave", (12,), ["P8", "Octava", "8ve"]),
    # Triads and extended chords
    ("Major", (4, 3), ["Maj", "M"]),
    ("Minor", (3, 4), ["min", "m", "-"]),
    ("Dim", (3, 3), ["dim", "o"]),
    ("Aug", (4, 4), ["aug", "+"]),
    ("Sus2", (2, 5), ["sus2"]),
    ("Sus4", (5, 2), ["sus4"]),
    ("Add9", (4, 3, 7), ["add9", "add2"]),
    ("6", (4, 3, 2), ["6", "add6"]),
    ("m6", (3, 4, 2), ["m6"]),
    ("6/9", (4, 3, 2, 5), ["6/9", "69"]),
    ("Maj7", (4, 3, 4), ["maj7", "M7"]),
    ("7", (4, 3, 3), ["7", "dom7"]),
    ("m7", (3, 4, 3), ["m7", "min7", "-7"]),
    ("mM7", (3, 4, 4), ["m(maj7)", "mMaj7"]),
    ("m7b5", (3, 3, 4), ["m7b5", "half-dim"]),
    ("Dim7", (3, 3, 3), ["dim7", "o7"]),
    ("AugMaj7", (4, 4, 3), ["aug(maj7)", "+maj7"]),
    ("Aug7", (4, 4, 2), ["aug7", "+7", "7#5"]),
    ("7b9", (4, 3, 3, 3), ["7(b9)"]),
    ("7#9", (4, 3, 3, 5), ["7(#9)"]),
    ("7#11", (4, 3, 3, 8), ["7(#11)"]),
    ("Maj9", (4, 3, 4, 3), ["maj9", "M9"]),
    ("9", (4, 3, 3, 4), ["9", "dom9"]),
    ("m9", (3, 4, 3, 4), ["m9", "min9"]),
    ("Maj11", (4, 3, 4, 7), ["maj11", "M11", "maj7#11"]),
    ("m11", (3, 4, 3, 4, 3), ["m11"]),
    ("11", (4, 3, 3, 4, 3), ["11", "dom11"]),
    ("Maj13", (4, 3, 4, 3, 7), ["maj13", "M13"]),
    ("13", (4, 3, 3, 4, 7), ["13", "dom13"]),
    ("m13", (3, 4, 3, 4, 3, 4), ["m13"]),
]

CHORD_TEMPLATES_METADATA = []
for name, intervals, aliases in _CHORD_TEMPLATE_DEFINITIONS:
    interval_tuple = tuple(int(step) for step in intervals)
    semitone_set = sorted(_intervals_to_semitone_set(interval_tuple))
    CHORD_TEMPLATES_METADATA.append(
        {
            "name": name,
            "aliases": [alias.strip() for alias in aliases if alias.strip()],
            "intervals": interval_tuple,
            "semitones": semitone_set,
        }
    )

_CHORD_TEMPLATE_BY_INTERVAL = {tpl["intervals"]: tpl for tpl in CHORD_TEMPLATES_METADATA}
CHORD_TYPE_INTERVALS = {
    intervals: {
        "name": tpl["name"],
        "aliases": list(tpl["aliases"]),
        "semitones": list(tpl["semitones"]),
    }
    for intervals, tpl in _CHORD_TEMPLATE_BY_INTERVAL.items()
}

# Toggle global para activar/desactivar la heuristica por scoring en nombrado de acordes
# Toggle global para activar/desactivar la heuristica por scoring en nombrado de acordes
CHORD_NAMING_USE_SCORING_DEFAULT = False


# Re-declaración (actualizada) de opciones de modelos para UI, incluyendo la variante vectorizada
MODELO_OPTIONS_LIST = [  # Opciones para el Dropdown de Modelos (actualizado)
    ("Sethares", "Sethares"),
    ("Sethares (Vec)", "SetharesVec"),
    ("Euler", "Euler"),
    ("Armónicos Criticos", "ArmonicosCriticos"),
]
# -------------------------------
# 2. Consultas SQL
# -------------------------------


# --- Consultas SQL ---
QUERY_CHORDS_3_NOTES = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE array_length(notes, 1) = 3
ORDER BY id
LIMIT 60;
"""

# --- Nuevas consultas para experimentos por cardinalidad --------------------

QUERY_CHORDS_3_NOTES_ALL = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE n = 3
ORDER BY id;
"""

# Conjuntos completos por cardinalidad (para selección de población en la GUI)
QUERY_CHORDS_4_NOTES_ALL = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE n = 4
ORDER BY id;
"""

QUERY_CHORDS_5_NOTES_ALL = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE n = 5
ORDER BY id;
"""

QUERY_CHORDS_4_NOTES_SAMPLE_25 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 4
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.25) FROM chords WHERE n = 4);
"""

QUERY_CHORDS_4_NOTES_SAMPLE_50 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 4
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.50) FROM chords WHERE n = 4);
"""

QUERY_CHORDS_4_NOTES_SAMPLE_75 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 4
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.75) FROM chords WHERE n = 4);
"""

QUERY_CHORDS_5_NOTES_SAMPLE_25 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 5
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.25) FROM chords WHERE n = 5);
"""

QUERY_CHORDS_5_NOTES_SAMPLE_50 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 5
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.50) FROM chords WHERE n = 5);
"""

QUERY_CHORDS_5_NOTES_SAMPLE_75 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 5
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.75) FROM chords WHERE n = 5);
"""
QUERY_CHORDS_WITH_NAME = """
SELECT *
FROM chords
WHERE interval IN (
    ARRAY[2,5], ARRAY[3,3], ARRAY[3,4], ARRAY[4,3],
    ARRAY[4,4], ARRAY[5,2], ARRAY[3,3,3], ARRAY[3,3,4],
    ARRAY[3,4,2], ARRAY[3,4,3], ARRAY[3,4,4], ARRAY[4,3,2],
    ARRAY[4,3,3], ARRAY[4,3,4], ARRAY[4,3,7], ARRAY[4,4,2],
    ARRAY[4,4,3], ARRAY[3,4,3,4], ARRAY[4,3,2,5], ARRAY[4,3,3,3],
    ARRAY[4,3,3,4], ARRAY[4,3,3,5], ARRAY[4,3,3,8], ARRAY[4,3,4,3],
    ARRAY[4,3,4,7], ARRAY[3,4,3,4,3], ARRAY[4,3,3,4,3], ARRAY[4,3,3,4,7],
    ARRAY[4,3,4,3,7], ARRAY[3,4,3,4,3,4]
)
AND ('x' || notes[1])::bit(32)::int = 0  -- Solo acordes que empiezan en la nota '0'
ORDER BY id;  -- Ordenar por ID para facilitar la lectura

"""


QUERY_TRIADS_WITH_INVERSIONS= """
WITH base_chords AS (
    -- Extraemos los acordes filtrados con sus notas ordenadas
    SELECT DISTINCT notes,
           ARRAY[notes[2], notes[3], notes[1]] AS inversion_1, -- 1ra inversión
           ARRAY[notes[3], notes[1], notes[2]] AS inversion_2  -- 2da inversión
    FROM chords
    WHERE n = 3
    AND (interval = ARRAY[4,3]::integer[] OR
         interval = ARRAY[3,4]::integer[] OR
         interval = ARRAY[3,3]::integer[])
    AND notes <@ ARRAY['0','2','4','5','7','9','B']::varchar[]
)
SELECT *
FROM chords
WHERE notes IN (
    SELECT notes FROM base_chords
    UNION
    SELECT inversion_1 FROM base_chords
    UNION
    SELECT inversion_2 FROM base_chords
)
ORDER BY notes, id;
"""

# Consultas adicionales (migradas desde el notebook)

# Triadas (mayor, menor, disminuida) restringidas a raíces diatónicas
QUERY_TRIADS_ROOT_ONLY_MOBIUS_MAZZOLA = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE n = 3
  AND interval IN (
      ARRAY[4,3],  -- mayor
      ARRAY[3,4],  -- menor
      ARRAY[3,3]   -- disminuida
  )
  AND notes <@ ARRAY['0','2','4','5','7','9','B']::varchar[]
ORDER BY notes, id;
"""

QUERY_CHORDS_3_NOTES_SAMPLE_25 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 3
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.25) FROM chords WHERE n = 3);
"""

QUERY_CHORDS_3_NOTES_SAMPLE_50 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 3
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.50) FROM chords WHERE n = 3);
"""

QUERY_CHORDS_3_NOTES_SAMPLE_75 = """
WITH subset AS (
    SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
    FROM chords
    WHERE n = 3
    ORDER BY RANDOM()
)
SELECT *
FROM subset
LIMIT (SELECT CEIL(COUNT(*) * 0.75) FROM chords WHERE n = 3);
"""

# Aleatorio con n > 2
QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE n > 2
ORDER BY RANDOM()
LIMIT 100;
"""

# Aleatorio dentro de misma octava (sum(interval) <= 12), raíz '0'
QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES_CLOSE_OCTAVE = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE n >= 2
  AND ('x' || notes[1])::bit(32)::int = 0
  AND (SELECT SUM(i) FROM unnest(interval) AS i) <= 12
ORDER BY RANDOM()
LIMIT 1000;
"""

# Triads with repeated notes (root 0), span <= 12, random sample
QUERY_TRIADS_WITH_REPEATED_NOTES = """
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM chords
WHERE n = 3
  AND ('x' || notes[1])::bit(32)::int = 0
  AND (SELECT SUM(i) FROM unnest(interval) AS i) <= 12
  AND (
    SELECT COUNT(*) FROM (
      SELECT unnest(notes) AS note
    ) sub
    GROUP BY note
    HAVING COUNT(*) > 1
    LIMIT 1
  ) IS NOT NULL
ORDER BY RANDOM()
LIMIT 100;
"""

# Extreme clusters of 10 notes: all-ones interval and 10-note variants with steps in {1,2}
QUERY_EXTREME_CLUSTER_10_NOTES = """
(
  SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
  FROM chords
  WHERE n = 10
    AND interval = ARRAY[1,1,1,1,1,1,1,1,1]
)
UNION
(
  SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
  FROM chords
  WHERE n = 10
    AND (
      SELECT bool_and(i BETWEEN 1 AND 2)
      FROM unnest(interval) AS i
    )
    AND interval != ARRAY[1,1,1,1,1,1,1,1,1]
  ORDER BY RANDOM()
  LIMIT 30
);
"""

# One dyad per semitone 1..11 plus the unison root 0 (random pick per class)
QUERY_DYADS_RANDOM_UNIQUE = """
WITH octava AS (
  SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
  FROM chords
  WHERE n = 2
    AND notes = ARRAY['0', '0']
    AND ('x' || notes[1])::bit(32)::int = 0
    LIMIT 1
),
all_dyads AS (
  SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code,
         interval[1] AS total_interval
  FROM chords
  WHERE n = 2
    AND notes != ARRAY['0', '0']
    AND ('x' || notes[1])::bit(32)::int = 0
    AND (SELECT SUM(i) FROM unnest(interval) AS i) BETWEEN 1 AND 11
),
ranked_dyads AS (
  SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code, total_interval,
         ROW_NUMBER() OVER (PARTITION BY total_interval ORDER BY RANDOM()) AS rn
  FROM all_dyads
)
SELECT id, n, interval, notes, bass, octave, frequencies, chroma, tag, code
FROM ranked_dyads
WHERE rn = 1
UNION ALL
SELECT * FROM octava
ORDER BY code;
"""


# Catálogo de díadas (unísono + una por semitono 1..12)
QUERY_DYADS_REFERENCE = """
WITH unison AS (
    SELECT
        c.id, c.n, c.interval, c.notes, c.bass, c.octave, c.frequencies,
        c.chroma, c.tag, c.code,
        'unison' AS quality,
        0 AS semitone,
        ROW_NUMBER() OVER (ORDER BY c.octave, c.id) AS rn
    FROM chords c
    WHERE c.n = 2
      AND c.notes = ARRAY['0','0']::varchar[]
),
non_unison AS (
    SELECT
        c.id, c.n, c.interval, c.notes, c.bass, c.octave, c.frequencies,
        c.chroma, c.tag, c.code,
        'dyad' AS quality,
        c.interval[1] AS semitone,
        ROW_NUMBER() OVER (
            PARTITION BY c.interval[1]
            ORDER BY c.octave, c.id
        ) AS rn
    FROM chords c
    WHERE c.n = 2
      AND c.notes[1] = '0'
      AND c.notes <> ARRAY['0','0']::varchar[]
      AND array_length(c.interval, 1) = 1
      AND c.interval[1] BETWEEN 1 AND 12
)
SELECT *
FROM (
    SELECT * FROM unison WHERE rn = 1
    UNION ALL
    SELECT * FROM non_unison WHERE rn = 1
) AS pick
ORDER BY semitone;
"""

# Catálogo canónico de triadas: una por (calidad, raíz)
QUERY_TRIADS_CORE = """
WITH triad_catalog(quality, intervals) AS (
    VALUES
        ('Maj', ARRAY[4,3]::integer[]),
        ('Min', ARRAY[3,4]::integer[]),
        ('Dim', ARRAY[3,3]::integer[]),
        ('Aug', ARRAY[4,4]::integer[]),
        ('Sus2', ARRAY[2,5]::integer[]),
        ('Sus4', ARRAY[5,2]::integer[])
),
ranked AS (
    SELECT
        c.id, c.n, c.interval, c.notes, c.bass, c.octave, c.frequencies,
        c.chroma, c.tag, c.code,
        triad_catalog.quality,
        c.notes[1] AS root,
        ROW_NUMBER() OVER (
            PARTITION BY triad_catalog.quality, c.notes[1]
            ORDER BY c.octave, c.id
        ) AS rn
    FROM chords c
    JOIN triad_catalog ON c.interval = triad_catalog.intervals
    WHERE c.n = 3
)
SELECT *
FROM ranked
WHERE rn = 1
ORDER BY quality, root;
"""

# Variante controlada: 7 triadas diatónicas (root en {0,2,4,5,7,9,B}) y sus 2 inversiones = 21 acordes
# Nota: Se retiró QUERY_TRIADS_WITH_INVERSIONS_21 porque la base sólo contiene raíz '0'.
QUERY_CHORDS_WITH_NAME_AND_RANDOM_CHORDS_POBLATION = """
WITH chords_filtered AS (
    -- Selección de acordes con los intervalos específicos que empiezan en la nota '0'
    SELECT *
    FROM chords
    WHERE interval IN (
        ARRAY[4,3], ARRAY[3,4], ARRAY[2,5],
        ARRAY[5,2], ARRAY[4,4], ARRAY[3,3], ARRAY[4,2],
        ARRAY[4,3,4], ARRAY[3,4,3], ARRAY[4,3,3], ARRAY[3,3,4],
        ARRAY[3,4,4], ARRAY[4,4,2], ARRAY[3,3,3], ARRAY[4,3,4,3],
        ARRAY[3,4,3,4], ARRAY[4,3,3,4], ARRAY[4,3,3,1], ARRAY[4,3,3,3],
        ARRAY[4,3,4,3,4], ARRAY[3,4,3,4,3], ARRAY[4,3,3,4,3],
        ARRAY[5,2,3,3], ARRAY[4,3,4,3,4,3], ARRAY[3,4,3,4,3,4],
        ARRAY[4,3,3,4,3,4], ARRAY[4,2,4], ARRAY[4,3,3,5,1], ARRAY[4,3,7],
        ARRAY[5,4]
    )
    AND ('x' || notes[1])::bit(32)::int = 0  -- ✅ Solo acordes que empiezan en la nota '0'
),
random_chords AS (
    -- Selección de 60 acordes aleatorios que NO están en la primera selección
    SELECT *
    FROM chords
    WHERE id NOT IN (SELECT id FROM chords_filtered)  -- 🔥 Excluir los acordes ya seleccionados
    ORDER BY RANDOM()
    LIMIT 60
)
-- Unión de ambas selecciones
SELECT * FROM chords_filtered
UNION
SELECT * FROM random_chords
ORDER BY id;

"""
# Usado en: CELDA 11 (carga de datos) y en pre_process.py para definir la extracción de acordes.

QUERY_CHORDS_SPECIFIC_INTERVALS_AND_RANDOM_SAME_OCTAVE = """
WITH chords_filtered AS (
    SELECT *
    FROM chords
    WHERE interval IN (
        ARRAY[4,3], ARRAY[3,4], ARRAY[2,5],
        ARRAY[5,2], ARRAY[4,4], ARRAY[3,3], ARRAY[4,2],
        ARRAY[4,3,4], ARRAY[3,4,3], ARRAY[4,3,3], ARRAY[3,3,4],
        ARRAY[3,4,4], ARRAY[4,4,2], ARRAY[3,3,3], ARRAY[4,3,4,3],
        ARRAY[3,4,3,4], ARRAY[4,3,3,4], ARRAY[4,3,3,1], ARRAY[4,3,3,3],
        ARRAY[4,3,4,3,4], ARRAY[3,4,3,4,3], ARRAY[4,3,3,4,3],
        ARRAY[5,2,3,3], ARRAY[4,3,4,3,4,3], ARRAY[3,4,3,4,3,4],
        ARRAY[4,3,3,4,3,4], ARRAY[4,2,4], ARRAY[4,3,3,5,1], ARRAY[4,3,7],
        ARRAY[5,4]
    )
    AND ('x' || notes[1])::bit(32)::int = 0
),
random_chords AS (
    SELECT *
    FROM chords
    WHERE id NOT IN (SELECT id FROM chords_filtered)
      AND (SELECT SUM(i) FROM unnest(interval) AS i) <= 12
    ORDER BY RANDOM()
    LIMIT 400
)
SELECT *
FROM chords_filtered
UNION
SELECT *
FROM random_chords
ORDER BY id;
"""



# -------------------------------
# 3. Parámetros para Evaluación y Reducción Dimensional
# -------------------------------
# Usado en: metrics.py, CELDA 10 y en la ejecución de experimentos (LaboratorioAcordes).
EVAL_N_NEIGHBORS = 5             # Número de vecinos para las métricas de evaluación (trustworthiness, continuity, etc.).
UMAP_N_COMPONENTS = 2            # Número de componentes para UMAP.
KERNEL_MDS_N_COMPONENTS = 2      # Número de componentes para Kernel MDS.

# -------------------------------
# 4. Parámetros para Ponderaciones de Rugosidad
# -------------------------------
# Usado en: pre_process.py, en las clases PonderacionConsonancia, PonderacionImportanciaPerceptual y PonderacionCombinada.
PONDERACION_IMPORTANCIA_BAJO_WEIGHT_DEFAULT = 1.5   # Peso para el primer intervalo en la ponderación de importancia perceptual.
PONDERACION_IMPORTANCIA_FINAL_WEIGHT_DEFAULT = 1.5   # Peso para el último intervalo.
PONDERACION_IMPORTANCIA_DECAY_FACTOR_DEFAULT = 0.5   # Factor de decaimiento para los intervalos intermedios.

# -------------------------------
# 5. Parámetros para Visualización
# -------------------------------
# Usado en: visualization.py y en la configuración de widgets en ui.py.
# Escala de colores para representar la rugosidad en gráficos (scatter, Sp–Sf, UMAP, etc.).
COMMON_RUG_COLORSCALE = [[0, '#a6cee3'], [1, '#fb9a99']]

# Configuración de la barra de color para la visualización de rugosidad.
COMMON_COLORBAR = {
    "title": "Rugosidad Total",
    "tickmode": "auto",
    "x": 1.15
}

# Parámetros para el contorno de densidad en gráficos (usados en density_contour de visualization.py).
DENSITY_NBINS_X = 30
DENSITY_NBINS_Y = 30
DENSITY_COLOR_SCALE = "Blues"
DENSITY_OPACITY = 0.3

# Parámetros para Scatter Plot (usados en visualizar_scatter_density de visualization.py).
MARKER_SIZE_KNOWN = 12       # Tamaño para acordes "Con Nombre".
MARKER_SIZE_UNKNOWN = 8      # Tamaño para acordes "Desconocido".
MARKER_LINE_KNOWN = {"color": "black", "width": 2}  # Contorno para acordes conocidos.
SCATTER_COLOR_SCALE = "Reds" # Escala de colores para scatter plot.

# Layout y posicionamiento en gráficos (usado en visualization.py para ajustar leyendas y barras de color).
LAYOUT_LEGEND = {"x": 1.02, "y": 1.14}
LAYOUT_COLORBAR = {"title": "TotalRug", "x": 1.02}

# -------------------------------
# 6. Otros Parámetros Globales
# -------------------------------
# Usado en: pre_process.py (función derive_fr) y en modelos de rugosidad.
SETHARES_BASE_FREQ = 261.63    # Frecuencia base (C4 en Hz), usada en cálculos de modelos de rugosidad.

# Fin de config.py
