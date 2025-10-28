# pre_process.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Set
from fractions import Fraction
import ast
import math
from functools import reduce
from scipy.ndimage import gaussian_filter1d
from itertools import combinations
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from chordcodex.model import QueryExecutor  # type: ignore
except Exception:  # pragma: no cover
    # Fallback local executor (usa psycopg2 directamente)
    from synth_tools import QueryExecutor  # type: ignore
# Importar constantes globales desde config.py
from config import (
    PONDERACION_IMPORTANCIA_BAJO_WEIGHT_DEFAULT,
    PONDERACION_IMPORTANCIA_FINAL_WEIGHT_DEFAULT,
    PONDERACION_IMPORTANCIA_DECAY_FACTOR_DEFAULT,
    PONDERACION_CONSONANCIA_DEFAULT_WEIGHTS,
    DERIVE_FR_BASE_FREQ_DEFAULT,
    DERIVE_FR_MAX_DENOMINATOR_DEFAULT,
    SETHARES_BASE_FREQ,
    SETHARES_D_STAR,
    SETHARES_S1,
    SETHARES_S2,
    SETHARES_C1,
    SETHARES_C2,
    SETHARES_A1,
    SETHARES_A2,
    SETHARES_N_HARMONICS,
    SETHARES_DECAY,
    SETHARES_BASE_WEIGHTS,
    SETHARES_CONSONANT_INTERVALS,
    UMAP_N_COMPONENTS,
    KERNEL_MDS_N_COMPONENTS,
    EVAL_N_NEIGHBORS,
    EULER_CONSONANCE_WEIGHTS,
    EULER_CONSONANCIA_WEIGHTS_DEFAULT,
    EULER_PRIME_FACTORS_CACHE,
    ARMONICOS_CRITICOS_BASE_FREQ_DEFAULT,
    CHORD_TYPE_INTERVALS
)



# --- Funciones Utilitarias Generales ---
def clean_vector(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Limpia un vector numérico, reemplazando NaNs con ceros y asegurando norma no nula.

    Args:
        vec (np.ndarray): Vector de entrada.
        eps (float, opcional): Epsilon para asegurar norma no nula. Por defecto 1e-8.

    Returns:
        np.ndarray: Vector limpio.
    """
    vec_clean = np.nan_to_num(vec, nan=0.0)
    norm = np.linalg.norm(vec_clean)
    if norm < eps:
        return vec_clean + eps
    return vec_clean


def safe_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normaliza un vector numérico de forma segura, evitando división por cero.

    Args:
        vec (np.ndarray): Vector a normalizar.
        eps (float, opcional): Epsilon para evitar división por cero. Por defecto 1e-8.

    Returns:
        np.ndarray: Vector normalizado o vector original si la norma es muy pequeña.
    """
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def interval_to_ui_bin(intervalo: int) -> int:
    """
    Mapea un intervalo en semitonos (mod 12) al índice de bin usado en el vector 12‑D.

    Convención de UI actual: la "posición 12" (índice 11) representa el intervalo 0 (unísono/octava).
    Para el resto, 1..11 -> índices 0..10.

    Args:
        intervalo: entero en [0..11]

    Returns:
        int: índice de bin en [0..11]
    """
    return (intervalo - 1) % 12  # 0 -> 11, 1..11 -> 0..10


from config import CHORD_TYPE_INTERVALS
from typing import List

@dataclass(frozen=True)
class ChordIdentity:
    name: str
    aliases: Tuple[str, ...] = ()

def _intervals_to_semitone_set(intervals: Tuple[int, ...]) -> Set[int]:
    total = 0
    semitones: Set[int] = {0}
    for step in intervals:
        total = (total + int(step)) % 12
        semitones.add(total)
    return semitones

def get_chord_type_from_intervals(intervals: List[int], *, with_alias: bool = False) -> Union[str, "ChordIdentity"]:
    interval_tuple = tuple(int(i) for i in intervals)
    entry = CHORD_TYPE_INTERVALS.get(interval_tuple)
    identity: Optional[ChordIdentity] = None
    if entry:
        expected = set(entry.get("semitones", []))
        actual = _intervals_to_semitone_set(interval_tuple)
        if not expected or expected == actual:
            identity = ChordIdentity(
                name=entry.get("name", "Unknown"),
                aliases=tuple(entry.get("aliases", ())),
            )
    if identity is None:
        identity = ChordIdentity(name="Unknown", aliases=())
    return identity if with_alias else identity.name



def lcm(a, b):
    """
    Calcula el mínimo común múltiplo (MCM) de dos números enteros.

    Args:
        a (int): Primer número entero.
        b (int): Segundo número entero.

    Returns:
        int: Mínimo común múltiplo de a y b.
    """
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def lcm_list(nums):
    """
    Calcula el mínimo común múltiplo (MCM) de una lista de números enteros.

    Args:
        nums (list of int): Lista de números enteros.

    Returns:
        int: Mínimo común múltiplo de todos los números en la lista.
    """
    return reduce(lcm, nums, 1)


# --- Adaptación de Datos ---
@dataclass
class Acorde:
    """
    Representa un acorde musical.

    Atributos:
      - name (str): Nombre del acorde.
      - intervals (List[int]): Intervalos en semitonos.
      - chroma (np.ndarray): Vector chroma (12 dimensiones).
      - frequencies (Optional[List[float]]): Lista de frecuencias (si proviene de la DB).
      - total_roughness (Optional[float]): Escalar de rugosidad total (se asigna tras el cálculo).
    """
    name: str
    intervals: List[int]
    chroma: np.ndarray = None
    frequencies: Optional[List[float]] = None
    notes: Optional[List[str]] = None
    total_roughness: Optional[float] = None

    def __post_init__(self):
        if self.chroma is None:
            self.chroma = self.compute_chroma()
        if not hasattr(self, 'frequencies'):
            self.frequencies = None

    def compute_chroma(self) -> np.ndarray:
        """
        Calcula el vector chroma del acorde basado en sus intervalos.
        """
        if self.chroma is not None:
            return self.chroma
        if not self.intervals:
            return np.zeros(12, dtype=int)
        semitonos = np.cumsum([0] + self.intervals)
        pitch_classes = np.mod(semitonos, 12)
        chroma = np.zeros(12, dtype=int)
        chroma[np.unique(pitch_classes)] = 1
        return chroma


class ChordAdapter:
    """
    Adaptador para crear objetos Acorde a partir de diferentes fuentes de datos.
    """
    @staticmethod
    def from_csv_row(row: Union[pd.Series, Dict]) -> Acorde:
        """
        Crea un objeto Acorde a partir de una fila de pandas Series o un diccionario.

        Extrae 'interval', 'chroma', 'code', y 'frequencies' de la fila.
        """
        try:
            intervals = ast.literal_eval(row.get('interval', '[]')) if isinstance(row.get('interval'), str) else row.get('interval', [])
            chroma = np.array(ast.literal_eval(row.get('chroma', '[0]*12'))) if isinstance(row.get('chroma'), str) else np.array(row.get('chroma', [0]*12))
            frequencies = ast.literal_eval(row.get('frequencies', 'None')) if isinstance(row.get('frequencies'), str) else row.get('frequencies', None)
            notes = ast.literal_eval(row.get('notes', 'None')) if isinstance(row.get('notes'), str) else row.get('notes', None)

        except Exception as e:
            logging.error(f"Error al procesar fila para Acorde: {e}. Usando valores por defecto.")
            intervals = []
            chroma = np.zeros(12, dtype=int)
            frequencies = None
        #print(f"[debug] extrayendo notes = {notes}")

        return Acorde(
            name=row.get('code', 'Sin nombre'),
            intervals=intervals,
            chroma=chroma,
            frequencies=frequencies,
            notes=notes
        )


def load_chord_data_from_db(query_executor: QueryExecutor, query: str) -> pd.DataFrame:
    """
    Carga datos de acordes desde una base de datos usando chordcodex.QueryExecutor.
    """
    try:
        df_chords = query_executor.as_pandas(query)
        print(f"[info] Datos de acordes cargados desde la base de datos. Número de registros: {len(df_chords)}")
        return df_chords
    except Exception as e:
        logging.error(f"Error al cargar datos de la base de datos: {e}")
        return pd.DataFrame() # Retorna DataFrame vacío en caso de error


# --- Modelos de Rugosidad ---
class ModeloRugosidad(ABC):
    """
    Clase abstracta base para modelos de cálculo de rugosidad musical.

    Define la interfaz para los modelos de rugosidad.
    """
    def __init__(self, config: Dict):
        """
        Inicializa el modelo de rugosidad con una configuración.
        """
        self.config = config

    @abstractmethod
    def calcular(self, acorde: 'Acorde') -> Tuple[np.ndarray, float]:
        """
        Método abstracto para calcular el vector de rugosidad y la rugosidad total de un acorde.

        Debe ser implementado por las subclases.
        """
        raise NotImplementedError("Subclase debe implementar este método.")


# --- ModeloSethares ---
"""

class ModeloSetharesObsoleto(ModeloRugosidad):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.Dstar = config.get('Dstar', SETHARES_D_STAR)
        self.S1 = config.get('S1', SETHARES_S1)
        self.S2 = config.get('S2', SETHARES_S2)
        self.C1 = config.get('C1', SETHARES_C1)
        self.C2 = config.get('C2', SETHARES_C2)
        self.A1 = config.get('A1', SETHARES_A1)
        self.A2 = config.get('A2', SETHARES_A2)
        self.pesos_base = config.get('pesos_base', SETHARES_BASE_WEIGHTS)
        self.consonantes = set(config.get('consonantes', SETHARES_CONSONANT_INTERVALS))

    def _calcular_disonancia_pairwise(self, f1, f2, a1, a2):
        Fmin = np.minimum(f1, f2)
        S = self.Dstar / (self.S1 * Fmin + self.S2)
        Fdif = np.abs(f2 - f1)
        # Usar producto de amplitudes (modo 'product') para alinear con el modelo teórico
        a = a1 * a2
        return a * (self.C1 * np.exp(self.A1 * S * Fdif) + self.C2 * np.exp(self.A2 * S * Fdif))

    def calcular(self, acorde: 'Acorde'):
        base_freq = self.config.get('base_freq', SETHARES_BASE_FREQ)
        # Parámetros canónicos (armónicos + decaimiento)
        n_harmonics = int(self.config.get('n_armonicos', SETHARES_N_HARMONICS))
        decaimiento = float(self.config.get('decaimiento', SETHARES_DECAY))
        n_harmonics = int(self.config.get('n_armonicos', SETHARES_N_HARMONICS))
        decaimiento = float(self.config.get('decaimiento', SETHARES_DECAY))
        # Parámetros del modelo (armónicos + decaimiento) con defaults de config
        n_harmonics = int(self.config.get('n_armonicos', SETHARES_N_HARMONICS))
        decaimiento = float(self.config.get('decaimiento', SETHARES_DECAY))
        n_harmonics = int(self.config.get('n_armonicos', SETHARES_N_HARMONICS))
        decaimiento = float(self.config.get('decaimiento', SETHARES_DECAY))
        n_armonicos = self.config.get('n_armonicos', SETHARES_N_HARMONICS)
        decaimiento = self.config.get('decaimiento', SETHARES_DECAY)
        chroma = np.array(acorde.chroma)

        # Usar frecuencias proporcionadas por la DB si están disponibles
        if acorde.frequencies is not None:
            frecuencias = np.array(acorde.frequencies)
            # Si la frecuencia viene como vector 1D, convertirla a 2D (1 fila)
            if frecuencias.ndim == 1:
                frecuencias = frecuencias.reshape(1, -1)
        else:
            semitonos = np.array(np.cumsum([0] + acorde.intervals))
            frecuencias = base_freq * 2 ** (semitonos.reshape(-1, 1) / 12) * (np.arange(1, n_armonicos + 1))

        #print("frecuencias=", frecuencias)

        amplitudes = decaimiento ** np.arange(n_armonicos)

        # Usar el número de filas (notas) en la matriz de frecuencias
        n_notes = frecuencias.shape[0]

        # Crear índices para todos los pares (flatten de la matriz)
        i, j = np.triu_indices(frecuencias.size, k=1)
        f1, f2 = frecuencias.flat[i], frecuencias.flat[j]

        # Repetir las amplitudes para cada nota (usando n_notes)
        a1 = np.repeat(amplitudes, n_notes)[i]
        a2 = np.repeat(amplitudes, n_notes)[j]
        #print("amplitudes=", a1)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(f2 > f1, f2 / f1, f1 / f2)
            semitonos_intervalo = (12 * np.log2(ratios)).round().astype(int) % 12
        #print("semitonos_intervalo=", semitonos_intervalo)
        contribuciones = self._calcular_disonancia_pairwise(f1, f2, a1, a2)
        #print("contribuciones=", len(contribuciones))
        fixed_amplitude = 1.0  # Define la amplitud fija (puedes cambiar este valor si quieres)
        #contribuciones = self._calcular_disonancia_pairwise(f1, f2, fixed_amplitude, fixed_amplitude) # Pasa amplitud fija en lugar de a1 y a2
        histograma = np.bincount(semitonos_intervalo, weights=contribuciones, minlength=12) + chroma
        from scipy.ndimage import gaussian_filter1d
        # Aplicar suavizado: sigma controla la fuerza del suavizado, ajusta según lo necesites.
        histograma_suavizado = gaussian_filter1d(histograma, sigma=0.5)

        # Calcular la rugosidad total a partir del histograma suavizado
        total_roughness = np.sum(np.bincount(semitonos_intervalo, weights=contribuciones, minlength=12))

        # Normalizar el vector de rugosidad suavizado
        vector_normalizado = safe_normalize(histograma)
        #print("histograma=", histograma)
        #total_roughness = np.sum(histograma)
        #vector_normalizado = safe_normalize(histograma)
        #print("Vector normalizado:", vector_normalizado)
        return vector_normalizado, total_roughness

"""

class ModeloSethares(ModeloRugosidad):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.Dstar = config.get('Dstar', SETHARES_D_STAR)
        self.S1 = config.get('S1', SETHARES_S1)
        self.S2 = config.get('S2', SETHARES_S2)
        self.C1 = config.get('C1', SETHARES_C1)
        self.C2 = config.get('C2', SETHARES_C2)
        self.A1 = config.get('A1', SETHARES_A1)
        self.A2 = config.get('A2', SETHARES_A2)
        self.pesos_base = config.get('pesos_base', SETHARES_BASE_WEIGHTS)
        self.consonantes = set(config.get('consonantes', SETHARES_CONSONANT_INTERVALS))
    
    def _calcular_disonancia_pairwise(self, f1, f2, a1, a2):
        """
        Calcula la disonancia entre dos notas según el modelo de Sethares.
        Parámetros:
          - f1, f2: frecuencias de las notas
          - a1, a2: amplitudes (en este caso, se usa 1.0 para ambas)
        """
        Fmin = np.minimum(f1, f2)
        S = self.Dstar / (self.S1 * Fmin + self.S2)
        Fdif = np.abs(f2 - f1)
        # Usar producto de amplitudes (modo 'product') para alinear con el modelo teórico
        a = a1 * a2
        return a * (self.C1 * np.exp(self.A1 * S * Fdif) + self.C2 * np.exp(self.A2 * S * Fdif))
    
    def calcular(self, acorde: 'Acorde'):
        """
        Calcula el vector de rugosidad y la rugosidad total del acorde utilizando
        únicamente las frecuencias fundamentales y una amplitud fija (1.0) para cada par.
        
        Se incluyen prints de debug para:
          - Mostrar los intervalos declarados y los semitonos resultantes.
          - Mostrar las frecuencias fundamentales.
          - Detallar, para cada par, el intervalo en semitonos, las frecuencias y la contribución.
          - Mostrar el histograma antes y después del suavizado, la rugosidad total y el vector normalizado.
        
        Retorna:
          - vector_normalizado: el histograma normalizado.
          - total_roughness: la rugosidad total (suma de las contribuciones).
        """
        base_freq = self.config.get('base_freq', SETHARES_BASE_FREQ)

        # Parámetros armónicos y flags de depuración (config opcional)
        n_harmonics = int(self.config.get('n_armonicos', SETHARES_N_HARMONICS))
        decaimiento = float(self.config.get('decaimiento', SETHARES_DECAY))
        debug = bool(self.config.get('debug_sethares', False))
        debug_max_pairs = self.config.get('debug_max_pairs', None)
        debug_print_hist = bool(self.config.get('debug_print_hist', False))
        printed_pairs = 0

        if debug:
            print("[DEBUG Sethares] intervals:", acorde.intervals)
        # 1) Calcular las posiciones (en semitonos) de las notas del acorde.
        semitonos_rel = np.array(np.cumsum([0] + acorde.intervals))
        if debug:
            print("[DEBUG Sethares] semitonos_rel:", semitonos_rel.tolist())
        n_notes = len(semitonos_rel)
        
        # 2) Obtener o calcular las frecuencias fundamentales.
        if acorde.frequencies is not None:
            fundamentals = np.array(acorde.frequencies)
            # Si vienen en 2D, tomamos la primera fila.
            if fundamentals.ndim > 1:
                fundamentals = fundamentals[0]
        else:
            fundamentals = base_freq * 2 ** (semitonos_rel / 12.0)
        #print("DEBUG: Frecuencias fundamentales:", fundamentals)
        
        if len(fundamentals) != n_notes:
            raise ValueError("El número de frecuencias no coincide con el número de notas deducidas de 'intervals'.")
        
        # 3) Inicializar el histograma de 12 bins y el acumulador de total por díadas
        histograma = np.zeros(12, dtype=float)
        total_roughness_pairs = 0.0
        # Asegurar parámetros locales definidos (algunos entornos no muestran las líneas superiores)
        try:
            n_harmonics
        except NameError:
            n_harmonics = int(self.config.get('n_armonicos', SETHARES_N_HARMONICS))
        try:
            decaimiento
        except NameError:
            decaimiento = float(self.config.get('decaimiento', SETHARES_DECAY))

        for i in range(n_notes - 1):
            for j in range(i + 1, n_notes):
                intervalo = int((semitonos_rel[j] - semitonos_rel[i]) % 12)
                bin_idx = interval_to_ui_bin(intervalo)  # 0->11, 1..11->0..10
                f1, f2 = fundamentals[i], fundamentals[j]
                # Sumar disonancia entre todos los parciales de las dos notas
                pair_total = 0.0
                for k1 in range(1, n_harmonics + 1):
                    for k2 in range(1, n_harmonics + 1):
                        p1 = f1 * k1
                        p2 = f2 * k2
                        a1 = decaimiento ** (k1 - 1)
                        a2 = decaimiento ** (k2 - 1)
                        pair_total += self._calcular_disonancia_pairwise(p1, p2, a1, a2)
                histograma[bin_idx] += pair_total
                total_roughness_pairs += pair_total
                if debug and (debug_max_pairs is None or printed_pairs < int(debug_max_pairs)):
                    printed_pairs += 1
                    print((
                        f"[DEBUG Sethares] pair (i={i}, j={j}) | semitonos=({int(semitonos_rel[i])}, {int(semitonos_rel[j])}) "
                        f"intervalo={intervalo} -> bin={bin_idx} | f1={float(f1):.3f}Hz, f2={float(f2):.3f}Hz | "
                        f"pair_total={pair_total:.6f} | cumulative_total={total_roughness_pairs:.6f}"
                    ))
        #print(f"🧪 DEBUG: acorde.notes = {getattr(acorde, 'notes', None)}")

        # 💡 Agregar rugosidad por octavas cuando una nota se repite
        if False and hasattr(acorde, "frequencies") and acorde.frequencies is not None and isinstance(acorde.notes, list):

            
                    for i, j in combinations(range(n_notes), 2):  # Todos los pares posibles
                        if acorde.notes[i] == acorde.notes[j]:
                            f1, f2 = fundamentals[i], fundamentals[j]
                            contrib = self._calcular_disonancia_pairwise(f1, f2, 1.0, 1.0)
                            histograma[11] += contrib
                            #print(f"🎯 Aporte por octava entre notas repetidas (i={i}, j={j}): {acorde.notes[i]} → contrib={contrib:.6f}")
        #print("DEBUG: Histograma (antes del suavizado):", histograma)
        
        # 5) (Opcional) Suavizado del histograma (usar mode='wrap' por circularidad de 12 bins)
        smoothing_sigma = float(self.config.get('smoothing_sigma', 0.5))
        histograma_suavizado = gaussian_filter1d(histograma, sigma=smoothing_sigma, mode='wrap')
        if debug or debug_print_hist:
            print("[DEBUG Sethares] histograma(sin suavizar):", np.round(histograma, 6))
            print("[DEBUG Sethares] histograma(suavizado):   ", np.round(histograma_suavizado, 6))
        
        # 6) Calcular la rugosidad total como suma de díadas (antes de suavizados o extras)
        total_roughness = float(total_roughness_pairs)
        vector_normalizado =histograma
        
        #print("DEBUG: Rugosidad total:", total_roughness)
        #print("DEBUG: Vector normalizado:", vector_normalizado)
        
        return vector_normalizado, total_roughness


class ModeloSetharesVec(ModeloRugosidad):
    """
    Variante vectorizada del Modelo de Sethares (vectoriza parciales HxH por par de notas).
    - Misma interfaz y semántica que ModeloSethares.
    - Bins por intervalo entre fundamentales (no parciales), usando interval_to_ui_bin.
    - Retorna histograma 12D (sin normalizar) y total_roughness (suma de pares).
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        self.Dstar = config.get('Dstar', SETHARES_D_STAR)
        self.S1 = config.get('S1', SETHARES_S1)
        self.S2 = config.get('S2', SETHARES_S2)
        self.C1 = config.get('C1', SETHARES_C1)
        self.C2 = config.get('C2', SETHARES_C2)
        self.A1 = config.get('A1', SETHARES_A1)
        self.A2 = config.get('A2', SETHARES_A2)

    def _pair_total(self, f1: float, f2: float, n_harmonics: int, decay: float) -> float:
        K = np.arange(1, int(n_harmonics) + 1, dtype=float)
        A = decay ** (K - 1)
        P1 = f1 * K
        P2 = f2 * K
        P1g = P1[:, None]
        P2g = P2[None, :]
        Fmin = np.minimum(P1g, P2g)
        DF = np.abs(P2g - P1g)
        S = self.Dstar / (self.S1 * Fmin + self.S2)
        Aprod = (A[:, None] * A[None, :])
        Dmat = Aprod * (self.C1 * np.exp(self.A1 * S * DF) + self.C2 * np.exp(self.A2 * S * DF))
        return float(np.sum(Dmat))

    def calcular(self, acorde: 'Acorde') -> Tuple[np.ndarray, float]:
        base_freq = self.config.get('base_freq', SETHARES_BASE_FREQ)
        n_harmonics = int(self.config.get('n_armonicos', SETHARES_N_HARMONICS))
        decay = float(self.config.get('decaimiento', SETHARES_DECAY))

        semitonos_rel = np.array(np.cumsum([0] + acorde.intervals))
        n_notes = len(semitonos_rel)

        if acorde.frequencies is not None:
            fundamentals = np.array(acorde.frequencies)
            if fundamentals.ndim > 1:
                fundamentals = fundamentals[0]
        else:
            fundamentals = base_freq * 2 ** (semitonos_rel / 12.0)

        if len(fundamentals) != n_notes:
            raise ValueError("El número de frecuencias no coincide con las notas deducidas de 'intervals'.")

        histograma = np.zeros(12, dtype=float)
        total_roughness_pairs = 0.0

        for i in range(n_notes - 1):
            for j in range(i + 1, n_notes):
                intervalo = int((semitonos_rel[j] - semitonos_rel[i]) % 12)
                bin_idx = interval_to_ui_bin(intervalo)
                f1, f2 = float(fundamentals[i]), float(fundamentals[j])
                pair_total = self._pair_total(f1, f2, n_harmonics, decay)
                histograma[bin_idx] += pair_total
                total_roughness_pairs += pair_total

        return histograma, float(total_roughness_pairs)

# --- Canonical alias ---
# La implementación canónica de Sethares en este repositorio es la versión vectorial.
# Este alias asegura que `ModeloSethares` apunte a `ModeloSetharesVec` para unificar el uso.
ModeloSethares = ModeloSetharesVec


class ModeloEuler(ModeloRugosidad):
    """
    Implementación del modelo de rugosidad de Euler.
    """
    def __init__(self, config: Dict):
        """
        Inicializa el Modelo Euler con parámetros de configuración.

        Si no se proporcionan en config, se usan los pesos de consonancia por defecto definidos como constante.
        """
        super().__init__(config)
        self.prime_factors_cache = EULER_PRIME_FACTORS_CACHE # Usa cache global (podría ser problemático en multi-threading si se modifica)
        self.consonancia_weights = config.get('consonancia_weights', EULER_CONSONANCIA_WEIGHTS_DEFAULT)

    def _prime_factors(self, n):
        """
        Calcula los factores primos (2, 3, 5) de un número.
        """
        if n == 0:
            return {}
        factors = {}
        for prime in [2, 3, 5]:
            while n % prime == 0:
                factors[prime] = factors.get(prime, 0) + 1
                n //= prime
        return factors

    def calcular(self, acorde: 'Acorde') -> Tuple[np.ndarray, float]:
        """
        Calcula el vector de rugosidad y la rugosidad total para un acorde usando el modelo de Euler.
        """
        semitonos = np.array(np.cumsum([0] + acorde.intervals))
        pairs = list(combinations(semitonos, 2))
        intervalos = np.array([(j - i) % 12 for i, j in pairs])
        gradiente = np.zeros(12)
        for intervalo in intervalos:
            complexity = 1.0
            if intervalo != 0:
                if intervalo not in self.prime_factors_cache: # Calcula y cachea si no existe
                    self.prime_factors_cache[intervalo] = self._prime_factors(intervalo)
                for prime, exp in self.prime_factors_cache[intervalo].items():
                    complexity *= (prime ** exp) * (1 + 0.5 * (exp - 1))
            gradiente[intervalo] += complexity
        total_roughness = np.sum(gradiente)
        vector_normalizado = safe_normalize(gradiente)
        return vector_normalizado, total_roughness


class ModeloArmonicosCriticos(ModeloRugosidad):
    """
    Implementación del modelo de rugosidad basado en armónicos críticos.
    """
    def __init__(self, config: Dict):
        """
        Inicializa el Modelo Armónicos Críticos con parámetros de configuración.

        Si no se proporcionan en config, se usa la frecuencia base por defecto definida como constante.
        """
        super().__init__(config)
        self.pos_weights = config.get('posicion_weights') # No se usa en el metodo calcular, se mantiene por consistencia con el original
        self.base_freq = config.get('base_freq', ARMONICOS_CRITICOS_BASE_FREQ_DEFAULT)

    def calcular(self, acorde: 'Acorde') -> Tuple[np.ndarray, float]:
        """
        Calcula el vector de rugosidad y la rugosidad total para un acorde usando el modelo de Armónicos Críticos.
        """
        base_freq = self.base_freq
        if acorde.frequencies is not None:
            frecuencias = np.array(acorde.frequencies)
        else:
            semitonos = np.array(np.cumsum([0] + acorde.intervals))
            frecuencias = base_freq * 2 ** (semitonos / 12)
        f_min, f_max = frecuencias.min(), frecuencias.max()
        n_points = 1000 # Podría ser configurable si se justifica
        f_axis = np.linspace(f_min * 0.9, f_max * 1.1, n_points)
        spectrum = np.zeros(n_points)
        for f in frecuencias:
            bandwidth = 25 + 75 * (1 + 1.4 * (f/1000) ** 2) ** 0.69 # Banda ancha podría ser configurable si se justifica
            spectrum += np.exp(-((f_axis - f) ** 2) / (2 * (bandwidth / 2.3548) ** 2))
        derivative = np.abs(np.gradient(spectrum))
        roughness = np.trapz(derivative ** 2, f_axis)
        intervalos = np.array([(j - i) % 12 for i, j in combinations(np.cumsum([0] + acorde.intervals), 2)])
        histograma = np.bincount(intervalos, minlength=12)
        raw_vector = histograma * roughness
        total_roughness = np.sum(raw_vector)
        vector_normalizado = safe_normalize(raw_vector)
        return vector_normalizado, total_roughness



# --- Reportes ---
@dataclass
class ResultadoExperimento:
    nombre_modelo: str
    metricas: Dict[str, float]
    embeddings: np.ndarray
    matriz_distancias: np.ndarray
    tiempo_ejecucion: float
    reducer_obj: Optional[object] = None
    X_original: Optional[np.ndarray] = None

# Puedes agregar una cadena de documentación si lo deseas
ResultadoExperimento.__doc__ = (
    "Esta clase encapsula los resultados de un experimento de modelado de rugosidad,\n"
    "almacenando el nombre del modelo, las métricas obtenidas, los embeddings, la\n"
    "matriz de distancias, el tiempo de ejecución, el objeto reductor (si aplica)\n"
    "y los vectores originales."
)


# --- Ponderaciones ---
class PonderacionConsonancia:
    """
    Pondera el vector de rugosidad basado en pesos de consonancia por intervalo.
    """
    def __init__(self, consonancia_weights: Optional[Dict[int, float]] = None):
        """
        Inicializa PonderacionConsonancia con pesos de consonancia.

        Si no se proporcionan, se usan los pesos por defecto definidos como constante.
        """
        default_weights = PONDERACION_CONSONANCIA_DEFAULT_WEIGHTS
        self.consonancia_weights = consonancia_weights if consonancia_weights is not None else default_weights

    def aplicar(self, vector_rugosidad: np.ndarray, acorde: 'Acorde', modelo: 'ModeloRugosidad') -> np.ndarray:
        """
        Aplica la ponderación de consonancia al vector de rugosidad.
        """
        pesos = np.ones(12)
        weights_to_use = getattr(modelo, 'consonancia_weights', self.consonancia_weights) # Permite que el modelo tenga sus propios pesos si es necesario

        for i in range(12):
            pesos[i] *= (1 + weights_to_use.get(i, 0))

        return safe_normalize(vector_rugosidad * pesos)


class PonderacionImportanciaPerceptual:
    """
    Pondera el vector de rugosidad basado en la importancia perceptual de los intervalos.
    """
    def __init__(self, bajo_weight: float = PONDERACION_IMPORTANCIA_BAJO_WEIGHT_DEFAULT,
                 final_weight: float = PONDERACION_IMPORTANCIA_FINAL_WEIGHT_DEFAULT,
                 decay_factor: float = PONDERACION_IMPORTANCIA_DECAY_FACTOR_DEFAULT):
        """
        Inicializa PonderacionImportanciaPerceptual con pesos y factor de decaimiento.

        Si no se proporcionan, se usan los valores por defecto definidos como constantes.
        """
        self.bajo_weight = bajo_weight
        self.final_weight = final_weight
        self.decay_factor = decay_factor

    def aplicar(self, vector_rugosidad: np.ndarray, acorde: 'Acorde', modelo: 'ModeloRugosidad') -> np.ndarray:
        """
        Aplica la ponderación de importancia perceptual al vector de rugosidad.
        """
        num_intervalos = len(acorde.intervals)
        if num_intervalos == 0:
            return vector_rugosidad

        pesos = np.ones(12)
        intervalos_aplanados = np.array([(j - i) % 12 for i, j in combinations(np.cumsum([0] + acorde.intervals), 2)])

        if len(intervalos_aplanados) > 0:
            pesos[intervalos_aplanados[0]] *= self.bajo_weight

        if len(intervalos_aplanados) > 1:
            pesos[intervalos_aplanados[-1]] *= self.final_weight

        for i in range(1, len(intervalos_aplanados) - 1):
            peso_decay = 1 + 1.5 * np.exp(-self.decay_factor * i)
            pesos[intervalos_aplanados[i]] *= peso_decay

        return safe_normalize(vector_rugosidad * pesos)


class PonderacionCombinada:
    """
    Combina dos tipos de ponderaciones para aplicar en secuencia.
    """
    def __init__(self, consonancia_ponderacion: PonderacionConsonancia, importancia_ponderacion: PonderacionImportanciaPerceptual):
        """
        Inicializa PonderacionCombinada con instancias de las ponderaciones a combinar.
        """
        self.consonancia = consonancia_ponderacion
        self.importancia = importancia_ponderacion

    def aplicar(self, vector_rugosidad: np.ndarray, acorde: 'Acorde', modelo: 'ModeloRugosidad') -> np.ndarray:
        """
        Aplica la ponderación combinada al vector de rugosidad.
        """
        vector_consonancia = self.consonancia.aplicar(vector_rugosidad, acorde, modelo)
        vector_combinado = self.importancia.aplicar(vector_consonancia, acorde, modelo)
        return vector_combinado


# --- Representación FR y Sp-Sf ---
def derive_fr(acorde: 'Acorde', max_denominator: int = DERIVE_FR_MAX_DENOMINATOR_DEFAULT, base_freq: float = DERIVE_FR_BASE_FREQ_DEFAULT) -> Tuple[int, ...]:
    """
    Deriva la Relación de Frecuencias (FR) de un objeto Acorde.
    """
    if acorde.frequencies is not None:
        freqs = np.array(acorde.frequencies, dtype=float)
        if freqs.ndim > 1:
            freqs = freqs[:, 0]
    else:
        semitonos = np.cumsum([0] + acorde.intervals)
        freqs = base_freq * 2 ** (np.array(semitonos) / 12.0)

    fundamental = freqs.min()
    ratios = [f / fundamental for f in freqs]
    approx_fracs = [Fraction(r).limit_denominator(max_denominator) for r in ratios]
    denominators = [frac.denominator for frac in approx_fracs]
    common_denom = lcm_list(denominators)
    fr_numbers = [int(frac * common_denom) for frac in approx_fracs]
    common_gcd = reduce(math.gcd, fr_numbers)
    fr_simplified = [num // common_gcd for num in fr_numbers]

    return tuple(fr_simplified)


def compute_sp_sf_representation(frac_ratios: Tuple[int, ...]) -> Dict[str, float]:
    """
    Calcula la representación Sp–Sf (junto con Sd y Sa) a partir de la relación FR.
    """
    ratios = sorted(frac_ratios)
    Sf = 1.0 / ratios[-1]
    M = lcm_list(ratios)
    Sp = ratios[0] / M if M != 0 else 0
    Sd = (Sf + Sp) / math.sqrt(2)
    Sa = (Sp - Sf) / math.sqrt(2)
    return {'Sf': Sf, 'Sp': Sp, 'Sd': Sd, 'Sa': Sa}


def representacion_sp_sf(acordes: List['Acorde']) -> np.ndarray:
    """
    Calcula la matriz de representación Sp–Sf para una lista de acordes.
    """
    reps = [compute_acorde_representation(ac) for ac in acordes]
    return np.array([[r['Sf'], r['Sp']] for r in reps])


def compute_acorde_representation(acorde: 'Acorde') -> Dict[str, float]:
    """
    Calcula la representación Sp–Sf de un acorde.
    """
    fr = derive_fr(acorde)
    return compute_sp_sf_representation(fr)

    ######### probablemtne esto es para otro modulo###
    # %% [code]
