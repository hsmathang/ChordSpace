"""
common.py — Shared functions for chord space experiments.
Self-contained but uses EXACT formulas from the ChordSpace repository.

Source mapping:
  - Sethares roughness: pre_process.py:ModeloSetharesVec (H=6, decay=0.88)
  - Voice Leading:      tools/proposals_pipeline/metrics.py:_voice_leading_distance
  - Quintas Profile:    tools/proposals_pipeline/metrics.py:_quintas_profile
  - Composite d_w:      tools/proposals_pipeline/metrics.py:_voiceleading_quintas_distance
  - EB dissimilarity:   DISCUSION_RIGUROSA §7.1 (NEW — operates on continuous MIDI R)
"""
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

# =====================================================================
# Sethares roughness model (from pre_process.py:ModeloSetharesVec)
# Parameters: H=6 harmonics, decay=0.88, Sethares (1993) formula
# =====================================================================
_D, _S1, _S2 = 0.24, 0.0207, 18.96
_C1, _C2, _A1, _A2 = 5.0, -5.0, -3.5, -5.75
N_H, DECAY, A4 = 6, 0.88, 440.0
EPS = 1e-12

def midi_to_freq(n):
    """MIDI (continuous float) → frequency (Hz). Standard: A4=69=440Hz."""
    return A4 * 2**((n - 69) / 12.0)

def _pair_roughness(f1, f2):
    """Sethares pairwise roughness between two fundamentals.
    Source: pre_process.py:ModeloSetharesVec._calcular_disonancia_pairwise"""
    K = np.arange(1, N_H + 1, dtype=float)
    A = DECAY ** (K - 1)
    P1, P2 = f1 * K, f2 * K
    Fm = np.minimum(P1[:, None], P2[None, :])
    Df = np.abs(P2[None, :] - P1[:, None])
    S = _D / (_S1 * Fm + _S2)
    Ap = A[:, None] * A[None, :]
    return float(np.sum(Ap * (_C1 * np.exp(_A1 * S * Df) + _C2 * np.exp(_A2 * S * Df))))

def _bin(iv):
    """Map interval (semitones) to 12-bin histogram index.
    Source: pre_process.py:interval_to_ui_bin"""
    return (iv - 1) % 12

def phi_raw(midi_notes):
    """Compute 12-bin roughness histogram Phi_raw in R^12.
    Aggregates over C(n,2) dyadic pairs.
    Source: pre_process.py:ModeloSetharesVec.calcular
    Args: midi_notes — array of MIDI values (floats).
    Returns: (histogram[12], total_roughness)"""
    freqs = sorted(midi_to_freq(n) for n in midi_notes)
    n = len(freqs)
    if n < 2:
        return np.zeros(12), 0.0
    st = [0.0] + [12.0 * np.log2(freqs[i] / freqs[0]) for i in range(1, n)]
    h = np.zeros(12)
    t = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            iv = int(round(st[j] - st[i])) % 12
            r = _pair_roughness(freqs[i], freqs[j])
            h[_bin(iv)] += r
            t += r
    return h, t

def phi_simplex(midi_notes):
    """Phi normalized to probability simplex: p = Phi / ||Phi||_1."""
    h, _ = phi_raw(midi_notes)
    s = np.sum(h)
    return h / (s + EPS) if s > EPS else np.full(12, 1.0 / 12.0)

# =====================================================================
# Distance metrics on roughness vectors
# =====================================================================
def d_jsd(a, b):
    """sqrt(JSD) between Phi_simplex of two chords. Base 2."""
    return float(jensenshannon(phi_simplex(a), phi_simplex(b), base=2.0))

def d_cosine(a, b):
    """Cosine distance on Phi_raw."""
    ha, _ = phi_raw(a)
    hb, _ = phi_raw(b)
    dot = np.dot(ha, hb)
    na, nb = np.linalg.norm(ha), np.linalg.norm(hb)
    return float(1.0 - dot / (na * nb + EPS))

def d_euclidean(a, b):
    """Euclidean distance on Phi_raw."""
    ha, _ = phi_raw(a)
    hb, _ = phi_raw(b)
    return float(np.linalg.norm(ha - hb))

# =====================================================================
# Circle of Fifths profile
# Source: tools/proposals_pipeline/metrics.py:_quintas_profile
# =====================================================================
COF = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5], dtype=int)
COF_IDX = {int(pc): i for i, pc in enumerate(COF)}

def quintas_profile(midi_notes):
    """Smoothed profile on circle of fifths. Kernel (1/4, 1/2, 1/4)."""
    vec = np.zeros(12)
    for n in midi_notes:
        pc = int(round(float(n))) % 12
        idx = COF_IDX.get(pc)
        if idx is not None:
            vec[idx] += 1.0
    sm = 0.5 * vec + 0.25 * np.roll(vec, 1) + 0.25 * np.roll(vec, -1)
    t = float(np.sum(sm))
    return sm / t if t > EPS else np.full(12, 1.0 / 12.0)

def d_q5(a, b):
    """Hellinger distance on quintas profiles."""
    qa, qb = quintas_profile(a), quintas_profile(b)
    return float(np.linalg.norm(np.sqrt(qa) - np.sqrt(qb)) / np.sqrt(2.0))

# =====================================================================
# Voice Leading — Hungarian matching
# Source: tools/proposals_pipeline/metrics.py:_voice_leading_distance
# =====================================================================
def step_continuous(x, y):
    """Voice step cost on CONTINUOUS MIDI values (R, not Z_12).
    Combines circular pitch-class distance + register penalty.
    Source: tools/proposals_pipeline/metrics.py:_voice_step_cost

    Operates on floats to respect the Riemannian geometry of
    Callender-Quinn-Tymoczko (2008) and Himpel (2022).
    The continuous space is log-frequency where 1 unit = 1 semitone.
    """
    # Circular distance in pitch-class space (continuous mod 12)
    pc_x = float(x) % 12.0
    pc_y = float(y) % 12.0
    diff = abs(pc_x - pc_y)
    semitone_fold = min(diff, 12.0 - diff)
    # Register penalty (absolute MIDI distance, capped at 24)
    register_penalty = min(abs(float(x) - float(y)), 24.0) / 24.0
    return float(semitone_fold + 0.35 * register_penalty)

def step_circular_pure(x, y):
    """Pure circular distance on continuous pitch classes.
    This is a TRUE metric on R/12Z.
    step(x,y) = min(|x mod 12 - y mod 12|, 12 - |x mod 12 - y mod 12|)"""
    pc_x = float(x) % 12.0
    pc_y = float(y) % 12.0
    diff = abs(pc_x - pc_y)
    return min(diff, 12.0 - diff)

def d_vl(a, b, step_fn=step_continuous, gap=6.5):
    """Voice leading distance via Hungarian algorithm.
    Source: tools/proposals_pipeline/metrics.py:_voice_leading_distance"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    M = max(a.size, b.size)
    if M == 0:
        return 0.0
    C = np.full((M, M), gap, dtype=float)
    for i in range(a.size):
        for j in range(b.size):
            C[i, j] = step_fn(float(a[i]), float(b[j]))
    ri, ci = linear_sum_assignment(C)
    return float(np.clip(np.sum(C[ri, ci]) / (M * gap), 0.0, 1.0))

# =====================================================================
# Composite d_w (voiceleading_quintas)
# Source: tools/proposals_pipeline/metrics.py:_voiceleading_quintas_distance
# Weights: w_VL=0.55, w_Q5=0.25, w_JS=0.20
# =====================================================================
W_VL, W_Q5, W_JS = 0.55, 0.25, 0.20

def d_w(a, b):
    """Composite metric: 0.55*VL + 0.25*Q5 + 0.20*JSD."""
    return W_VL * d_vl(a, b) + W_Q5 * d_q5(a, b) + W_JS * d_jsd(a, b)

# =====================================================================
# Expansion Biyectiva (EB) — NEW
# Source: DISCUSION_RIGUROSA_UNIFICADA §7.1
#
# CRITICAL: operates on CONTINUOUS MIDI values (R), NOT integer PCs.
# This respects the log-frequency space of Callender-Quinn-Tymoczko
# and Himpel, enabling smooth convergence at Whitney boundaries.
# =====================================================================
def _distinct_notes(midi_notes, tol=1e-6):
    """Get distinct notes (support) from continuous MIDI values.
    Two notes are 'the same' if they differ by less than tol semitones."""
    notes = sorted(float(n) for n in midi_notes)
    distinct = [notes[0]]
    for n in notes[1:]:
        if abs(n - distinct[-1]) > tol:
            distinct.append(n)
    return distinct

def _expansions_continuous(midi_notes, K, tol=1e-6):
    """Generate expansions of chord to size K by duplicating existing notes.
    Works with CONTINUOUS MIDI values (floats).
    Returns list of sorted tuples of length K."""
    distinct = _distinct_notes(midi_notes, tol)
    m = len(distinct)
    if m == 0:
        return []
    if m >= K:
        return [tuple(distinct[:K])]
    extras_needed = K - m
    expansions = set()
    for combo in combinations_with_replacement(range(m), extras_needed):
        exp = tuple(sorted(distinct + [distinct[i] for i in combo]))
        expansions.add(exp)
    return list(expansions)

def d_eb(a, b, step_fn=step_circular_pure, tol=1e-6):
    """Expansion Biyectiva dissimilarity on CONTINUOUS MIDI space (R).

    d_EB(A,B) = min over expansions of A,B to size K=max(|supp(A)|,|supp(B)|)
                of the optimal bijective matching cost / K.

    Guarantees:
      - M1 (non-negativity): YES
      - M3 (symmetry): YES
      - E0 (duplication cost 0): YES — d_EB(A, A∪{a}) = 0
      - E0 (limit continuity): YES — lim_{t→0} d_EB(A, B(t)) = 0
      - M4 (triangle inequality): CONJECTURAL (empirical audit needed)
    """
    da = _distinct_notes(a, tol)
    db = _distinct_notes(b, tol)
    K = max(len(da), len(db))
    if K == 0:
        return 0.0

    exp_a = _expansions_continuous(a, K, tol)
    exp_b = _expansions_continuous(b, K, tol)

    best = float('inf')
    for ea in exp_a:
        for eb in exp_b:
            C = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    C[i, j] = step_fn(ea[i], eb[j])
            ri, ci = linear_sum_assignment(C)
            cost = float(np.sum(C[ri, ci])) / K
            if cost < best:
                best = cost
    return best

# =====================================================================
# Chord corpus generation
# =====================================================================
CHORD_TYPES = {
    'maj':   [0, 4, 7],
    'min':   [0, 3, 7],
    'dim':   [0, 3, 6],
    'aug':   [0, 4, 8],
    'sus4':  [0, 5, 7],
    'sus2':  [0, 2, 7],
    'dom7':  [0, 4, 7, 10],
    'maj7':  [0, 4, 7, 11],
    'min7':  [0, 3, 7, 10],
    'dim7':  [0, 3, 6, 9],
    'hdim7': [0, 3, 6, 10],
    'minmaj7': [0, 3, 7, 11],
    'aug7':  [0, 4, 8, 10],
}

NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def generate_corpus(types=None, base_octave=60):
    """Generate chord corpus as MIDI note arrays (floats).
    Compatible with services/combinatorial_generator.py output format."""
    if types is None:
        types = list(CHORD_TYPES.keys())
    corpus = []
    for ct in types:
        intervals = CHORD_TYPES[ct]
        for root in range(12):
            midi = [float(base_octave + root + iv) for iv in intervals]
            name = f"{NOTE_NAMES[root]}{ct}"
            corpus.append({
                'name': name, 'midi': midi, 'type': ct,
                'root': root, 'root_name': NOTE_NAMES[root],
                'pc': sorted(set(int(n) % 12 for n in midi)),
                'card': len(midi),
            })
    return corpus
