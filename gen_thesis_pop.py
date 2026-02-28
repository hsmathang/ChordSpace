"""
gen_thesis_pop.py
-----------------
Genera EXACTAMENTE la población requerida para los experimentos de tesis:
  - 12 díadas básicas de la octava 3 (C3 con cada semitono de 1 a 12)
  - 21 tríadas diatónicas de C mayor (7 acordes × 3 formas: fundamental, 1ª inv, 2ª inv)
    usando las notas de la octava 3 y 4 según corresponda.

Crea: outputs/tesis_resultados_finales/population.json
Compatible con el pipeline via FilePopulationStore -> load_chords(df_override=...).
"""
import json, math, hashlib
from pathlib import Path

# ── Tablas de referencia ──────────────────────────────────────────────────────
A4_MIDI = 69
A4_FREQ = 440.0

def midi2freq(n):
    return round(A4_FREQ * (2.0 ** ((int(n) - A4_MIDI) / 12.0)), 6)

def stable_id(notes_abs):
    digest = hashlib.blake2b(
        json.dumps(list(notes_abs)).encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")

HEX12 = "0123456789AB"

def build_record(notes_abs, identity_name, family_id, inv_flag=False, inv_rotation=None):
    notes_abs = [int(n) for n in notes_abs]
    root = notes_abs[0]
    normalized  = [n - root for n in notes_abs]
    intervals   = [notes_abs[i+1] - notes_abs[i] for i in range(len(notes_abs)-1)]
    pcs_real    = [str(n % 12) for n in notes_abs]
    code        = "".join(HEX12[n % 12] for n in notes_abs)
    freqs       = [midi2freq(n) for n in notes_abs]
    mask        = 0
    for n in notes_abs: mask |= 1 << n

    return {
        # ── core pipeline columns ──────────────────────────────────────
        "id":               stable_id(notes_abs),
        "n":                len(notes_abs),
        "interval":         intervals,
        "notes":            pcs_real,
        "code":             code,
        "bass":             str(root % 12),
        "octave":           (root // 12) - 1,
        "frequencies":      freqs,
        "tag":              "thesis",
        "span_semitones":   notes_abs[-1] - notes_abs[0],
        "chroma":           None,
        "notes_abs_json":   json.dumps(notes_abs),
        "abs_mask_int":     None,
        "abs_mask_hex":     None,
        "abs_mask_midi":    format(mask, "x"),
        "source_id":        None,
        "rotation":         None,
        "family_id":        family_id,
        "family_size":      None,
        # ── annotation columns ─────────────────────────────────────────
        "__source__":               "GENERATED:THESIS",
        "__transposition__":        0,
        "__root_midi":              root,
        "__norm_interval":          normalized[1:],
        "__norm_notes":             [str(n) for n in normalized],
        "__norm_code":              code,
        "__norm_bass":              str(root % 12),
        "__struct_semitones":       normalized,
        "__structure_id":           "|".join(str(v) for v in normalized),
        # ── inversion columns (usados por load_chords) ─────────────────
        "__family_id":              family_id,
        "__inv_flag":               inv_flag,
        "__inv_source_id":          family_id,
        "__inv_rotation":           inv_rotation,
        # ── nombre legible (usado por el scatter plot en el hover) ──────
        "identity_name":            identity_name,
    }

# ── Escala de C mayor: MIDI absolutos octavam 3 + 4 + boundary ───────────────
# C3=48 D3=50 E3=52 F3=53 G3=55 A3=57 B3=59
# C4=60 D4=62 E4=64 F4=65 G4=67 A4=69 B4=71
# C5=72 (boundary)
SCALE = [48, 50, 52, 53, 55, 57, 59,   # oct 3: índices 0-6
         60, 62, 64, 65, 67, 69, 71,   # oct 4: índices 7-13
         72]                            # C5 boundary

CHORD_NAMES = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
# family_id para agrupar tríada + inversiones
FAMILY_IDS  = [100, 200, 300, 400, 500, 600, 700]

records = []

# ── 12 DÍADAS básicas: C3 (MIDI 48) + intervalo k ────────────────────────────
C3 = 48
DYAD_NAMES = ["m2","M2","m3","M3","P4","Tritono","P5","m6","M6","m7","M7","P8"]
for k, dname in enumerate(DYAD_NAMES, start=1):
    records.append(build_record(
        notes_abs=[C3, C3 + k],
        identity_name=dname,
        family_id=900 + k,
        inv_flag=False
    ))

# ── 21 TRÍADAS diatónicas: 7 grados × (Fund / 1ª Inv / 2ª Inv) ──────────────
for deg in range(7):  # grado diatónico (0=C, 1=D, …, 6=B)
    fid  = FAMILY_IDS[deg]
    name = CHORD_NAMES[deg]

    # índices dentro de SCALE
    i0, i2, i4, i7 = deg, deg+2, deg+4, deg+7  # 3ª y 5ª sobre el grado
    # Los índices 7..13 aterrizan en la 4ª octava

    # Posición fundamental
    fund  = [SCALE[i0], SCALE[i2], SCALE[i4]]
    # 1ª inversión: (3ª, 5ª, 8ª)
    inv1  = [SCALE[i2], SCALE[i4], SCALE[i7]]
    # 2ª inversión: (5ª, 8ª, 10ª)
    inv2  = [SCALE[i4], SCALE[i7], SCALE[i0+7]]

    records.append(build_record(fund, f"{name} (Fund)",  fid, inv_flag=False, inv_rotation=0))
    records.append(build_record(inv1, f"{name} (1ª Inv)", fid, inv_flag=True,  inv_rotation=1))
    records.append(build_record(inv2, f"{name} (2ª Inv)", fid, inv_flag=True,  inv_rotation=2))

# ── Guardar ───────────────────────────────────────────────────────────────────
out_dir = Path("outputs/tesis_resultados_finales")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "population.json"

with open(out_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"[OK] {len(records)} acordes escritos en {out_path}")
for r in records:
    print(f"  {r['identity_name']:20s}  notas={json.loads(r['notes_abs_json'])}  freqs=[{', '.join(f'{x:.1f}' for x in r['frequencies'])}]")
