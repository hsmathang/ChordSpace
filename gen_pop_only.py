import pandas as pd
import numpy as np
import json
from pathlib import Path

# Provide a simple local implementation of _process_chord_record to avoid heavy imports
import hashlib
def _midi_to_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))
def _stable_id(notes_abs: list) -> int:
    digest = hashlib.blake2b(json.dumps(notes_abs).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")
def _actual_mask(notes_abs: list) -> str:
    mask = 0
    for note in notes_abs:
        mask |= 1 << int(note)
    return format(mask, "x")

def _process_chord_record(notes_abs: list, tag: str) -> dict:
    root_midi = notes_abs[0]
    normalized = [note - root_midi for note in notes_abs]
    record = {}
    record['notes_abs_json'] = json.dumps(notes_abs)
    record['octave'] = (root_midi // 12) - 1
    record['frequencies'] = [_midi_to_freq(note) for note in notes_abs]
    record['span_semitones'] = notes_abs[-1] - notes_abs[0]
    record['notes'] = [str(n % 12) for n in notes_abs]
    record['bass'] = str(root_midi % 12)
    HEX12 = "0123456789AB"
    record['code'] = ''.join(HEX12[int(n) % 12] for n in notes_abs)
    record['interval'] = [int(notes_abs[i+1] - notes_abs[i]) for i in range(len(notes_abs)-1)]
    record['__source__'] = "GENERATED:CUSTOM_SCRIPT"
    record['__transposition__'] = 0
    record['__root_midi'] = root_midi
    record['abs_mask_midi'] = _actual_mask(notes_abs)
    record['id'] = _stable_id(notes_abs)
    record['__struct_semitones'] = [int(v) for v in normalized]
    record['__structure_id'] = "|".join(str(int(v)) for v in normalized)
    record['tag'] = tag
    record['n'] = len(notes_abs)
    return record

def build_diatonic_triads():
    scale = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
    names = ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]
    records = []
    
    for i in range(7):
        root_name = names[i]
        
        # Fundamental (1, 3, 5)
        root_pos = [scale[i], scale[i+2], scale[i+4]]
        rec_root = _process_chord_record(root_pos, tag="diatonic_triad")
        rec_root['identity_name'] = f"{root_name} (Fund)"
        rec_root['family_id'] = "Fundamental"
        records.append(rec_root)
        
        # 1st Inversion (3, 5, 8)
        inv1_pos = [scale[i+2], scale[i+4], scale[i+7]]
        rec_inv1 = _process_chord_record(inv1_pos, tag="diatonic_triad")
        rec_inv1['identity_name'] = f"{root_name} (1ra)"
        rec_inv1['family_id'] = "1ra Inversion"
        records.append(rec_inv1)
        
        # 2nd Inversion (5, 8, 10)
        inv2_pos = [scale[i+4], scale[i+7], scale[i+9]]
        rec_inv2 = _process_chord_record(inv2_pos, tag="diatonic_triad")
        rec_inv2['identity_name'] = f"{root_name} (2da)"
        rec_inv2['family_id'] = "2da Inversion"
        records.append(rec_inv2)
        
    return records

def build_basic_dyads():
    c3 = 48
    intervals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    names = ["m2", "M2", "m3", "M3", "P4", "trit", "P5", "m6", "M6", "m7", "M7", "P8"]
    
    records = []
    for interval, name in zip(intervals, names):
        pos = [c3, c3 + interval]
        rec = _process_chord_record(pos, tag="basic_dyad")
        rec['identity_name'] = name
        rec['family_id'] = "Diadas"
        records.append(rec)
        
    return records

def main():
    print("Generating isolated population...")
    triads = build_diatonic_triads()
    dyads = build_basic_dyads()
    all_records = dyads + triads
    
    df = pd.DataFrame.from_records(all_records)
    
    output_dir = Path("outputs/tesis_resultados_finales")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "population.json"
    df.to_json(output_path, orient='records', lines=True)
    print(f"Done! Created {len(all_records)} chords at {output_path}")

if __name__ == "__main__":
    main()
