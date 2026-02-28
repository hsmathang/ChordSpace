import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pre_process import Acorde, ModeloSetharesVec
from config import SETHARES_BASE_FREQ, SETHARES_DECAY

print("Loading bowling_data.csv...")
df = pd.read_csv("bowling_data.csv")

# Initialize Sethares model
configuracion = {
    'base_freq': SETHARES_BASE_FREQ,
    'n_armonicos': 6,
    'decaimiento': SETHARES_DECAY,
}
modelo = ModeloSetharesVec(configuracion)

results = []

print(f"Running ModeloSetharesVec on {len(df)} chords...")
# Process each chord
for idx, row in df.iterrows():
    k = row['k']
    tones_str = row['tones']
    rating = row['rating']
    
    # Parse semitones: these are the actual pitch class gaps over the root (0)
    # E.g., "0_4_7". 
    semitones = [int(x) for x in tones_str.split("_")]
    
    # The 'Acorde' expects INTERVALS between consecutive notes
    intervals = [semitones[i] - semitones[i-1] for i in range(1, len(semitones))]
    
    # Create the chord object
    acorde = Acorde(name=f"Chord_{tones_str}", intervals=intervals)
    
    # Calculate roughness Metrics
    vector_12d, total_scalar = modelo.calcular(acorde)
    
    # Create result row
    res = {
        'k': k,
        'tones': tones_str,
        'rating': rating,
        'scalar_roughness': total_scalar
    }
    
    # Add 12D vector components
    for j in range(12):
        res[f'v{j}'] = vector_12d[j]
        
    results.append(res)

out_df = pd.DataFrame(results)
out_df.to_csv("bowling_results.csv", index=False)
print("Saved 298 results to bowling_results.csv.")
print(out_df.head())
