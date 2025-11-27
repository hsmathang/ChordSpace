
import sys
import pandas as pd
from services.combinatorial_generator import generate_combinatorial_chords

def main():
    print("Generating reference population...")
    # C Major scale pitch classes
    alphabet = [0, 2, 4, 5, 7, 9, 11]
    octave_min = 3
    octave_max = 3
    cardinalities = [2] # Dyads

    df = generate_combinatorial_chords(
        alphabet=alphabet,
        octave_min=octave_min,
        octave_max=octave_max,
        cardinalities=cardinalities,
        structural_mode=False
    )

    output_path = "data/reference_population.jsonl"
    df.to_json(output_path, orient="records", lines=True)
    print(f"Generated {len(df)} records to {output_path}")

if __name__ == "__main__":
    main()
