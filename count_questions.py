import re

INPUT_FILE = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\metodologia_temporal\ESTRUCTURA_MATEMATICA_DETALLADA.md"

def count_questions():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    q_lines = []
    total_q_marks = 0
    
    for i, line in enumerate(lines):
        if "Nota de santimath" in line:
            continue
            
        if "?" in line:
            count = line.count("?")
            if count > 0:
                 total_q_marks += count
                 stripped = line.strip()
                 q_lines.append(f"{i+1}: ({count}) {stripped}")

    print(f"Total lines with '?': {len(q_lines)}")
    print(f"Total '?' characters: {total_q_marks}")
    print("\n--- DETAILED LIST ---")
    for l in q_lines:
        print(l)

if __name__ == "__main__":
    count_questions()
