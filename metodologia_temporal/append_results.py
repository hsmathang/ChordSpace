import json
import os
import re
import glob

RESULTS_FILE = r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\metodologia_temporal\respuestas_completadas.json"

def main():
    # Initialize JSON if not exists
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
            
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        try:
            current_data = json.load(f)
        except json.JSONDecodeError:
            current_data = []

    # Find temp files
    temp_files = glob.glob(r"c:\Users\SANTIAGO\Documents\GitHub\ChordSpace\metodologia_temporal\result_Q*.txt")
    
    cnt = 0
    for tf in temp_files:
        basename = os.path.basename(tf)
        q_id = basename.replace("result_", "").replace(".txt", "")
        
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check if already processed
        if any(item['id'] == q_id for item in current_data):
            # Update existing? Or skip. Let's update.
            for item in current_data:
                if item['id'] == q_id:
                    item['respuesta_notebooklm'] = content
                    # Extract bibtex if present (simple regex heuristic)
                    citations = re.findall(r'@\w+\{.*\}', content, re.DOTALL)
                    item['citas_bibtex_detectadas'] = citations
        else:
            # Add new
            new_item = {
                "id": q_id,
                "respuesta_notebooklm": content,
                "citas_bibtex_detectadas": re.findall(r'@\w+\{.*\}', content, re.DOTALL)
            }
            current_data.append(new_item)
            
        cnt += 1
        
    # Write back
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(current_data, f, indent=2, ensure_ascii=False)
        
    # Delete temp files
    for tf in temp_files:
        try:
            os.remove(tf)
        except:
            pass
            
    print(f"Processed {cnt} result files. Total items: {len(current_data)}")

if __name__ == "__main__":
    main()
