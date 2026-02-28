import pandas as pd
import numpy as np

fpath = r"d:\Documents\GitHub\ChordSpace\metodologia_temporal\pnas.1713206115.sd01.xlsx"

out_rows = []

xl = pd.ExcelFile(fpath, engine="openpyxl")
for sheet in xl.sheet_names:
    k = 0
    chord_cols = []
    mean_col = 0
    sh_min = sheet.lower()
    if sh_min == "dyads":
        k = 2
        chord_cols = [2, 3]
        mean_col = 15
    elif sh_min == "triads":
        k = 3
        chord_cols = [2, 3, 4]
        mean_col = 17
    elif sh_min == "tetrads":
        k = 4
        chord_cols = [2, 3, 4, 5]
        mean_col = 19
    else:
        continue
        
    df = pd.read_excel(fpath, sheet_name=sheet, header=None)
    
    header_idx = -1
    for i in range(len(df)):
        if "CHORD #" in str(df.iloc[i, 0]):
            header_idx = i
            break
            
    if header_idx == -1:
        continue
        
    # data starts at header_idx + 3
    start_row = header_idx + 3
    
    for i in range(start_row, len(df)):
        # Check if first tone is nan
        val1 = df.iloc[i, chord_cols[0]]
        if pd.isna(val1):
            break
            
        tones = [df.iloc[i, c] for c in chord_cols]
        rating = df.iloc[i, mean_col]
        
        # tones are 0, 4, 7 for example, or sometimes string? 
        # let's parse as ints
        try:
            tones_int = [int(float(t)) for t in tones if not pd.isna(t)]
            tones_str = "_".join(map(str, sorted(tones_int)))
            out_rows.append({
                "k": k,
                "tones": tones_str,
                "rating": float(rating)
            })
        except Exception as e:
            print(f"Row {i} skipped: {e}")

out_df = pd.DataFrame(out_rows)
out_df.to_csv("bowling_data.csv", index=False)
print(f"Extracted {len(out_df)} records to bowling_data.csv")
print(out_df.head())
print(out_df.tail())
