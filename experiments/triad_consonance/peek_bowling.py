import pandas as pd
import numpy as np
import os

fpath = r"d:\Documents\GitHub\ChordSpace\metodologia_temporal\pnas.1713206115.sd01.xlsx"

out_data = []

xl = pd.ExcelFile(fpath, engine="openpyxl")
for sheet in xl.sheet_names:
    print(f"\nProcessing {sheet}...")
    # Read the sheet, skipping the first 2 rows (0 and 1) because the actual headers are mixed
    # Let's read without header and figure out the columns manually.
    df = pd.read_excel(fpath, sheet_name=sheet, header=None)
    
    # Let's find the row that has "CHORD #"
    header_idx = -1
    for i in range(len(df)):
        if "CHORD #" in str(df.iloc[i, 0]):
            header_idx = i
            break
            
    if header_idx == -1:
        print(f"Header not found in {sheet}")
        continue
        
    # Data starts a couple rows after header row
    # In triads, CHORD TONES spans multiple columns.
    # SINGAPORE, VIENNA, COMBINED each have "Mean" and "SE" or similar.
    
    # Just print the header row and a few data rows to see EXACTLY the column indices
    print("Header Row:")
    for c in range(len(df.columns)):
        print(f"Col {c}: {df.iloc[header_idx, c]} | {df.iloc[header_idx+1, c]} | {df.iloc[header_idx+2, c]}")
        
    print("\nFirst data row:")
    for c in range(len(df.columns)):
        print(f"Col {c}: {df.iloc[header_idx+2, c]}")
    
