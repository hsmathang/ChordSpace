import os
import glob
import datetime

files = glob.glob('d:/Documents/GitHub/ChordSpace/**/*.bib', recursive=True)
files.sort(key=os.path.getmtime, reverse=True)

for f in files:
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    size = os.path.getsize(f)
    print(f"{f} - Modified: {mtime} - Size: {size/1024:.2f} KB")
