import os
import datetime

files = [
    r"d:\Documents\GitHub\ChordSpace\docs\thesis\referencias_metodologia.bib",
    r"d:\Documents\GitHub\ChordSpace\docs\thesis\referencias_metodologia_vf.bib",
    r"d:\Documents\GitHub\ChordSpace\metodologia_temporal\Tesis MSc UNALvf\Tesis MSc UNALvf.bib"
]

with open("d:\\Documents\\GitHub\\ChordSpace\\bib_dates.txt", "w") as out:
    for f in files:
        if os.path.exists(f):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            out.write(f"{f} : {mtime}\n")
        else:
            out.write(f"{f} : NOT FOUND\n")
