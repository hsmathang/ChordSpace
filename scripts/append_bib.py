import os

bib_path = os.path.join(
    r'd:\Documents\GitHub\ChordSpace\docs',
    'Tesis_Maes\u00edtría_Matemáticas_Aplicadas_UNAL_2024 pacho',
    'ReferenciasRugosas.bib'
)

# Fallback: find the bib file directly
import glob
matches = glob.glob(r'd:\Documents\GitHub\ChordSpace\docs\**\ReferenciasRugosas.bib', recursive=True)
if not matches:
    print("ERROR: No se encontro el archivo .bib")
    exit(1)
bib_path = matches[0]
print(f"Usando: {bib_path}")

entries = r"""
% ─── ENTRADAS PARA ORACIONES 18-30 ────────────────────────────────────────────

@article{mcdermott2016indifference,
  author  = {McDermott, Josh H. and Schultz, Alan F. and Undurraga, Eduardo A. and Godoy, Ricardo A.},
  title   = {Indifference to dissonance in native {A}mazonians reveals cultural variation in music perception},
  journal = {Nature},
  year    = {2016},
  volume  = {535},
  number  = {7613},
  pages   = {547--550},
  doi     = {10.1038/nature18635},
}

@article{milne2023universal,
  author  = {Milne, Andrew J. and Smit, Eline A. and Sarvasy, Hannah S. and Dean, Roger T.},
  title   = {Evidence for a universal association of auditory roughness with musical stability},
  journal = {PLOS ONE},
  year    = {2023},
  volume  = {18},
  number  = {1},
  pages   = {e0278268},
  doi     = {10.1371/journal.pone.0278268},
}

@article{terhardt1974roughness,
  author  = {Terhardt, Ernst},
  title   = {On the perception of periodic sound fluctuations (roughness)},
  journal = {Acustica},
  year    = {1974},
  volume  = {30},
  number  = {4},
  pages   = {201--213},
}

@article{sethares1993local,
  author  = {Sethares, William A.},
  title   = {Local consonance and the relationship between timbre and scale},
  journal = {The Journal of the Acoustical Society of America},
  year    = {1993},
  volume  = {94},
  number  = {3},
  pages   = {1218--1228},
  doi     = {10.1121/1.408175},
}

@book{sethares2005tuning,
  author    = {Sethares, William A.},
  title     = {Tuning, Timbre, Spectrum, Scale},
  year      = {2005},
  edition   = {2},
  publisher = {Springer},
  address   = {London},
  doi       = {10.1007/b138834},
}

@article{harrison2020simultaneous,
  author  = {Harrison, Peter M. C. and Pearce, Marcus T.},
  title   = {Simultaneous consonance in music perception and composition},
  journal = {Psychological Review},
  year    = {2020},
  volume  = {127},
  number  = {2},
  pages   = {216--244},
  doi     = {10.1037/rev0000169},
}

@article{bernardes2016multilevel,
  author  = {Bernardes, Gilberto and Cocharro, Diogo and Caetano, Marcelo and Guedes, Carlos and Davies, Matthew E. P.},
  title   = {A multi-level tonal interval space for modelling pitch relatedness and musical consonance},
  journal = {Journal of New Music Research},
  year    = {2016},
  volume  = {45},
  number  = {4},
  pages   = {281--294},
  doi     = {10.1080/09298215.2016.1182192},
}

@article{parncutt2011consonance,
  author  = {Parncutt, Richard and Hair, Graham},
  title   = {Consonance and dissonance in music theory and psychology: {D}isentangling dissonant dichotomies},
  journal = {Journal of Interdisciplinary Music Studies},
  year    = {2011},
  volume  = {5},
  number  = {2},
  pages   = {119--166},
  doi     = {10.4407/jims.2011.11.007},
}
"""

with open(bib_path, 'a', encoding='utf-8') as f:
    f.write(entries)

print("OK: 8 entradas agregadas exitosamente al .bib")
