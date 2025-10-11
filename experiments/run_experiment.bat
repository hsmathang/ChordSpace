@echo off
rem ==============================================================
rem  Experimentos A/B/C y modo conjunto (joint) para MDS/UMAP
rem  - Color de figuras: rugosidad (sin cambios en visualization)
rem  - Este .bat acepta hasta 4 argumentos posicionales opcionales
rem    1) TYPE       = A | B | C            (por defecto B)
rem    2) QUERY      = const en config.py   (por defecto QUERY_CHORDS_WITH_NAME)
rem    3) REDUCTION  = MDS | UMAP           (por defecto MDS)
rem    4) OUTDIR     = carpeta salida       (por defecto outputs\experiment)
rem
rem  Uso básico (defaults):
rem    experiments\run_experiment.bat
rem
rem  Ejemplos:
rem    - Triadas CORE sin inversiones (A, MDS):
rem      experiments\run_experiment.bat A QUERY_TRIADS_CORE MDS outputs\triads_A_mds
rem
rem    - Triadas CORE con inversiones DB (B, UMAP):
rem      experiments\run_experiment.bat B QUERY_TRIADS_CORE UMAP outputs\triads_B_umap
rem
rem    - Inversiones sintéticas (C):
rem      experiments\run_experiment.bat C QUERY_TRIADS_CORE MDS outputs\triads_C_mds
rem
rem  Modo conjunto (joint): pasar --pops al final (usar "" para defaults):
rem    experiments\run_experiment.bat "" "" MDS outputs\joint_AB --pops A:QUERY_TRIADS_CORE --pops B:QUERY_TRIADS_CORE
rem    experiments\run_experiment.bat "" "" UMAP outputs\joint_ABC --pops A --pops B --pops C
rem    (si no se especifica QUERY en --pops, usa QUERY_CHORDS_WITH_NAME)
rem ==============================================================
rem  Consultas disponibles (config.py / notebook) y breve descripcion:
rem    - QUERY_CHORDS_3_NOTES
rem        60 triads (n=3) con columnas estandar.
rem    - QUERY_CHORDS_WITH_NAME
rem        Catalogo de 30 acordes (triadas/extendidos) con raiz 0 definido en CHORD_TEMPLATES.
rem    - QUERY_TRIADS_WITH_INVERSIONS
rem        Triadas mayor/menor/disminuida mas sus dos inversiones presentes en la DB.
rem    - QUERY_TRIADS_ROOT_ONLY_MOBIUS_MAZZOLA
rem        Triadas mayor/menor/disminuida restringidas a raices diatonicas {0,2,4,5,7,9,B}.
rem    - QUERY_TRIADS_CORE
rem        Catalogo canonico de triadas (Maj/Min/Dim/Aug/Sus2/Sus4) una por raiz.
rem    - QUERY_SEVENTHS_CORE
rem        Catalogo canonico de acordes de septima (7, m7, Maj7, m7b5, Dim7) uno por raiz.
rem    - QUERY_TRIADS_WITH_REPEATED_NOTES
rem        Triadas con raiz 0, notas repetidas y span <= 12 (deteccion de duplicados).
rem    - QUERY_EXTREME_CLUSTER_10_NOTES
rem        Clusters de 10 notas: incluye el acorde de intervalos 1 y variantes 1-2 semitonos.
rem    - QUERY_DYADS_REFERENCE
rem        Catalogo de diadas: unisono y una diada por cada semitono 1..12.
rem    - QUERY_DYADS_RANDOM_UNIQUE
rem        Una diada aleatoria por semitono 1..11 mas el unisono raiz 0.
rem    - QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES
rem        100 acordes aleatorios con n > 2.
rem    - QUERY_RANDOM_CHORD_MORE_THAN_2_NOTES_CLOSE_OCTAVE
rem        900 acordes aleatorios con raiz 0 y suma de intervalos <= 12.
rem    - QUERY_CHORDS_WITH_NAME_AND_RANDOM_CHORDS_POBLATION
rem        Union del catalogo con raiz 0 mas 60 acordes aleatorios externos.
rem    - QUERY_CHORDS_SPECIFIC_INTERVALS_AND_RANDOM_SAME_OCTAVE
rem        Igual que la anterior pero 400 acordes aleatorios con span <= 12.

rem ==============================================================

setlocal EnableDelayedExpansion

rem Defaults
set TYPE=B
set QUERY=QUERY_CHORDS_WITH_NAME
set REDUCTION=MDS
set OUTDIR=outputs\experiment
set EXTRA=

rem Named-arg parser: accepts TYPE=, QUERY=, REDUCTION=, OUTDIR= and passes everything else as EXTRA
:parse_args
if "%~1"=="" goto parsed_done
for /f "tokens=1* delims==" %%A in ("%~1") do (
  set KEY=%%~A
  set VAL=%%~B
)
if /I "!KEY!"=="TYPE" (
  set TYPE=!VAL!
) else if /I "!KEY!"=="QUERY" (
  set QUERY=!VAL!
) else if /I "!KEY!"=="REDUCTION" (
  set REDUCTION=!VAL!
) else if /I "!KEY!"=="OUTDIR" (
  set OUTDIR=!VAL!
) else (
  set EXTRA=!EXTRA! %~1
)
shift
goto parse_args
:parsed_done

echo Ejecutando: TYPE=%TYPE% QUERY=%QUERY% REDUCTION=%REDUCTION% OUT=%OUTDIR%
python -m tools.experiment_inversions --type %TYPE% --query %QUERY% --reduction %REDUCTION% --out %OUTDIR% %EXTRA%
if %errorlevel% neq 0 (
  echo Fallo al ejecutar el experimento
  exit /b %errorlevel%
)
echo OK. Artefactos en %OUTDIR%

endlocal
