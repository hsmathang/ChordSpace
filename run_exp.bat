@echo off
.venv\Scripts\python.exe experiments\topological_variety\run_topology_experiment.py > out3.txt 2> err3.txt
exit /b %ERRORLEVEL%
