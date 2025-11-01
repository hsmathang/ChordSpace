# GUI experiment launcher

`tools/gui_experiment_launcher.py` ofrece una interfaz sencilla para ejecutar consultas definidas en `config.py`, calcular metricas y exportar visualizaciones interactivas.

---

## Preparacion rapida
- Asegurate de tener la base `ChordCodex` poblada (consulta `README.md` y `docs/DB_SETUP.md`).
- Activa el entorno virtual del proyecto:
  - Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
  - macOS/Linux: `source .venv/bin/activate`
- Instala dependencias si aun no lo hiciste: `pip install -r requirements.txt`

Lanza la GUI:
```bash
python -m tools.gui_experiment_launcher
```

---

## Flujo de trabajo recomendado
1. Selecciona una consulta base de la lista (las mismas constantes que en `config.py`).
2. Ajusta filtros opcionales (por ejemplo, limite de filas) para evitar traer millones de registros en la primera ejecucion.
3. Escoge los graficos a generar (scatter, heatmap, shepard, etc.).
4. Ejecuta y espera a que el progreso marque 100 %. El tiempo depende del tamano del subconjunto y de si la base esta en Docker o nativa.
5. Revisa los archivos generados en `outputs/gui_runs/<timestamp>/`.

> Sugerencia: inicia con consultas como `QUERY_CHORDS_WITH_NAME` o `QUERY_CHORDS_WITH_NAME_AND_RANDOM_CHORDS_POBLATION` que regresan subconjuntos razonables.

---

## Salidas
Cada ejecucion crea un directorio con:
- `config.json`: parametros usados en la sesion.
- Archivos HTML interactivos (scatter, parallel coordinates, etc.).
- CSV o Parquet con los datos filtrados.

Puedes abrir los HTML directamente en el navegador o versionarlos como evidencia del experimento.

---

## Extender la GUI con comandos

`tools/gui_commands.py` introduce un protocolo `ExperimentCommand` y un contenedor `CommandRegistry` para enchufar nuevas acciones en la ventana sin modificar su código base. El lanzador expone el registro mediante `ExperimentLauncher.command_registry` y ofrece el helper `snapshot_state()` para capturar el estado actual (`LauncherState`).

Los comandos predefinidos son:

- `RunComparisonCommand`: delega en un servicio con método `run(state)` (por ejemplo, un comparador de propuestas).
- `GenerateNeighboursCommand`: espera un servicio con método `generate(state)` que devuelva vecinos para la selección actual.
- `AugmentPopulationCommand`: llama a un servicio `augment(state)` encargado de fusionar nuevas filas en la población.

Todos los servicios deben devolver un `CommandResult`, que agrupa `datasets`, `metrics`, `visualizations` y campos adicionales (`extras`) para que la GUI encadene resultados o actualice pestañas.

### Registrar un comando personalizado

```python
from tools.gui_commands import CommandResult, LauncherState, RunComparisonCommand
from tools.gui_experiment_launcher import ExperimentLauncher


class SummaryService:
    def run(self, state: LauncherState) -> CommandResult:
        population = state.get_dataset("population")
        size = len(population) if population is not None else 0
        return CommandResult(metrics={"population_size": size})


launcher = ExperimentLauncher()
launcher.register_command("demo.summary", RunComparisonCommand(SummaryService()))

# Ejecutar el comando usando el estado actual de la GUI
result = launcher.dispatch_command("demo.summary")
print(result.metrics["population_size"])
```

Los módulos externos (audio, navegación 3D, etc.) pueden seguir el mismo patrón: definir un servicio que consuma `LauncherState`, retornar un `CommandResult` con los artefactos generados y registrarlo bajo un identificador propio. Esto permite ampliar la interfaz desde paquetes separados o notebooks sin tocar la ventana principal.

---

## Solucion de problemas
- **La ventana no abre**: verifica que estas usando el Python del entorno virtual (`where python` o `Get-Command python` debe apuntar a `.venv`).
- **Errores `Illegal instruction` en Apple Silicon**: reinstala `polars-lts-cpu==1.32.3` o usa un entorno Python ARM nativo (`arch -arm64 python3.11 -m venv .venv`).
- **Consultas demasiado grandes**: ajusta limites en la GUI o modifica la constante en `config.py` para incluir `LIMIT`.
- **Archivos HTML vacios**: confirma que la consulta devuelve filas con `python -m tools.run_sql --query TU_CONSTANTE --limit 5`.
- **Docker lento**: si usas el contenedor de Postgres, asegurate de que la maquina virtual tiene suficiente RAM y disco.

---

## Referencias
- `config.py`: catalogo completo de consultas y constantes usadas por la GUI.
- `tools/run_sql.py`: runner minimo para depurar consultas antes de llevarlas a la interfaz.
- `docs/DB_SETUP.md`: pasos detallados para montar la base y resolver problemas comunes.
