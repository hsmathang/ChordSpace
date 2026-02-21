# Arquitectura de la capa de acceso a datos

La herramienta de experimentos comparte ahora una capa de acceso unificada
implementada en `services/data_gateway.py`. Esta capa encapsula la
resolución de consultas SQL, la deduplicación de poblaciones y el acceso
al catálogo global de plantillas de acordes. Permite que los consumidores
(CLI, GUI o scripts) cambien la fuente de datos sin modificar la lógica
de los experimentos.

## Contrato principal

La interfaz `ExperimentDataGateway` define los siguientes puntos de
extensión:

- `resolve_sql(query_or_alias: str) -> str`: resuelve un identificador de
  consulta (`QUERY_*`, `config:nombre`, `custom:nombre`, `sql:...` o SQL en
  crudo) utilizando `tools.query_registry`.
- `fetch_population(sources: Sequence[str], *, dedupe: bool = True) ->
  PopulationResult`: ejecuta y combina poblaciones provenientes de una o
  varias fuentes (consultas, archivos, etc.).
- `ingest_population(frame: pd.DataFrame, *, dedupe: bool = True,
  source: Optional[str] = None, metadata: Optional[Mapping[str, Any]] =
  None) -> PopulationResult`: aplica las mismas reglas de normalización a
  un `DataFrame` ya materializado (por ejemplo, datos cargados desde la
  CLI en JSON).
- `get_templates() -> Sequence[Mapping[str, Any]]`: expone el catálogo
  `CHORD_TEMPLATES_METADATA` sin que los consumidores dependan de
  `config.py`.
- `available_queries() -> Mapping[str, Dict[str, str]]`: devuelve el
  registro completo de consultas (`nombre -> {sql, source}`).

El resultado `PopulationResult` entrega la población deduplicada junto con
estadísticas útiles (`raw_count`, `final_count`, `removed`, conteos por
fuente, clave de dedupe aplicada, etc.) que pueden mostrarse en la UI o
logearse en scripts.

## Implementaciones registradas

`services/data_gateway.py` expone dos adaptadores listos para usar:

- `DatabaseQueryGateway`: ejecuta consultas mediante `QueryExecutor` y
  aplica deduplicación en memoria. Resuelve referencias usando el registro
  global para aceptar nombres (`QUERY_*`) o SQL en crudo.
- `CSVPopulationGateway`: carga poblaciones exportadas desde archivos
  CSV, JSON (newline-delimited) o Parquet. Hereda la deduplicación y las
  estadísticas comunes de la clase base.

Los adaptadores se registran mediante `register_data_gateway(name,
factory)` y se instancian con `create_data_gateway(name, **kwargs)`. El
registro por defecto incluye las llaves `database` y `csv`, lo que
permite cambiar de backend sin modificar los consumidores.

## Integración en los experimentos

`tools/compare_proposals.py` crea el gateway solicitado por CLI (`--data-
 gateway`) e inyecta opciones adicionales (`--gateway-option
 clave=valor`). El gateway produce el `PopulationResult` y entrega el
catálogo de plantillas para etiquetar acordes. Toda la interacción con
`QueryExecutor`, `config.config_db`, `CHORD_TEMPLATES_METADATA` y la
función `dedupe_population` quedó centralizada en el gateway.

## Extender la capa

Para añadir una nueva fuente de datos:

1. Implementa una clase que derive de `BaseExperimentDataGateway` o
   satisfaga el protocolo `ExperimentDataGateway`.
2. Registra la fábrica: `register_data_gateway("mi_gateway",
   MiGatewayClass)`.
3. (Opcional) expone parámetros configurables aceptando `**kwargs` en el
   constructor, que podrán pasarse desde CLI/GUI vía `--gateway-option`.

De esta forma, la lógica de los experimentos permanece desacoplada de la
fuente concreta (base de datos, archivos, APIs externas, etc.) y los
nuevos adaptadores comparten la deduplicación y el acceso a plantillas.
