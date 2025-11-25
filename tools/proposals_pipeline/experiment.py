"""Orquestación ligera de escenarios para propuestas de normalización."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple


def build_scenarios(
    proposals: Iterable[str],
    metrics: Iterable[str],
    preprocessors: Mapping[str, Tuple[str, object, Mapping[str, float]]],
    *,
    include_identity: bool = True,
) -> List[Dict[str, object]]:
    """Construye la lista de escenarios (propuesta x métrica).

    Parameters
    ----------
    proposals:
        Identificadores de propuesta solicitados (p. ej., simplex, perclass_alpha1).
    metrics:
        Identificadores de métricas (cosine, js, hellinger, ...).
    preprocessors:
        Mapeo preproc_id -> (descripcion, funcion, kwargs) usado para resolver propuestas.
    include_identity:
        Si es True, garantiza que exista un escenario baseline identity por métrica.
    """
    scenarios: List[Dict[str, object]] = []
    metrics = list(metrics)
    for proposal in proposals:
        proposal = proposal.strip().lower()
        if proposal in preprocessors:
            preproc_id = proposal
            preproc_func = preprocessors[proposal][1]
            kwargs = preprocessors[proposal][2]
            description = preprocessors[proposal][0]
        elif proposal in {"simplex", "simplex_cosine"}:
            preproc_id = "simplex"
            preproc_func = preprocessors["simplex"][1]
            kwargs = preprocessors["simplex"][2]
            description = preprocessors["simplex"][0]
        elif proposal in {"simplexsqrt", "simplex_sqrt"}:
            preproc_id = "simplex_sqrt"
            preproc_func = preprocessors["simplex_sqrt"][1]
            kwargs = preprocessors["simplex_sqrt"][2]
            description = preprocessors["simplex_sqrt"][0]
        elif proposal in {"simplexsmooth", "simplex_smooth"}:
            preproc_id = "simplex_smooth"
            preproc_func = preprocessors["simplex_smooth"][1]
            kwargs = preprocessors["simplex_smooth"][2]
            description = preprocessors["simplex_smooth"][0]
        elif proposal == "perclass_alpha1":
            preproc_id = "perclass_alpha1"
            preproc_func = preprocessors["perclass_alpha1"][1]
            kwargs = preprocessors["perclass_alpha1"][2]
            description = preprocessors["perclass_alpha1"][0]
        elif proposal in {"perclass_alpha0_5", "perclass_alpha05"}:
            preproc_id = "perclass_alpha0_5"
            preproc_func = preprocessors["perclass_alpha0_5"][1]
            kwargs = preprocessors["perclass_alpha0_5"][2]
            description = preprocessors["perclass_alpha0_5"][0]
        elif proposal in {"perclass_alpha0_75", "perclass_alpha075", "perclass_alpha75"}:
            preproc_id = "perclass_alpha0_75"
            preproc_func = preprocessors["perclass_alpha0_75"][1]
            kwargs = preprocessors["perclass_alpha0_75"][2]
            description = preprocessors["perclass_alpha0_75"][0]
        elif proposal in {"perclass_alpha0_25", "perclass_alpha025", "perclass_alpha25"}:
            preproc_id = "perclass_alpha0_25"
            preproc_func = preprocessors["perclass_alpha0_25"][1]
            kwargs = preprocessors["perclass_alpha0_25"][2]
            description = preprocessors["perclass_alpha0_25"][0]
        elif proposal == "global_pairs":
            preproc_id = "global_pairs"
            preproc_func = preprocessors["global_pairs"][1]
            kwargs = preprocessors["global_pairs"][2]
            description = preprocessors["global_pairs"][0]
        elif proposal in {"divide_mminus1", "divide_m-1"}:
            preproc_id = "divide_mminus1"
            preproc_func = preprocessors["divide_mminus1"][1]
            kwargs = preprocessors["divide_mminus1"][2]
            description = preprocessors["divide_mminus1"][0]
        elif proposal in {"baseline_identity", "identity"}:
            preproc_id = "identity"
            preproc_func = preprocessors["identity"][1]
            kwargs = preprocessors["identity"][2]
            description = "Histograma original (control)"
        else:
            print(f"[warn] Propuesta desconocida: {proposal}. Se ignora.")
            continue

        for metric in metrics:
            metric = metric.strip().lower()
            scenarios.append(
                {
                    "name": f"{proposal} | {metric}",
                    "description": description,
                    "preproc_id": preproc_id,
                    "preproc_func": preproc_func,
                    "preproc_kwargs": kwargs,
                    "metric": metric,
                }
            )
    if include_identity:
        for metric in metrics:
            if not any(s["preproc_id"] == "identity" and s["metric"] == metric for s in scenarios):
                scenarios.append(
                    {
                        "name": f"identity | {metric}",
                        "description": "Histograma original (control)",
                        "preproc_id": "identity",
                        "preproc_func": preprocessors["identity"][1],
                        "preproc_kwargs": preprocessors["identity"][2],
                        "metric": metric,
                    }
                )

    return scenarios


__all__ = ["build_scenarios"]
