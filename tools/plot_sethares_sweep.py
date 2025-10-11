"""
Este script dibuja dos tipos de graficas para explicar, de forma visual,
como se comporta un modelo teorico de rugosidad musical (tipo Sethares):

- Barrido de decaimiento: comparamos varias curvas cambiando el parametro
  de decaimiento de los armonicos (por ejemplo 0.82, 0.88 y 0.94), con un
  numero fijo de armonicos.
- Barrido de armonicos: comparamos varias curvas cambiando cuantos armonicos
  usamos (por ejemplo N=6, 8 y 10), con el decaimiento fijo.

En ambas figuras mostramos:
- Una linea suave y continua (la curva) para ver la tendencia general.
- Unos puntos en cada semitono (marcadores) para que se vea tambien el
  comportamiento en los intervalos musicales exactos.
- Una linea roja punteada en la octava para ubicar ese punto de referencia.
- Nombres cortos de los intervalos en el eje X (Unisono, 2m, 2M, ... 15va).

Uso rapido desde consola:
  python -m tools.plot_sethares_sweep --base-freq 500 \
      --decays 0.82 0.88 0.94 --n-fixed 6 \
      --n-list 6 8 10 --decay-fixed 0.88 \
      --max-semitones 24 --dense-points 800 \
      --out outputs/sethares

Si no entiendes programacion, no pasa nada: piensa que el script recibe
los parametros (por ejemplo el decaimiento o cuantos armonicos) y dibuja
las curvas por ti, guardando las imagenes en la carpeta indicada.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Etiquetas musicales para cada semitono desde 0 hasta 24.
# 0 es Unisono (mismas frecuencias) y 12 es la Octava. 24 es 15va (doble octava).
# Usamos abreviaturas: m=menor, M=mayor, J=justa, Trit.=tritono.
INTERVAL_LABELS_24 = [
    "Unisono",
    "2b",
    "2",
    "3b",
    "3",
    "4",
    "Tritono",
    "5",
    "6b",
    "6",
    "7b",
    "7",
    "Octava",
    "8va+2b",
    "8va+2",
    "8va+3b",
    "8va+3",
    "8va+4",
    "8va+Trit.",
    "8va+5",
    "8va+6b",
    "8va+6",
    "8va+7b",
    "8va+7",
    "Doble octava",
]


@dataclass
class TheoreticalParams:
    base_freq: float = 440.0
    n_harmonics: int = 6
    decay: float = 0.88
    amplitude_mode: str = "product"  # "product" or "min"
    C1: float = 5.0
    C2: float = -5.0
    A1: float = -3.51
    A2: float = -5.75
    Dstar: float = 0.24
    S1: float = 0.0207
    S2: float = 18.96


def _pair_theoretical(f1: float, f2: float, a1: float, a2: float, p: TheoreticalParams) -> float:
    """Rugosidad entre dos parciales (una pareja de frecuencias).

    No hace falta memorizar la formula: solo ten en cuenta que la
    rugosidad depende de cuan separadas estan las frecuencias y de la
    amplitud relativa de cada parcial (controlada por el decaimiento).
    """
    fmin = min(f1, f2)
    s = p.Dstar / (p.S1 * fmin + p.S2)
    df = abs(f2 - f1)
    if p.amplitude_mode == "product":
        a = a1 * a2
    elif p.amplitude_mode == "min":
        a = min(a1, a2)
    else:
        raise ValueError("amplitude_mode must be 'product' or 'min'")
    return a * (p.C1 * np.exp(p.A1 * s * df) + p.C2 * np.exp(p.A2 * s * df))


def theoretical_total_roughness(fundamentals: List[float], p: TheoreticalParams) -> float:
    """Rugosidad total de dos notas superpuestas.

    Suma la contribucion de todas las parejas de parciales entre las dos
    notas. Si N es el numero de armonicos, se comparan N x N parejas.
    """
    total = 0.0
    for i in range(len(fundamentals) - 1):
        for j in range(i + 1, len(fundamentals)):
            f1, f2 = fundamentals[i], fundamentals[j]
            for k1 in range(1, p.n_harmonics + 1):
                for k2 in range(1, p.n_harmonics + 1):
                    p1 = f1 * k1
                    p2 = f2 * k2
                    a1 = p.decay ** (k1 - 1)
                    a2 = p.decay ** (k2 - 1)
                    total += _pair_theoretical(p1, p2, a1, a2, p)
    return float(total)


def _roughness_for_intervals(base_freq: float, semitone_values: Sequence[float], p: TheoreticalParams) -> List[float]:
    """Calcula la rugosidad para varios intervalos dados en semitonos.

    Ejemplo: si semitone_values contiene [0, 1, 2], se evalua Unisono,
    segunda menor y segunda mayor respecto a la frecuencia base.
    """
    results: List[float] = []
    for semitone in semitone_values:
        ratio = 2 ** (semitone / 12.0)
        freqs = [base_freq, base_freq * ratio]
        results.append(theoretical_total_roughness(freqs, p))
    return results


def cont_curve(
    base_freq: float,
    p: TheoreticalParams,
    *,
    max_semitones: float = 24.0,
    num_points: int = 800,
) -> Tuple[np.ndarray, List[float]]:
    """Curva continua y suave a lo largo de los intervalos.

    Recorre el eje X de forma fina (muchos puntos) desde Unisono hasta la
    segunda octava. Esta es la linea suave que ves en las figuras.
    """
    semitones = np.linspace(0.0, max_semitones, num_points, endpoint=True)
    roughness = _roughness_for_intervals(base_freq, semitones, p)
    return semitones, roughness


def dyad_curve(
    base_freq: float,
    p: TheoreticalParams,
    *,
    max_semitones: int = 24,
    step: int = 1,
) -> Tuple[np.ndarray, List[float]]:
    """Valores discretos en cada semitono (o cada 'step').

    Estos puntos se usan como marcadores sobre la curva, para destacar
    los intervalos musicales exactos (0, 1, 2, ... 24 semitonos).
    """
    if step <= 0:
        raise ValueError("step must be a positive integer")
    semitones = np.arange(0, max_semitones + step, step)
    roughness = _roughness_for_intervals(base_freq, semitones, p)
    return semitones, roughness


def normalize(vals: Sequence[float], vmax: float | None = None) -> List[float]:
    """Escala los valores a 0-1 para poder comparar curvas.

    La forma de la curva no cambia: solo se ajusta la altura maxima a 1.
    """
    if not vals:
        return []
    if vmax is None:
        vmax = max(vals)
    if vmax == 0:
        return [0.0 for _ in vals]
    return [v / vmax for v in vals]


def _interval_labels(max_semitones: int) -> List[str]:
    """Devuelve los nombres de intervalos hasta max_semitones.

    Se usa para rotular el eje X de manera musical, en vez de numeros.
    """
    if max_semitones > 24:
        raise ValueError("interval labels defined up to 24 semitones")
    if max_semitones < 0:
        raise ValueError("max_semitones must be non-negative")
    return INTERVAL_LABELS_24[: max_semitones + 1]


def _apply_axes_style(
    ax: plt.Axes,
    max_semitones: float,
    *,
    xticks: Sequence[float] | None = None,
    xtick_labels: Sequence[str] | None = None,
    tick_rotation: float | None = None,
    normalized: bool = True,
    y_max: float | None = None,
    y_label: str | None = None,
) -> None:
    """Pone los ejes con un estilo claro para articulos.

    Rejilla suave, limites, etiquetas y ocultar marcos superiores.
    Si ``normalized`` es False, el limite vertical se ajusta con ``y_max``.
    """
    ax.set_xlim(0, max_semitones)
    if xticks is None:
        xticks = np.arange(0, max_semitones + 1, 3)
    ax.set_xticks(xticks)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
        if tick_rotation is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(tick_rotation)
                label.set_horizontalalignment("right")
    ax.set_xlabel("Intervalo (semitonos)")
    ax.set_ylabel(y_label or ("Rugosidad normalizada" if normalized else "Rugosidad"))
    if normalized:
        ax.set_ylim(0, 1.05)
    elif y_max is not None and y_max > 0:
        ax.set_ylim(0, y_max * 1.05)
    ax.grid(True, color="#b0b0b0", linewidth=0.6, alpha=0.4)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_decay_sweep(
    base_freq: float,
    n_fixed: int,
    decays: Sequence[float],
    out_dir: Path,
    *,
    amplitude_mode: str = "product",
    max_semitones: float = 24.0,
    dense_points: int = 800,
    normalize_curves: bool = True,
) -> None:
    """Figura 1: comparamos varios decaimientos con N armonicos fijo.

    Para cada valor de decaimiento:
    1) calculamos la curva suave (muchos puntos),
    2) la normalizamos para que el maximo sea 1,
    3) dibujamos tambien marcadores en cada semitono,
    4) etiquetamos el eje X con los nombres musicales y
    5) marcamos la octava con una linea punteada.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(decays)))

    global_ymax = 0.0
    for color, decay in zip(colors, decays):
        params = TheoreticalParams(
            base_freq=base_freq,
            n_harmonics=n_fixed,
            decay=decay,
            amplitude_mode=amplitude_mode,
        )
        sem_dense, rough_dense = cont_curve(
            base_freq,
            params,
            max_semitones=max_semitones,
            num_points=dense_points,
        )
        vmax = max(rough_dense) if rough_dense else 1.0
        if normalize_curves:
            y_dense = normalize(rough_dense, vmax=vmax)
        else:
            y_dense = rough_dense
            global_ymax = max(global_ymax, vmax)
        ax.plot(
            sem_dense,
            y_dense,
            color=color,
            linewidth=2.3,
            label=f"decay={decay:.2f}",
        )

        sem_markers, rough_markers = dyad_curve(
            base_freq,
            params,
            max_semitones=int(max_semitones),
            step=1,
        )
        if normalize_curves:
            y_markers = normalize(rough_markers, vmax=vmax)
        else:
            y_markers = rough_markers
            if rough_markers:
                global_ymax = max(global_ymax, max(rough_markers))
        ax.plot(
            sem_markers,
            y_markers,
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            alpha=0.95,
        )

    # Estilo de ejes (al final, con info completa)
    ax.set_title(
        f"Sethares teorico - barrido de decaimiento (N={n_fixed})",
        fontsize=12,
        pad=14,
    )
    ax.legend(frameon=False, fontsize=9, title="Coeficiente de decaimiento")

    xticks = np.arange(0, int(max_semitones) + 1, 1)
    labels = _interval_labels(int(max_semitones))
    _apply_axes_style(
        ax,
        max_semitones,
        xticks=xticks,
        xtick_labels=labels,
        tick_rotation=45.0,
        normalized=normalize_curves,
        y_max=global_ymax,
    )
    ax.set_xlabel("Intervalo (primera y segunda octava)")
    ax.axvline(12, color="#b22222", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(12, 1.03, "Octava", color="#b22222", fontsize=9, ha="center", va="bottom")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"__sethares_teorico_decay_sweep_b{int(base_freq)}_N{n_fixed}_m{amplitude_mode}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_harmonics_sweep(
    base_freq: float,
    n_list: Sequence[int],
    decay_fixed: float,
    out_dir: Path,
    *,
    amplitude_mode: str = "product",
    max_semitones: int = 24,
    dense_points: int = 800,
    normalize_curves: bool = True,
) -> None:
    """Figura 2: comparamos N armonicos (decay fijo).

    Misma idea que la figura 1, pero ahora cada color es un valor de N.
    Trazamos la curva suave y superponemos marcadores en semitonos.
    """
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_list)))

    global_ymax = 0.0
    for color, n_harmonics in zip(colors, n_list):
        params = TheoreticalParams(
            base_freq=base_freq,
            n_harmonics=n_harmonics,
            decay=decay_fixed,
            amplitude_mode=amplitude_mode,
        )

        # Curva continua densa
        sem_dense, rough_dense = cont_curve(
            base_freq,
            params,
            max_semitones=max_semitones,
            num_points=dense_points,
        )
        vmax = max(rough_dense) if rough_dense else 1.0
        if normalize_curves:
            y_dense = normalize(rough_dense, vmax=vmax)
        else:
            y_dense = rough_dense
            global_ymax = max(global_ymax, vmax)
        ax.plot(
            sem_dense,
            y_dense,
            color=color,
            linewidth=2.3,
            label=f"N={n_harmonics}",
        )

        # Marcadores discretos en cada semitono
        sem_markers, rough_markers = dyad_curve(
            base_freq,
            params,
            max_semitones=max_semitones,
            step=1,
        )
        if normalize_curves:
            y_markers = normalize(rough_markers, vmax=vmax)
        else:
            y_markers = rough_markers
            if rough_markers:
                global_ymax = max(global_ymax, max(rough_markers))
        ax.plot(
            sem_markers,
            y_markers,
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            alpha=0.95,
        )

    xticks = np.arange(0, max_semitones + 1, 1)
    labels = _interval_labels(max_semitones)
    _apply_axes_style(
        ax,
        max_semitones,
        xticks=xticks,
        xtick_labels=labels,
        tick_rotation=45.0,
        normalized=normalize_curves,
        y_max=global_ymax,
    )
    ax.set_xlabel("Intervalo (primera y segunda octava)")
    ax.set_title(
        f"Sethares teorico - comparacion de N-harmonics (decay={decay_fixed:.2f})",
        fontsize=12,
        pad=16,
    )
    ax.axvline(12, color="#b22222", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(12, 1.03, "Octava", color="#b22222", fontsize=9, ha="center", va="bottom")
    ax.legend(frameon=False, fontsize=9, title="N-harmonics")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"sethares_teorico_harmonics_sweep_b{int(base_freq)}_d{decay_fixed}_m{amplitude_mode}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_single_curve(
    base_freq: float,
    n_fixed: int,
    decay_fixed: float,
    out_dir: Path,
    *,
    amplitude_mode: str = "product",
    max_semitones: int = 24,
    dense_points: int = 800,
    normalize_curves: bool = False,
) -> None:
    """Figura unica: una sola curva con N fijo y decaimiento fijo.

    Mantiene el mismo estilo: curva suave + marcadores por semitono,
    etiquetas musicales en el eje X y linea de referencia en la octava.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)

    color = plt.cm.viridis(0.55)
    params = TheoreticalParams(
        base_freq=base_freq,
        n_harmonics=n_fixed,
        decay=decay_fixed,
        amplitude_mode=amplitude_mode,
    )

    # Curva continua
    sem_dense, rough_dense = cont_curve(
        base_freq, params, max_semitones=max_semitones, num_points=dense_points
    )
    vmax = max(rough_dense) if rough_dense else 1.0
    if normalize_curves:
        y_dense = normalize(rough_dense, vmax=vmax)
        global_ymax = 1.0
    else:
        y_dense = rough_dense
        global_ymax = vmax
    ax.plot(sem_dense, y_dense, color=color, linewidth=2.6)

    # Marcadores discretos
    sem_markers, rough_markers = dyad_curve(
        base_freq, params, max_semitones=max_semitones, step=1
    )
    y_markers = normalize(rough_markers, vmax=vmax) if normalize_curves else rough_markers
    if not normalize_curves and rough_markers:
        global_ymax = max(global_ymax, max(rough_markers))
    ax.plot(
        sem_markers,
        y_markers,
        linestyle="none",
        marker="o",
        markersize=4.5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        alpha=0.95,
    )

    # Ejes y decoracion
    xticks = np.arange(0, max_semitones + 1, 1)
    labels = _interval_labels(max_semitones)
    _apply_axes_style(
        ax,
        max_semitones,
        xticks=xticks,
        xtick_labels=labels,
        tick_rotation=45.0,
        normalized=normalize_curves,
        y_max=global_ymax,
    )
    ax.set_xlabel("Intervalo (primera y segunda octava)")
    ax.set_title(
        f"Sethares teorico - curva (N={n_fixed}, decay={decay_fixed:.2f})",
        fontsize=12,
        pad=16,
    )
    ax.axvline(12, color="#b22222", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(12, 1.03, "Octava", color="#b22222", fontsize=9, ha="center", va="bottom")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        out_dir
        / f"sethares_teorico_single_curve_b{int(base_freq)}_N{n_fixed}_d{decay_fixed}_m{amplitude_mode}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_amplitude_mode(
    base_freq: float,
    n_fixed: int,
    decay_fixed: float,
    out_dir: Path,
    *,
    max_semitones: float = 24.0,
    normalize_curves: bool = True,
) -> None:
    """Figura opcional: compara dos formas de combinar amplitudes.

    Normalmente no hace falta tocarla, pero queda aqui por si quieres
    ver la diferencia entre usar el producto o el minimo de amplitudes.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    colors = {"product": "#1f77b4", "min": "#d62728"}

    for mode in ["product", "min"]:
        params = TheoreticalParams(
            base_freq=base_freq,
            n_harmonics=n_fixed,
            decay=decay_fixed,
            amplitude_mode=mode,
        )
        semitones, roughness = dyad_curve(
            base_freq,
            params,
            max_semitones=int(max_semitones),
            step=1,
        )
        if normalize_curves:
            norm = normalize(roughness)
            yvals = norm
        else:
            yvals = roughness
        ax.plot(
            semitones,
            yvals,
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=colors[mode],
            label=f"modo={mode}",
        )

    # En esta figura no calculamos y_max global (no es la figura principal)
    _apply_axes_style(ax, max_semitones, normalized=normalize_curves)
    ax.set_title(
        f"Sethares teorico - modo de amplitud (N={n_fixed}, decay={decay_fixed})",
        fontsize=12,
        pad=14,
    )
    ax.legend(frameon=False, fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"sethares_teorico_amplitude_modes_b{int(base_freq)}_N{n_fixed}_d{decay_fixed}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """Punto de entrada cuando ejecutas el archivo como modulo.

    Lee los parametros de la linea de comandos y llama a las funciones
    que generan y guardan las figuras.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-freq", type=float, default=500.0)
    ap.add_argument("--decays", type=float, nargs="*", default=[0.82, 0.88, 0.94])
    ap.add_argument("--n-fixed", type=int, default=6)
    ap.add_argument("--n-list", type=int, nargs="*", default=[6, 8, 10])
    ap.add_argument("--decay-fixed", type=float, default=0.88)
    ap.add_argument("--max-semitones", type=int, default=24)
    ap.add_argument("--dense-points", type=int, default=800)
    ap.add_argument("--out", type=Path, default=Path("outputs/sethares"))
    # Normalizacion opcional de las curvas (0-1). Por defecto DESACTIVADA.
    ap.add_argument("--normalize", dest="normalize", action="store_true", help="Normaliza curvas a 0-1")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", help="Muestra valores sin normalizar (por defecto)")
    ap.set_defaults(normalize=False)
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    plot_decay_sweep(
        args.base_freq,
        args.n_fixed,
        args.decays,
        out,
        amplitude_mode="product",
        max_semitones=float(args.max_semitones),
        dense_points=args.dense_points,
        normalize_curves=args.normalize,
    )
    plot_harmonics_sweep(
        args.base_freq,
        args.n_list,
        args.decay_fixed,
        out,
        amplitude_mode="product",
        max_semitones=args.max_semitones,
        dense_points=args.dense_points,
        normalize_curves=args.normalize,
    )
    # Figura unica adicional (misma configuracion de N y decay fijos)
    plot_single_curve(
        args.base_freq,
        args.n_fixed,
        args.decay_fixed,
        out,
        amplitude_mode="product",
        max_semitones=args.max_semitones,
        dense_points=args.dense_points,
        normalize_curves=args.normalize,
    )
    # plot_amplitude_mode(
    #     args.base_freq,
    #     args.n_fixed,
    #     args.decay_fixed,
    #     out,
    #     max_semitones=float(args.max_semitones),
    # )

    print("Saved figures to:", out)


if __name__ == "__main__":
    main()
