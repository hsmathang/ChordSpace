"""
GUI launcher for ChordSpace experiments.

Provides a simple Tk interface to configure populations, manage query definitions
and execute the experiment runner without dealing with command line syntax.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from tools import experiment_inversions
from tools.query_registry import add_custom_query, get_all_queries, resolve_query_sql
from tools.compare_proposals import PROPOSAL_INFO, METRIC_INFO, AVAILABLE_REDUCTIONS, PREPROCESSORS
from tools.population_utils import dedupe_population
from config import config_db
from config import CHORD_TEMPLATES_METADATA
import subprocess
import sys
import re
import webbrowser
import datetime as _dt
import pandas as pd
try:
    from chordcodex.model import QueryExecutor  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from synth_tools import QueryExecutor  # type: ignore
from tools.experiment_inversions import _parse_pop_spec, _build_population
import time
from config import MODELO_OPTIONS_LIST, METRICA_OPTIONS_LIST, PONDERACION_OPTIONS_LIST


class QueueWriter(io.TextIOBase):
    """Redirects stdout/stderr writes into a queue for the GUI to consume."""

    def __init__(self, bucket: queue.Queue, category: str = "log"):
        super().__init__()
        self.bucket = bucket
        self.category = category

    def write(self, msg: str) -> int:
        if msg:
            self.bucket.put((self.category, msg))
        return len(msg)

    def flush(self) -> None:  # pragma: no cover - nothing to flush
        return None


class ExperimentLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ChordSpace - Experiment Launcher")
        self.geometry("1060x760")
        self.minsize(900, 660)

        self.log_queue: queue.Queue = queue.Queue()
        self.running_thread: threading.Thread | None = None
        self.pops_entries: list[str] = []

        self._load_queries()
        self._init_state_vars()
        # Estado para la nueva pestaña de población/preview
        self.population_df: pd.DataFrame | None = None
        self.population_selected_rows: set[int] = set()
        self.population_row_ids: dict[int, int | None] = {}
        self._create_layout()

        self.after(100, self._process_log_queue)

    # ------------------------------------------------------------------ setup
    def _load_queries(self) -> None:
        self.query_registry = get_all_queries()

    def _init_state_vars(self) -> None:
        self.type_var = tk.StringVar(value="B")
        self.reduction_var = tk.StringVar(value="MDS")
        options = ["<Ninguna>"] + sorted(self.query_registry.keys())
        # Por defecto no usar consulta base; solo A/B/C a menos que el usuario elija una
        self.base_query_var = tk.StringVar(value="<Ninguna>")
        self.output_var = tk.StringVar(value=str(self._default_output_dir()))

        self.pop_type_var = tk.StringVar(value="A")
        pop_default_query = options[1] if len(options) > 1 else "<Ninguna>"
        self.pop_query_var = tk.StringVar(value=pop_default_query)

        self.custom_name_var = tk.StringVar()

        # Model/Metric/Ponderation dictionaries (label -> value) and state vars (labels)
        self.model_label_to_value = {label: value for (label, value) in MODELO_OPTIONS_LIST}
        self.metric_label_to_value = {label: value for (label, value) in METRICA_OPTIONS_LIST}
        self.ponder_label_to_value = {label: value for (label, value) in PONDERACION_OPTIONS_LIST}

        self.model_var = tk.StringVar(value=list(self.model_label_to_value.keys())[0])
        self.metric_var = tk.StringVar(value=list(self.metric_label_to_value.keys())[0])
        self.ponder_var = tk.StringVar(value=list(self.ponder_label_to_value.keys())[0])
        self.base_query_var.trace_add("write", lambda *_: self._mark_population_dirty())

        self._init_compare_vars()

    def _init_compare_vars(self) -> None:
        # Proposals: combinación de definidos en PROPOSAL_INFO y PREPROCESSORS
        all_props = sorted(set(PREPROCESSORS.keys()) | set(PROPOSAL_INFO.keys()))
        default_props = {"baseline_identity", "simplex", "perclass_alpha1"}
        self.proposals_order: list[str] = all_props
        self.proposal_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=(name in default_props))
            for name in all_props
        }
        self.proposal_display_map = {
            name: f"{PROPOSAL_INFO.get(name, {}).get('title', name)} ({name})"
            for name in self.proposals_order
        }
        self.proposal_display_inverse = {display: name for name, display in self.proposal_display_map.items()}

        # Métricas disponibles
        metric_keys = sorted(METRIC_INFO.keys())
        default_metrics = {"euclidean"}
        self.metrics_order: list[str] = metric_keys
        self.metric_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=(name in default_metrics))
            for name in metric_keys
        }

        # Reducciones soportadas (todas seleccionadas por defecto)
        self.reductions_order: list[str] = list(AVAILABLE_REDUCTIONS)
        self.reduction_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=True) for name in self.reductions_order
        }

        # Estado para comparación de reducciones (propuesta única)
        default_prop_display = self.proposal_display_map.get(self.proposals_order[0], "") if self.proposals_order else ""
        self.reduction_compare_prop_var = tk.StringVar(value=default_prop_display)
        self.reduction_compare_metric_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=(name in default_metrics))
            for name in metric_keys
        }
        self.reduction_compare_reduction_vars: dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=True) for name in self.reductions_order
        }
        self.reduction_compare_seeds_var = tk.StringVar(value="42")

    def _create_layout(self) -> None:
        main = ttk.Frame(self, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        # Notebook con 3 pestañas
        nb = ttk.Notebook(main)
        nb.pack(fill=tk.BOTH, expand=True)

        tab_population = ttk.Frame(nb)
        tab_experiment = ttk.Frame(nb)
        tab_compare = ttk.Frame(nb)
        tab_compare_reduction = ttk.Frame(nb)
        nb.add(tab_population, text="Elegir población")
        nb.add(tab_experiment, text="Parámetros de experimento")
        nb.add(tab_compare, text="Parámetros de comparación")
        nb.add(tab_compare_reduction, text="Comparar reducciones")

        # Tab: Población
        pop_container = ttk.Frame(tab_population, padding=6)
        pop_container.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(pop_container, text="Fuente base y salida")
        config_frame.pack(fill=tk.X, pady=(0, 8))
        self._build_population_config_frame(config_frame)

        middle_frame = ttk.Frame(pop_container)
        middle_frame.pack(fill=tk.BOTH, expand=True)
        left_col = ttk.Frame(middle_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        pops_frame = ttk.LabelFrame(left_col, text="Poblaciones conjuntas (A/B/C)")
        pops_frame.pack(fill=tk.BOTH, expand=True)
        self._build_pops_frame(pops_frame)

        preview_row = ttk.Frame(pop_container)
        preview_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(preview_row, text="Construir / Previsualizar", command=self._on_population_preview_clicked).pack(side=tk.LEFT)
        ttk.Label(preview_row, text="(usa la lista A/B/C y la consulta base opcional)").pack(side=tk.LEFT, padx=(8,0))

        table_frame = ttk.Frame(pop_container)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        columns = ("use","id","n","interval","notes","code","bass","octave","tag","span_semitones","abs_mask_int","abs_mask_hex")
        self.pop_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        for c in columns:
            self.pop_tree.heading(c, text=c)
            self.pop_tree.column(c, width=90 if c != "interval" else 120, anchor="center")
        self.pop_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.pop_tree.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pop_tree.configure(yscrollcommand=scroll.set)
        self.pop_tree.bind("<Double-1>", self._on_population_toggle_row)

        sel_tools = ttk.Frame(pop_container)
        sel_tools.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(sel_tools, text="Seleccionar todo", command=self._population_select_all).pack(side=tk.LEFT)
        ttk.Button(sel_tools, text="Limpiar", command=self._population_clear).pack(side=tk.LEFT, padx=(6,0))
        ttk.Button(sel_tools, text="Invertir", command=self._population_invert).pack(side=tk.LEFT, padx=(6,0))

        self.pop_stats_var = tk.StringVar(value="—")
        ttk.Label(pop_container, textvariable=self.pop_stats_var, foreground="#555").pack(fill=tk.X, padx=6, pady=(0,6))

        custom_frame = ttk.LabelFrame(pop_container, text="Guardar nueva consulta (opcional)")
        custom_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self._build_custom_query_frame(custom_frame)

        pop_log_frame = ttk.LabelFrame(pop_container, text="Registro de población")
        pop_log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.pop_log = ScrolledText(pop_log_frame, height=8, state=tk.DISABLED, font=("Consolas", 10))
        self.pop_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab: Experimento
        exp_container = ttk.Frame(tab_experiment, padding=6)
        exp_container.pack(fill=tk.BOTH, expand=True)
        exp_frame = ttk.LabelFrame(exp_container, text="Parámetros de experimento")
        exp_frame.pack(fill=tk.X)
        self._build_experiment_params_frame(exp_frame)
        exp_actions = ttk.Frame(exp_container)
        exp_actions.pack(fill=tk.X, pady=(10, 0))
        self.run_button = ttk.Button(exp_actions, text="Ejecutar experimento", command=self._on_run_clicked)
        self.run_button.pack(side=tk.LEFT)

        exp_log_frame = ttk.LabelFrame(exp_container, text="Registro del experimento")
        exp_log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.exp_log = ScrolledText(exp_log_frame, height=8, state=tk.DISABLED, font=("Consolas", 10))
        self.exp_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab: Comparación
        compare_container = ttk.Frame(tab_compare, padding=6)
        compare_container.pack(fill=tk.BOTH, expand=True)
        compare_frame = ttk.LabelFrame(compare_container, text="Reporte de comparación")
        compare_frame.pack(fill=tk.BOTH, expand=True)
        self._build_compare_frame(compare_frame)

        compare_log_frame = ttk.LabelFrame(compare_container, text="Registro de comparación")
        compare_log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.compare_log = ScrolledText(compare_log_frame, height=8, state=tk.DISABLED, font=("Consolas", 10))
        self.compare_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab: Comparación de reducciones
        red_container = ttk.Frame(tab_compare_reduction, padding=6)
        red_container.pack(fill=tk.BOTH, expand=True)
        red_frame = ttk.LabelFrame(red_container, text="Comparación de reducciones dimensionales")
        red_frame.pack(fill=tk.BOTH, expand=True)
        self._build_reduction_compare_frame(red_frame)

        red_log_frame = ttk.LabelFrame(red_container, text="Registro comparación reducciones")
        red_log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.reduction_log = ScrolledText(red_log_frame, height=8, state=tk.DISABLED, font=("Consolas", 10))
        self.reduction_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Log global
        log_frame = ttk.LabelFrame(main, text="Registro de ejecución")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_widget = ScrolledText(log_frame, height=10, state=tk.DISABLED, font=("Consolas", 10))
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Listo.")
        status_label = ttk.Label(main, textvariable=self.status_var, foreground="#555", anchor="w")
        status_label.pack(fill=tk.X, pady=(6, 0))

    # --------------------------- sub frames
    def _build_population_config_frame(self, frame: ttk.Frame) -> None:
        # Solo la carpeta de salida; se retira la complejidad de 'consulta base'
        ttk.Label(frame, text="Carpeta de salida:").grid(row=0, column=0, sticky="w", pady=(4, 0))
        out_entry = ttk.Entry(frame, textvariable=self.output_var, width=58)
        out_entry.grid(row=0, column=1, sticky="we", padx=(4, 4), pady=(4, 0))
        ttk.Button(frame, text="Examinar…", command=self._choose_output_dir).grid(row=0, column=2, sticky="e", pady=(4, 0))

        ttk.Label(frame, text="Consulta base (opcional):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        base_values = ["<Ninguna>"] + sorted(self.query_registry.keys())
        self.base_query_combo = ttk.Combobox(frame, textvariable=self.base_query_var, values=base_values, state="readonly", width=44)
        self.base_query_combo.grid(row=1, column=1, sticky="we", padx=(4,4), pady=(6,0))
        self.base_query_combo.bind("<<ComboboxSelected>>", lambda *_: self._mark_population_dirty())
        frame.columnconfigure(1, weight=1)

    def _build_experiment_params_frame(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Reducción dimensional:").grid(row=0, column=0, sticky="w")
        red_combo = ttk.Combobox(frame, textvariable=self.reduction_var, values=["MDS", "UMAP"], width=8, state="readonly")
        red_combo.grid(row=0, column=1, padx=(4, 16), sticky="w")

        ttk.Label(frame, text="Modelo:").grid(row=0, column=2, sticky="w")
        model_labels = list(self.model_label_to_value.keys())
        self.model_combo = ttk.Combobox(frame, textvariable=self.model_var, values=model_labels, width=20, state="readonly")
        self.model_combo.grid(row=0, column=3, sticky="w", padx=(4, 16))

        ttk.Label(frame, text="Métrica:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        metric_labels = list(self.metric_label_to_value.keys())
        self.metric_combo = ttk.Combobox(frame, textvariable=self.metric_var, values=metric_labels, width=18, state="readonly")
        self.metric_combo.grid(row=1, column=1, sticky="w", padx=(4, 16), pady=(6, 0))

        ttk.Label(frame, text="Ponderación:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ponder_labels = list(self.ponder_label_to_value.keys())
        self.ponder_combo = ttk.Combobox(frame, textvariable=self.ponder_var, values=ponder_labels, width=22, state="readonly")
        self.ponder_combo.grid(row=1, column=3, sticky="w", padx=(4, 16), pady=(6, 0))

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

    def _build_pops_frame(self, frame: ttk.Frame) -> None:
        selector = ttk.Frame(frame)
        selector.pack(fill=tk.X, padx=6, pady=6)

        ttk.Label(selector, text="Tipo:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(selector, textvariable=self.pop_type_var, values=["A", "B", "C"], width=6, state="readonly").grid(row=0, column=1, padx=(4, 12))

        ttk.Label(selector, text="Consulta:").grid(row=0, column=2, sticky="w")
        self.pop_query_combo = ttk.Combobox(selector, textvariable=self.pop_query_var,
                                            values=sorted(self.query_registry.keys()), width=36, state="readonly")
        self.pop_query_combo.grid(row=0, column=3, padx=(4, 12), sticky="we")
        selector.columnconfigure(3, weight=1)

        ttk.Button(selector, text="Agregar población", command=self._add_population).grid(row=0, column=4, sticky="e")

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.pops_listbox = tk.Listbox(list_frame, height=4, selectmode=tk.SINGLE)
        self.pops_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pops_listbox.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.pops_listbox.configure(yscrollcommand=scroll.set)

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(buttons, text="Eliminar seleccionada", command=self._remove_selected_pop).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Limpiar lista", command=self._clear_pops).pack(side=tk.RIGHT)

    def _build_custom_query_frame(self, frame: ttk.Frame) -> None:
        form = ttk.Frame(frame)
        form.pack(fill=tk.X, padx=6, pady=6)

        ttk.Label(form, text="Nombre (sin espacios):").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.custom_name_var).grid(row=0, column=1, sticky="we", padx=(4, 0))
        form.columnconfigure(1, weight=1)

        ttk.Label(frame, text="SQL (SELECT / WITH):").pack(anchor="w", padx=6)
        self.custom_sql_text = tk.Text(frame, height=6, font=("Consolas", 10))
        self.custom_sql_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(button_row, text="Guardar consulta", command=self._save_custom_query).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Limpiar", command=self._clear_custom_form).pack(side=tk.RIGHT)

        hint = ttk.Label(frame, text="Las consultas nuevas se guardan en tools/custom_queries.json.", foreground="#777")
        hint.pack(anchor="w", padx=6, pady=(0, 6))

    # ------------------------ comparison report UI
    def _build_compare_frame(self, frame: ttk.Frame) -> None:
        params = ttk.Frame(frame)
        params.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))

        # Proposals
        prop_frame = ttk.LabelFrame(params, text="Proposals")
        prop_frame.grid(row=0, column=0, sticky="nwe", padx=(0, 12), pady=(0, 8))
        for idx, name in enumerate(self.proposals_order):
            title = PROPOSAL_INFO.get(name, {}).get("title", name)
            label_text = f"{title} ({name})"
            ttk.Checkbutton(
                prop_frame,
                text=label_text,
                variable=self.proposal_vars[name],
            ).grid(row=idx // 2, column=idx % 2, sticky="w", padx=6, pady=3)
        prop_frame.columnconfigure(0, weight=1)
        prop_frame.columnconfigure(1, weight=1)

        # Métricas
        metric_frame = ttk.LabelFrame(params, text="Métricas")
        metric_frame.grid(row=0, column=1, sticky="nwe", pady=(0, 8))
        for idx, name in enumerate(self.metrics_order):
            title = METRIC_INFO.get(name, {}).get("title", name.title())
            label_text = f"{title} ({name})"
            ttk.Checkbutton(
                metric_frame,
                text=label_text,
                variable=self.metric_vars[name],
            ).grid(row=idx, column=0, sticky="w", padx=6, pady=3)
        metric_frame.columnconfigure(0, weight=1)

        # Reducciones
        reduction_frame = ttk.LabelFrame(params, text="Reducciones")
        reduction_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 8))
        for idx, name in enumerate(self.reductions_order):
            ttk.Checkbutton(
                reduction_frame,
                text=name,
                variable=self.reduction_vars[name],
            ).grid(row=0, column=idx, sticky="w", padx=6, pady=3)
            reduction_frame.columnconfigure(idx, weight=1)

        # Semillas
        seed_row = ttk.Frame(params)
        seed_row.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(seed_row, text="Seeds:").pack(side=tk.LEFT)
        self.compare_seeds_var = tk.StringVar(value="42")
        ttk.Entry(seed_row, textvariable=self.compare_seeds_var, width=18).pack(side=tk.LEFT, padx=(6,0))

        params.columnconfigure(0, weight=1)
        params.columnconfigure(1, weight=1)

        # Row: run/open
        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, padx=6, pady=(4,6))
        self.compare_run_button = ttk.Button(actions, text="Ejecutar comparación", command=self._on_compare_run_clicked)
        self.compare_run_button.pack(side=tk.LEFT)
        self.compare_open_button = ttk.Button(actions, text="Abrir reporte", command=self._on_compare_open_clicked, state=tk.DISABLED)
        self.compare_open_button.pack(side=tk.LEFT, padx=(8,0))
        self.compare_status_var = tk.StringVar(value="—")
        ttk.Label(actions, textvariable=self.compare_status_var, foreground="#555").pack(side=tk.RIGHT)

        # State holders (población se gestiona en la primera pestaña)
        self.compare_last_report: Path | None = None

    def _build_reduction_compare_frame(self, frame: ttk.Frame) -> None:
        params = ttk.Frame(frame)
        params.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))

        # Propuesta (única)
        prop_row = ttk.Frame(params)
        prop_row.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0,8))
        ttk.Label(prop_row, text="Propuesta:").pack(side=tk.LEFT)
        prop_values = [self.proposal_display_map[name] for name in self.proposals_order]
        self.reduction_prop_combo = ttk.Combobox(
            prop_row,
            textvariable=self.reduction_compare_prop_var,
            values=prop_values,
            state="readonly",
            width=44,
        )
        if prop_values:
            self.reduction_prop_combo.set(prop_values[0])
        self.reduction_prop_combo.pack(side=tk.LEFT, padx=(8,0))

        # Métricas
        metric_frame = ttk.LabelFrame(params, text="Métricas")
        metric_frame.grid(row=1, column=0, sticky="nwe", pady=(0, 8))
        for idx, name in enumerate(self.metrics_order):
            title = METRIC_INFO.get(name, {}).get("title", name.title())
            label_text = f"{title} ({name})"
            ttk.Checkbutton(
                metric_frame,
                text=label_text,
                variable=self.reduction_compare_metric_vars[name],
            ).grid(row=idx, column=0, sticky="w", padx=6, pady=3)
        metric_frame.columnconfigure(0, weight=1)

        # Reducciones
        reduction_frame = ttk.LabelFrame(params, text="Reducciones")
        reduction_frame.grid(row=1, column=1, sticky="nwe", pady=(0,8))
        for idx, name in enumerate(self.reductions_order):
            ttk.Checkbutton(
                reduction_frame,
                text=name,
                variable=self.reduction_compare_reduction_vars[name],
            ).grid(row=0, column=idx, sticky="w", padx=6, pady=3)
            reduction_frame.columnconfigure(idx, weight=1)

        # Semillas
        seed_row = ttk.Frame(params)
        seed_row.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(seed_row, text="Seeds:").pack(side=tk.LEFT)
        ttk.Entry(seed_row, textvariable=self.reduction_compare_seeds_var, width=18).pack(side=tk.LEFT, padx=(6,0))

        params.columnconfigure(0, weight=1)
        params.columnconfigure(1, weight=1)

        # Botones
        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, padx=6, pady=(4,6))
        self.reduction_run_button = ttk.Button(actions, text="Comparar reducciones", command=self._on_reduction_run_clicked)
        self.reduction_run_button.pack(side=tk.LEFT)
        self.reduction_open_button = ttk.Button(actions, text="Abrir reporte", command=self._on_reduction_open_clicked, state=tk.DISABLED)
        self.reduction_open_button.pack(side=tk.LEFT, padx=(8,0))
        self.reduction_status_var = tk.StringVar(value="—")
        ttk.Label(actions, textvariable=self.reduction_status_var, foreground="#555").pack(side=tk.RIGHT)

        self.reduction_last_report: Path | None = None

    # ------------------------ comparison logic
    def _on_population_preview_clicked(self) -> None:
        try:
            t0 = time.perf_counter()
            df = self._load_population()
            t1 = time.perf_counter()
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Error al cargar población", str(exc))
            return
        if df is None or df.empty:
            messagebox.showwarning("Población vacía", "No se obtuvieron filas. Ajusta la fuente y vuelve a intentar.")
            return
        # Reporte por fuente (antes del dedupe)
        try:
            if "__source__" in df.columns:
                counts_raw = df["__source__"].value_counts()
                self._append_pop_log("[población] Conteo por fuente (antes de dedupe):\n" +
                                     "\n".join(f"  {k}: {v}" for k, v in counts_raw.items()) + "\n")
        except Exception:
            pass
        dedup_df, dedupe_key = dedupe_population(df)
        try:
            if "__source__" in dedup_df.columns:
                counts_dedup = dedup_df["__source__"].value_counts()
                self._append_pop_log("[población] Conteo por fuente (después de dedupe):\n" +
                                     "\n".join(f"  {k}: {v}" for k, v in counts_dedup.items()) + "\n")
                self._append_pop_log(f"[población] Clave de dedupe usada: {dedupe_key}\n")
        except Exception:
            pass
        self.population_df = dedup_df.reset_index(drop=True)
        self._fill_population_tree()
        # Stats y log
        stats_text = self._compute_population_stats(self.population_df)
        self.pop_stats_var.set(stats_text)
        cardinality_summary = ""
        if "n" in self.population_df.columns:
            counts = self.population_df["n"].value_counts().sort_index()
            cardinality_summary = ", ".join(f"n={int(idx)}:{int(val)}" for idx, val in counts.items())
        log_lines = [
            f"[población] Previsualización cargada en {(t1 - t0):.3f}s.",
            f"Total acordes: {len(self.population_df)}",
        ]
        base_selected = self.base_query_var.get().strip()
        if base_selected and base_selected != "<Ninguna>":
            log_lines.append(f"Consulta base: {base_selected}")
        if self.pops_entries:
            log_lines.append("Poblaciones conjuntas: " + ", ".join(self.pops_entries))
        # Conteo de fuentes únicas actual
        if "__source__" in self.population_df.columns:
            try:
                unique_sources = self.population_df["__source__"].value_counts()
                log_lines.append("Fuentes en tabla (post-dedupe): " +
                                 ", ".join(f"{k}:{v}" for k, v in unique_sources.items()))
            except Exception:
                pass
        if cardinality_summary:
            log_lines.append(f"Cardinalidades: {cardinality_summary}")
        if "tag" in self.population_df.columns:
            tag_counts = self.population_df["tag"].value_counts().head(5)
            if not tag_counts.empty:
                top_tags = ", ".join(f"{k}:{v}" for k, v in tag_counts.items())
                log_lines.append(f"Top tags: {top_tags}")
        self._append_pop_log("\n".join(log_lines) + "\n")
        self._append_log(f"[población] Cargada en {(t1 - t0):.3f}s. {stats_text}\n")

    def _load_population(self) -> pd.DataFrame:
        qe = QueryExecutor(**config_db)
        frames: list[pd.DataFrame] = []

        base = self.base_query_var.get().strip()
        if base and base != "<Ninguna>":
            sql = resolve_query_sql(base)
            df_base = qe.as_pandas(sql)
            try:
                df_base = df_base.copy()
                df_base["__source__"] = f"BASE:{base}"
            except Exception:
                pass
            frames.append(df_base)

        for spec in self.pops_entries:
            ptype, qname = _parse_pop_spec(spec)
            df_p = _build_population(ptype, qname)
            try:
                df_p = df_p.copy()
                df_p["__source__"] = f"{ptype}:{qname}"
            except Exception:
                pass
            frames.append(df_p)

        if not frames:
            raise ValueError("No hay fuentes configuradas. Agrega al menos una población conjunta o selecciona una consulta base.")

        return pd.concat(frames, ignore_index=True)

    def _fill_population_tree(self) -> None:
        # Clear
        for item in self.pop_tree.get_children():
            self.pop_tree.delete(item)
        self.population_selected_rows.clear()
        self.population_row_ids.clear()
        if self.population_df is None:
            return
        for idx, row in self.population_df.iterrows():
            rowd = row.to_dict()
            raw_id = rowd.get("id")
            chord_id: int | None
            if pd.notnull(raw_id):
                try:
                    chord_id = int(raw_id)
                except Exception:
                    try:
                        chord_id = int(float(raw_id))
                    except Exception:
                        chord_id = None
            else:
                chord_id = None
            display_id = chord_id if chord_id is not None else ""
            values = ["✓", display_id, rowd.get("n"), rowd.get("interval"), rowd.get("notes"), rowd.get("code"),
                      rowd.get("bass"), rowd.get("octave"), rowd.get("tag"), rowd.get("span_semitones"),
                      rowd.get("abs_mask_int"), rowd.get("abs_mask_hex")]
            self.pop_tree.insert("", tk.END, iid=str(idx), values=values)
            self.population_selected_rows.add(int(idx))
            self.population_row_ids[int(idx)] = chord_id

    def _compute_population_stats(self, df: pd.DataFrame) -> str:
        total = len(df)
        by_n = df["n"].value_counts().sort_index() if "n" in df.columns else {}
        triad_set = set()
        for tpl in CHORD_TEMPLATES_METADATA:
            intervals = tuple(int(x) for x in tpl.get("intervals", ()))
            if len(intervals) == 2:
                triad_set.add(intervals)
        seventh_set = {(4,3,4),(4,3,3),(3,4,3),(3,3,4),(3,3,3),(4,4,3)}
        triads_named = 0
        sevenths_named = 0
        if "interval" in df.columns:
            def _parse_intv(v):
                if isinstance(v, (list, tuple)):
                    return tuple(int(x) for x in v)
                if isinstance(v, str):
                    s = v.strip("{}[]")
                    parts = [p.strip() for p in s.split(',') if p.strip()]
                    return tuple(int(x) for x in parts)
                return tuple()
            for v in df["interval"].tolist():
                t = _parse_intv(v)
                if len(t) == 2 and t in triad_set:
                    triads_named += 1
                elif len(t) == 3 and t in seventh_set:
                    sevenths_named += 1
        by_n_str = ", ".join(f"n={int(k)}:{int(v)}" for k,v in (by_n.items() if hasattr(by_n,'items') else []))
        return f"Total: {total} | {by_n_str} | Triads(named): {triads_named} | Sevenths(named): {sevenths_named}"

    def _on_population_toggle_row(self, event) -> None:  # pylint: disable=unused-argument
        item = self.pop_tree.focus()
        if not item:
            return
        vals = list(self.pop_tree.item(item, "values"))
        row_idx = int(item)
        if vals[0] == "✓":
            vals[0] = ""
            self.population_selected_rows.discard(row_idx)
        else:
            vals[0] = "✓"
            self.population_selected_rows.add(row_idx)
        self.pop_tree.item(item, values=vals)

    def _population_select_all(self) -> None:
        for item in self.pop_tree.get_children():
            vals = list(self.pop_tree.item(item, "values"))
            vals[0] = "✓"
            self.pop_tree.item(item, values=vals)
            try:
                row_idx = int(item)
                self.population_selected_rows.add(row_idx)
            except Exception:
                continue

    def _population_clear(self) -> None:
        for item in self.pop_tree.get_children():
            vals = list(self.pop_tree.item(item, "values"))
            vals[0] = ""
            self.pop_tree.item(item, values=vals)
        self.population_selected_rows.clear()

    def _population_invert(self) -> None:
        current = set(self.population_selected_rows)
        self._population_clear()
        for item in self.pop_tree.get_children():
            vals = list(self.pop_tree.item(item, "values"))
            row_idx = int(item)
            if row_idx not in current:
                vals[0] = "✓"
                self.pop_tree.item(item, values=vals)
                self.population_selected_rows.add(row_idx)

    def _on_compare_run_clicked(self) -> None:
        if self.running_thread and self.running_thread.is_alive():
            return
        if self.population_df is None or self.population_df.empty:
            self._on_population_preview_clicked()
            if self.population_df is None or self.population_df.empty:
                messagebox.showwarning("Sin población", "Carga una población en la pestaña anterior antes de ejecutar la comparación.")
                return
        row_indices = self._selected_population_rows()
        if not row_indices:
            messagebox.showwarning("Sin selección", "Selecciona al menos un acorde en la lista.")
            return
        ids = self._selected_population_ids(row_indices)
        if not ids:
            messagebox.showwarning(
                "Población sin IDs",
                "La población seleccionada no contiene IDs válidos para ejecutar la comparación.\n"
                "Elige una fuente basada en la base de datos."
            )
            return
        # Compose SQL: single query carrying the full population
        id_list = ",".join(str(i) for i in ids)
        sql = f"SELECT * FROM chords WHERE id IN ({id_list}) ORDER BY id;"
        out_dir = Path(self.output_var.get().strip()).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = out_dir / f"compare_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Ruta esperada del reporte
        self.compare_last_report = None
        self.compare_expected_report = sub / "report.html"
        # Build process args
        self._append_log(f"\n--- Ejecutando comparación ---\n[población] {len(ids)} acordes seleccionados\n")
        self._append_compare_log(
            f"[comparación] Ejecutando con {len(ids)} acordes seleccionados "
            f"({len(row_indices)} filas en tabla). "
            f"Salida: {sub}\n"
        )
        props_sel = self._selected_proposals()
        metrics_sel = self._selected_metrics()
        reds_sel = self._selected_reductions()
        props_arg = ",".join(props_sel) if props_sel else "baseline_identity"
        metrics_arg = ",".join(metrics_sel) if metrics_sel else "euclidean"
        reductions_arg = ",".join(reds_sel) if reds_sel else "MDS"
        prop_titles = ", ".join(
            f"{PROPOSAL_INFO.get(name, {}).get('title', name)} ({name})" for name in props_sel
        ) if props_sel else "baseline_identity"
        metric_titles = ", ".join(
            f"{METRIC_INFO.get(name, {}).get('title', name)} ({name})" for name in metrics_sel
        ) if metrics_sel else "Euclidean"
        self._append_compare_log(
            f"Propuestas: {prop_titles} | Métricas: {metric_titles} | Reducciones: {reductions_arg} | Semillas: {self.compare_seeds_var.get().strip()}\n"
        )
        try:
            self.compare_open_button.configure(state=tk.DISABLED)
        except Exception:
            pass
        args = [
            sys.executable, "-m", "tools.compare_proposals",
            "--dyads-query", sql,
            "--triads-query", "",
            "--sevenths-query", "",
            "--proposals", props_arg,
            "--metrics", metrics_arg,
            "--reductions", reductions_arg,
            "--seeds", self.compare_seeds_var.get().strip(),
            "--output", str(sub),
        ]
        self._append_log("\n--- Ejecutando reporte de comparación ---\n")
        self.status_var.set("Ejecutando comparación…")
        self.compare_status_var.set("Ejecutando…")
        self._set_controls_state(tk.DISABLED)
        self.running_thread = threading.Thread(target=self._run_compare_thread, args=(args,), daemon=True)
        self.running_thread.start()

    def _run_compare_thread(self, proc_args: list[str]) -> None:
        report_re = re.compile(r"Reporte generado en:\s*(.*)")
        t0 = time.perf_counter()
        try:
            with subprocess.Popen(proc_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_queue.put(("compare_log", line))
                    m = report_re.search(line)
                    if m:
                        path = m.group(1).strip()
                        try:
                            self.compare_last_report = Path(path)
                        except Exception:
                            self.compare_last_report = None
            # Si no se detectó por regex, intentar la esperada
            try:
                expected = getattr(self, "compare_expected_report", None)
                if self.compare_last_report is None and expected and expected.exists():
                    self.compare_last_report = expected
            except Exception:
                pass
            t1 = time.perf_counter()
            self.log_queue.put(("compare_status", f"Comparación completada en {(t1 - t0):.2f}s."))
        except Exception as exc:  # pylint: disable=broad-except
            self.log_queue.put(("compare_error", str(exc)))
            self.log_queue.put(("error", str(exc)))
        finally:
            self.log_queue.put(("done", None))

    def _on_compare_open_clicked(self) -> None:
        if not self.compare_last_report:
            messagebox.showwarning("Nada que abrir", "Aún no hay reporte de comparación disponible.")
            return
        if not self.compare_last_report.exists():
            messagebox.showerror("Reporte no encontrado", f"No se encontró el archivo:\n{self.compare_last_report}")
            return
        try:
            webbrowser.open_new_tab(str(self.compare_last_report))
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("No se pudo abrir", str(exc))

    def _on_reduction_run_clicked(self) -> None:
        if self.running_thread and self.running_thread.is_alive():
            return
        if self.population_df is None or self.population_df.empty:
            self._on_population_preview_clicked()
            if self.population_df is None or self.population_df.empty:
                messagebox.showwarning("Sin población", "Carga una población en la pestaña anterior antes de ejecutar la comparación.")
                return
        row_indices = self._selected_population_rows()
        if not row_indices:
            messagebox.showwarning("Sin selección", "Selecciona al menos un acorde en la lista.")
            return
        ids = self._selected_population_ids(row_indices)
        if not ids:
            messagebox.showwarning(
                "Población sin IDs",
                "La población seleccionada no contiene IDs válidos para ejecutar la comparación.\n"
                "Elige una fuente basada en la base de datos."
            )
            return

        id_list = ",".join(str(i) for i in ids)
        sql = f"SELECT * FROM chords WHERE id IN ({id_list}) ORDER BY id;"
        out_dir = Path(self.output_var.get().strip()).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = out_dir / f"compare_reductions_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.reduction_last_report = None
        self.reduction_expected_report = sub / "report.html"

        proposal_display = self.reduction_compare_prop_var.get().strip()
        proposal_id = self.proposal_display_inverse.get(proposal_display, proposal_display.split('(')[-1].split(')')[0].strip())
        metrics_sel = [name for name, var in self.reduction_compare_metric_vars.items() if var.get()]
        reductions_sel = [name for name, var in self.reduction_compare_reduction_vars.items() if var.get()]
        metrics_arg = ",".join(metrics_sel) if metrics_sel else "euclidean"
        reductions_arg = ",".join(reductions_sel) if reductions_sel else "MDS"

        prop_log = self.proposal_display_map.get(proposal_id, proposal_id)
        metric_log = ", ".join(
            f"{METRIC_INFO.get(name, {}).get('title', name.title())} ({name})" for name in metrics_sel
        ) if metrics_sel else "Euclidiana"
        self._append_log(
            f"\n--- Comparación de reducciones ---\n[población] {len(ids)} acordes seleccionados\n"
        )
        self._append_tab_log(self.reduction_log, f"[reducciones] {len(ids)} acordes seleccionados. Salida: {sub}\n")
        self._append_tab_log(
            self.reduction_log,
            f"Propuesta: {prop_log} | Métricas: {metric_log} | Reducciones: {reductions_arg} | Semillas: {self.reduction_compare_seeds_var.get().strip()}\n"
        )
        try:
            self.reduction_open_button.configure(state=tk.DISABLED)
        except Exception:
            pass

        args = [
            sys.executable,
            "-m",
            "tools.compare_reductions",
            "--proposal",
            proposal_id,
            "--metrics",
            metrics_arg,
            "--reductions",
            reductions_arg,
            "--seeds",
            self.reduction_compare_seeds_var.get().strip(),
            "--dyads-query",
            sql,
            "--triads-query",
            "",
            "--sevenths-query",
            "",
            "--output",
            str(sub),
        ]
        self.reduction_status_var.set("Ejecutando…")
        self.status_var.set("Ejecutando comparación…")
        self._set_controls_state(tk.DISABLED)
        self.running_thread = threading.Thread(target=self._run_reduction_thread, args=(args,), daemon=True)
        self.running_thread.start()

    def _run_reduction_thread(self, proc_args: list[str]) -> None:
        report_re = re.compile(r"Reporte generado en:\s*(.*)")
        t0 = time.perf_counter()
        try:
            with subprocess.Popen(proc_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log_queue.put(("reduction_log", line))
                    m = report_re.search(line)
                    if m:
                        path = m.group(1).strip()
                        try:
                            self.reduction_last_report = Path(path)
                        except Exception:
                            self.reduction_last_report = None
            expected = getattr(self, "reduction_expected_report", None)
            if self.reduction_last_report is None and expected and expected.exists():
                self.reduction_last_report = expected
            t1 = time.perf_counter()
            self.log_queue.put(("reduction_status", f"Comparación completada en {(t1 - t0):.2f}s."))
        except Exception as exc:  # pylint: disable=broad-except
            self.log_queue.put(("reduction_error", str(exc)))
            self.log_queue.put(("error", str(exc)))
        finally:
            self.log_queue.put(("done", None))

    def _on_reduction_open_clicked(self) -> None:
        if not self.reduction_last_report:
            messagebox.showwarning("Nada que abrir", "Aún no hay reporte disponible.")
            return
        if not self.reduction_last_report.exists():
            messagebox.showerror("Reporte no encontrado", f"No se encontró el archivo:\n{self.reduction_last_report}")
            return
        try:
            webbrowser.open_new_tab(str(self.reduction_last_report))
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("No se pudo abrir", str(exc))

    # ---------------------------------------------------------------- actions
    def _default_output_dir(self) -> Path:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("outputs") / "gui_runs" / timestamp

    def _choose_output_dir(self) -> None:
        current = Path(self.output_var.get()).expanduser()
        initial = current if current.exists() else self._default_output_dir().parent
        selected = filedialog.askdirectory(initialdir=initial, title="Selecciona carpeta de salida")
        if selected:
            self.output_var.set(selected)

    def _add_population(self) -> None:
        query = self.pop_query_var.get().strip()
        pop_type = self.pop_type_var.get().strip()
        if not query:
            messagebox.showwarning("Datos incompletos", "Debes seleccionar una consulta para la población conjunta.")
            return
        spec = f"{pop_type}:{query}"
        if spec in self.pops_entries:
            messagebox.showinfo("Duplicado", "Esa población ya está en la lista.")
            return
        self.pops_entries.append(spec)
        self.pops_listbox.insert(tk.END, spec)
        self._mark_population_dirty()

    def _remove_selected_pop(self) -> None:
        selection = self.pops_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self.pops_listbox.delete(index)
        del self.pops_entries[index]
        self._mark_population_dirty()

    def _clear_pops(self) -> None:
        self.pops_entries.clear()
        self.pops_listbox.delete(0, tk.END)
        self._mark_population_dirty()

    def _save_custom_query(self) -> None:
        name = self.custom_name_var.get().strip()
        sql = self.custom_sql_text.get("1.0", tk.END).strip()
        if not name or not sql:
            messagebox.showwarning("Datos incompletos", "Introduce un nombre y el SQL de la consulta.")
            return
        try:
            normalized = add_custom_query(name, sql)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("No se pudo guardar", str(exc))
            return
        messagebox.showinfo("Consulta guardada", f"Consulta '{normalized}' almacenada correctamente.")
        self._refresh_queries(select=normalized)
        self._clear_custom_form()

    def _clear_custom_form(self) -> None:
        self.custom_name_var.set("")
        self.custom_sql_text.delete("1.0", tk.END)

    def _refresh_queries(self, select: str | None = None) -> None:
        current_select = select or self.base_query_var.get()
        self._load_queries()
        names = sorted(self.query_registry.keys())
        base_values = ["<Ninguna>"] + names
        if hasattr(self, "base_query_combo"):
            self.base_query_combo.configure(values=base_values)
        if current_select in base_values:
            self.base_query_var.set(current_select)
        else:
            self.base_query_var.set(base_values[1] if len(base_values) > 1 else "<Ninguna>")
        self._mark_population_dirty()

        self.pop_query_combo.configure(values=names or [""])
        if names:
            if current_select in names:
                self.pop_query_var.set(current_select)
            elif self.pop_query_var.get() not in names:
                self.pop_query_var.set(names[0])

    # ----------------------------------------------------------------- running
    def _on_run_clicked(self) -> None:
        if self.running_thread and self.running_thread.is_alive():
            return

        if (self.population_df is None or self.population_df.empty) and (self.pops_entries or (self.base_query_var.get().strip() and self.base_query_var.get().strip() != "<Ninguna>")):
            self._on_population_preview_clicked()

        df_override: pd.DataFrame | None = None
        descriptor: str | None = None
        selected_rows: list[int] = []
        selected_ids: list[int] = []
        if self.population_df is not None and not self.population_df.empty:
            selected_rows = self._selected_population_rows()
            if not selected_rows:
                selected_rows = list(range(len(self.population_df)))
            df_override = self.population_df.iloc[selected_rows].copy()
            selected_ids = self._selected_population_ids(selected_rows)
            if selected_ids:
                preview = ", ".join(str(i) for i in selected_ids[:10])
                ellipsis = ", ..." if len(selected_ids) > 10 else ""
                descriptor = f"SELECT * FROM chords WHERE id IN ({preview}{ellipsis})"
            else:
                descriptor = f"Selección manual ({len(selected_rows)} filas)"

        output_dir = self.output_var.get().strip()
        if not output_dir:
            output_dir = str(self._default_output_dir())
            self.output_var.set(output_dir)

        out_path = Path(output_dir)
        if out_path.exists():
            try:
                has_content = any(out_path.iterdir())
            except PermissionError:
                has_content = True
        else:
            has_content = False

        if has_content:
            answer = messagebox.askyesno(
                "Carpeta ya existe",
                f"La carpeta de salida:\n\n{out_path}\n\nya contiene archivos. "
                "¿Deseas continuar y sobrescribir los artefactos?",
                icon="warning"
            )
            if not answer:
                self._append_log("Ejecución cancelada por el usuario (carpeta existente).\n")
                self.status_var.set("Cancelado.")
                return

        if df_override is not None:
            args = argparse.Namespace(
                out=Path(output_dir),
                type=self.type_var.get(),
                query=descriptor or "<selección manual>",
                reduction=self.reduction_var.get(),
                pops=None,
                pops_csv=None,
                pops_file=None,
                model=self.model_label_to_value.get(self.model_var.get(), "Sethares"),
                metric=self.metric_label_to_value.get(self.metric_var.get(), "euclidean"),
                ponderation=self.ponder_label_to_value.get(self.ponder_var.get(), "ninguna"),
            )
            self._append_exp_log(
                f"[experimento] Población seleccionada: {len(selected_rows)} filas "
                f"({len(selected_ids)} IDs válidos)\n"
            )
        else:
            pops = list(self.pops_entries)
            base_query = self.base_query_var.get()
            if base_query == "<Ninguna>":
                base_query = None
            if not pops and not base_query:
                messagebox.showwarning("Parámetros insuficientes", "Selecciona una consulta base o agrega al menos una población conjunta, o usa la previsualización para seleccionar acordes.")
                return
            args = argparse.Namespace(
                out=Path(output_dir),
                type=self.type_var.get(),
                query=base_query,
                reduction=self.reduction_var.get(),
                pops=pops if pops else None,
                pops_csv=None,
                pops_file=None,
                model=self.model_label_to_value.get(self.model_var.get(), "Sethares"),
                metric=self.metric_label_to_value.get(self.metric_var.get(), "euclidean"),
                ponderation=self.ponder_label_to_value.get(self.ponder_var.get(), "ninguna"),
            )
            self._append_exp_log(
                f"[experimento] Población configurada vía pops/base (modo {args.type}).\n"
            )

        self._append_log("\n--- Ejecutando experimento ---\n")
        self._append_exp_log(
            f"[experimento] Modelo={args.model}, Métrica={args.metric}, "
            f"Reducción={args.reduction}, Ponderación={args.ponderation}\n"
        )
        self.status_var.set("Ejecutando…")
        self._set_controls_state(tk.DISABLED)

        self.running_thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(args, df_override, descriptor),
            daemon=True
        )
        self.running_thread.start()

    def _run_experiment_thread(self, args: argparse.Namespace, df_override: pd.DataFrame | None = None, descriptor: str | None = None) -> None:
        writer = QueueWriter(self.log_queue, category="exp_log")
        try:
            import contextlib

            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                print(f"[experimento] Iniciando… modelo={args.model}, metrica={args.metric}, reduccion={args.reduction}")
                result = experiment_inversions.run_experiment_with_args(
                    args,
                    df_override=df_override,
                    descriptor=descriptor,
                )
            self.log_queue.put(("exp_log", f"[experimento] Completado. Artefactos en: {result['output_dir']}\n"))
            self.log_queue.put(("exp_status", f"Experimento completado. Artefactos en: {result['output_dir']}"))
        except Exception as exc:  # pylint: disable=broad-except
            self.log_queue.put(("exp_error", str(exc)))
            self.log_queue.put(("error", str(exc)))
        finally:
            self.log_queue.put(("done", None))

    def _set_controls_state(self, state: str) -> None:
        names = [
            "run_button",
            "base_query_combo",   # puede no existir en el layout actual
            "pop_query_combo",
            "model_combo",
            "metric_combo",
            "ponder_combo",
            "compare_run_button",
            "reduction_run_button",
        ]
        for name in names:
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _append_tab_log(self, widget: ScrolledText, text: str) -> None:
        if widget is None:
            return
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _append_pop_log(self, text: str) -> None:
        self._append_tab_log(getattr(self, "pop_log", None), text)

    def _append_exp_log(self, text: str) -> None:
        self._append_tab_log(getattr(self, "exp_log", None), text)

    def _append_compare_log(self, text: str) -> None:
        self._append_tab_log(getattr(self, "compare_log", None), text)

    def _append_reduction_log(self, text: str) -> None:
        self._append_tab_log(getattr(self, "reduction_log", None), text)

    def _mark_population_dirty(self, *_args) -> None:
        self.population_df = None
        self.population_selected_rows.clear()
        self.population_row_ids.clear()
        if hasattr(self, "pop_tree"):
            for item in self.pop_tree.get_children():
                self.pop_tree.delete(item)
        if hasattr(self, "pop_stats_var"):
            self.pop_stats_var.set("—")

    def _selected_population_rows(self) -> list[int]:
        if self.population_df is None or self.population_df.empty:
            return []
        if not self.population_selected_rows:
            return []
        return sorted(self.population_selected_rows)

    def _selected_population_ids(self, rows: list[int] | None = None) -> list[int]:
        if self.population_df is None or self.population_df.empty:
            return []
        if rows is None:
            rows = self._selected_population_rows()
        ids: list[int] = []
        for row_idx in rows:
            chord_id = self.population_row_ids.get(row_idx)
            if chord_id is None:
                continue
            try:
                ids.append(int(chord_id))
            except Exception:
                continue
        return sorted(set(ids))

    def _selected_proposals(self) -> list[str]:
        return [name for name in self.proposals_order if self.proposal_vars[name].get()]

    def _selected_metrics(self) -> list[str]:
        return [name for name in self.metrics_order if self.metric_vars[name].get()]

    def _selected_reductions(self) -> list[str]:
        return [name for name in self.reductions_order if self.reduction_vars[name].get()]

    def _process_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(payload)
            elif kind == "exp_log":
                self._append_log(payload)
                self._append_exp_log(payload)
            elif kind == "compare_log":
                self._append_log(payload)
                self._append_compare_log(payload)
            elif kind == "reduction_log":
                self._append_log(payload)
                self._append_reduction_log(payload)
            elif kind == "exp_status":
                self._append_log(f"{payload}\n")
                self.status_var.set(payload)
                self._append_exp_log(f"{payload}\n")
            elif kind == "compare_status":
                self._append_log(f"{payload}\n")
                self.status_var.set(payload)
                self.compare_status_var.set(payload)
                self._append_compare_log(f"{payload}\n")
            elif kind == "reduction_status":
                self._append_log(f"{payload}\n")
                self.status_var.set(payload)
                self.reduction_status_var.set(payload)
                self._append_reduction_log(f"{payload}\n")
            elif kind == "exp_error":
                self._append_exp_log(f"Error: {payload}\n")
            elif kind == "compare_error":
                self._append_compare_log(f"Error: {payload}\n")
                self.compare_status_var.set("Error")
            elif kind == "reduction_error":
                self._append_reduction_log(f"Error: {payload}\n")
                self.reduction_status_var.set("Error")
            elif kind == "status":
                self._append_log(f"{payload}\n")
                self.status_var.set(payload)
            elif kind == "error":
                message = f"Error: {payload}"
                self._append_log(f"{message}\n")
                self.status_var.set("Error durante la ejecución.")
                messagebox.showerror("Error en el experimento", payload)
            elif kind == "done":
                self._set_controls_state(tk.NORMAL)
                # Como salvaguarda, habilitar abrir si hay reporte
                if getattr(self, "compare_last_report", None):
                    try:
                        if self.compare_last_report.exists():
                            self.compare_open_button.configure(state=tk.NORMAL)
                            if self.compare_status_var.get() == "Ejecutando…":
                                self.compare_status_var.set("Listo.")
                        else:
                            self.compare_open_button.configure(state=tk.DISABLED)
                            if self.compare_status_var.get() != "Error":
                                self.compare_status_var.set("Reporte no encontrado")
                    except Exception:
                        pass
                if getattr(self, "reduction_last_report", None):
                    try:
                        if self.reduction_last_report.exists():
                            self.reduction_open_button.configure(state=tk.NORMAL)
                            if self.reduction_status_var.get() == "Ejecutando…":
                                self.reduction_status_var.set("Listo.")
                        else:
                            self.reduction_open_button.configure(state=tk.DISABLED)
                            if self.reduction_status_var.get() != "Error":
                                self.reduction_status_var.set("Reporte no encontrado")
                    except Exception:
                        pass
        self.after(120, self._process_log_queue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-queries", action="store_true", help="Muestra las consultas disponibles y termina.")
    args = parser.parse_args()
    if args.list_queries:
        queries = get_all_queries()
        for name, meta in queries.items():
            print(f"{name} [{meta['source']}]")
        return

    app = ExperimentLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
