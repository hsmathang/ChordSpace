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
from tools.query_registry import add_custom_query, get_all_queries


class QueueWriter(io.TextIOBase):
    """Redirects stdout/stderr writes into a queue for the GUI to consume."""

    def __init__(self, bucket: queue.Queue):
        super().__init__()
        self.bucket = bucket

    def write(self, msg: str) -> int:
        if msg:
            self.bucket.put(("log", msg))
        return len(msg)

    def flush(self) -> None:  # pragma: no cover - nothing to flush
        return None


class ExperimentLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ChordSpace - Experiment Launcher")
        self.geometry("980x720")
        self.minsize(860, 640)

        self.log_queue: queue.Queue = queue.Queue()
        self.running_thread: threading.Thread | None = None
        self.pops_entries: list[str] = []

        self._load_queries()
        self._init_state_vars()
        self._create_layout()

        self.after(100, self._process_log_queue)

    # ------------------------------------------------------------------ setup
    def _load_queries(self) -> None:
        self.query_registry = get_all_queries()

    def _init_state_vars(self) -> None:
        self.type_var = tk.StringVar(value="B")
        self.reduction_var = tk.StringVar(value="MDS")
        options = ["<Ninguna>"] + sorted(self.query_registry.keys())
        default_query = "QUERY_CHORDS_WITH_NAME" if "QUERY_CHORDS_WITH_NAME" in self.query_registry else options[1] if len(options) > 1 else "<Ninguna>"
        self.base_query_var = tk.StringVar(value=default_query if default_query in options else "<Ninguna>")
        self.output_var = tk.StringVar(value=str(self._default_output_dir()))

        self.pop_type_var = tk.StringVar(value="A")
        pop_default_query = default_query if default_query != "<Ninguna>" else (options[1] if len(options) > 1 else "<Ninguna>")
        self.pop_query_var = tk.StringVar(value=pop_default_query)

        self.custom_name_var = tk.StringVar()

    def _create_layout(self) -> None:
        main = ttk.Frame(self, padding=14)
        main.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.LabelFrame(main, text="Configuración general")
        config_frame.pack(fill=tk.X, pady=(0, 8))
        self._build_config_frame(config_frame)

        middle_frame = ttk.Frame(main)
        middle_frame.pack(fill=tk.BOTH, expand=True)

        pops_frame = ttk.LabelFrame(middle_frame, text="Poblaciones conjuntas")
        pops_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self._build_pops_frame(pops_frame)

        custom_frame = ttk.LabelFrame(middle_frame, text="Registrar nueva consulta SQL")
        custom_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self._build_custom_query_frame(custom_frame)

        log_frame = ttk.LabelFrame(main, text="Registro de ejecución")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_widget = ScrolledText(log_frame, height=12, state=tk.DISABLED, font=("Consolas", 10))
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        button_bar = ttk.Frame(main)
        button_bar.pack(fill=tk.X, pady=(10, 0))
        self.run_button = ttk.Button(button_bar, text="Ejecutar experimento", command=self._on_run_clicked)
        self.run_button.pack(side=tk.LEFT)
        ttk.Button(button_bar, text="Cerrar", command=self.destroy).pack(side=tk.RIGHT)

        self.status_var = tk.StringVar(value="Listo.")
        status_label = ttk.Label(main, textvariable=self.status_var, foreground="#555", anchor="w")
        status_label.pack(fill=tk.X, pady=(6, 0))

    # --------------------------- sub frames
    def _build_config_frame(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Tipo de población base:").grid(row=0, column=0, sticky="w")
        type_combo = ttk.Combobox(frame, textvariable=self.type_var, values=["A", "B", "C"], width=6, state="readonly")
        type_combo.grid(row=0, column=1, padx=(4, 16), sticky="w")

        ttk.Label(frame, text="Reducción dimensional:").grid(row=0, column=2, sticky="w")
        red_combo = ttk.Combobox(frame, textvariable=self.reduction_var, values=["MDS", "UMAP"], width=8, state="readonly")
        red_combo.grid(row=0, column=3, padx=(4, 16), sticky="w")

        ttk.Label(frame, text="Consulta base:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.base_query_combo = ttk.Combobox(frame, textvariable=self.base_query_var,
                                             values=["<Ninguna>"] + sorted(self.query_registry.keys()),
                                             state="readonly", width=42)
        self.base_query_combo.grid(row=1, column=1, columnspan=3, sticky="we", padx=(4, 0), pady=(6, 0))

        ttk.Label(frame, text="Carpeta de salida:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        out_entry = ttk.Entry(frame, textvariable=self.output_var, width=48)
        out_entry.grid(row=2, column=1, columnspan=2, sticky="we", padx=(4, 4), pady=(8, 0))
        ttk.Button(frame, text="Examinar…", command=self._choose_output_dir).grid(row=2, column=3, sticky="e", pady=(8, 0))

        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

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
        self.pops_listbox = tk.Listbox(list_frame, height=8, selectmode=tk.SINGLE)
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
        self.custom_sql_text = tk.Text(frame, height=12, font=("Consolas", 10))
        self.custom_sql_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        button_row = ttk.Frame(frame)
        button_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(button_row, text="Guardar consulta", command=self._save_custom_query).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Limpiar", command=self._clear_custom_form).pack(side=tk.RIGHT)

        hint = ttk.Label(frame, text="Las consultas nuevas se guardan en tools/custom_queries.json.", foreground="#777")
        hint.pack(anchor="w", padx=6, pady=(0, 6))

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

    def _remove_selected_pop(self) -> None:
        selection = self.pops_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        self.pops_listbox.delete(index)
        del self.pops_entries[index]

    def _clear_pops(self) -> None:
        self.pops_entries.clear()
        self.pops_listbox.delete(0, tk.END)

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
        self.base_query_combo.configure(values=base_values)
        if current_select in base_values:
            self.base_query_var.set(current_select)
        else:
            self.base_query_var.set(base_values[1] if len(base_values) > 1 else "<Ninguna>")

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

        pops = list(self.pops_entries)
        base_query = self.base_query_var.get()
        if base_query == "<Ninguna>":
            base_query = None

        if not pops and not base_query:
            messagebox.showwarning("Parámetros insuficientes", "Selecciona una consulta base o agrega al menos una población conjunta.")
            return

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

        args = argparse.Namespace(
            out=Path(output_dir),
            type=self.type_var.get(),
            query=base_query,
            reduction=self.reduction_var.get(),
            pops=pops if pops else None,
            pops_csv=None,
            pops_file=None,
        )

        self._append_log("\n--- Ejecutando experimento ---\n")
        self.status_var.set("Ejecutando…")
        self._set_controls_state(tk.DISABLED)

        self.running_thread = threading.Thread(target=self._run_experiment_thread, args=(args,), daemon=True)
        self.running_thread.start()

    def _run_experiment_thread(self, args: argparse.Namespace) -> None:
        writer = QueueWriter(self.log_queue)
        try:
            import contextlib

            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = experiment_inversions.run_experiment_with_args(args)
            self.log_queue.put(("status", f"Experimento completado. Artefactos en: {result['output_dir']}"))
        except Exception as exc:  # pylint: disable=broad-except
            self.log_queue.put(("error", str(exc)))
        finally:
            self.log_queue.put(("done", None))

    def _set_controls_state(self, state: str) -> None:
        for widget in (self.run_button, self.base_query_combo, self.pop_query_combo):
            widget.configure(state=state)

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text)
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _process_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(payload)
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
