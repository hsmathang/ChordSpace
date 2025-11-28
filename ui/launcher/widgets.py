"""
Reusable widgets for the Experiment Launcher GUI.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class ScrollableFrame(ttk.Frame):
    """Wrapper that provides a vertical scrollbar for large tab content."""

    def __init__(self, master: tk.Misc, *, padding: int | tuple[int, int, int, int] = 0) -> None:
        super().__init__(master)
        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._vscroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)
        self._vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.configure(yscrollcommand=self._vscroll.set)

        self.content = ttk.Frame(self._canvas, padding=padding)
        self._window = self._canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_content_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_content_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        try:
            # Check if mouse is over this widget
            # Note: This check is a bit fragile in Tkinter but works for most cases
            widget_path = self.tk.call(
                "winfo",
                "containing",
                self.winfo_pointerx(),
                self.winfo_pointery(),
            )
        except tk.TclError:
            return
        if not widget_path:
            return
        # Don't scroll if inside a dropdown
        if "popdown" in str(widget_path):
            return
        try:
            pointer_widget = self.nametowidget(widget_path)
        except KeyError:
            return
        if pointer_widget is None:
            return
        if not self._is_descendant(pointer_widget):
            return

        delta = 0
        if getattr(event, "delta", 0):
            delta = int(-event.delta / 120)
        elif getattr(event, "num", None) in {4, 5}:
            delta = 1 if event.num == 5 else -1

        if delta:
            self._canvas.yview_scroll(delta, "units")

    def _is_descendant(self, widget: tk.Misc) -> bool:
        current = widget
        while current is not None:
            if current is self:
                return True
            try:
                current = current.master
            except Exception:
                return False
        return False
