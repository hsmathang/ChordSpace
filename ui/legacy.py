"""
ui.py
-----
Este módulo define la interfaz de usuario interactiva para el laboratorio de acordes.
Se crean widgets, se configuran callbacks y se organiza la UI en contenedores para facilitar
la ejecución de experimentos y la visualización de resultados (incluyendo la gestión de favoritos).

La función principal 'create_ui' recibe la instancia del laboratorio y la función de 
ejecución del experimento, y retorna la UI completa para ser desplegada.
"""

from ipywidgets import Tab, Output, VBox, HBox, IntSlider, Button, Dropdown, interactive_output
import plotly.io as pio
pio.renderers.default = "colab"  # Configuración para Colab

# Importar listas de opciones y constantes desde config.py
from config import MODELO_OPTIONS_LIST, PONDERACION_OPTIONS_LIST, METRICA_OPTIONS_LIST, REDUCCION_OPTIONS_LIST

# Importar funciones/clases necesarias
from lab import LaboratorioAcordes, ResultadoExperimento
from visualization import (
    visualizar_scatter_density,
    visualizar_heatmap,
    graficar_shepard,
    explore_umap_graph,
    visualizar_laplacian_scatter,
    visualizar_sp_sf,
    VisualizadorAcordesModular
)
from pre_process import ChordAdapter
from audio import ChordPlayer
import itertools
from itertools import combinations 


# ui.py

import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import ast
from reduction import laplacian_eigenmaps_embedding
from config import (
    MODELO_OPTIONS_LIST, PONDERACION_OPTIONS_LIST,
    METRICA_OPTIONS_LIST, REDUCCION_OPTIONS_LIST,
    PONDERACION_CONSONANCIA_DEFAULT_WEIGHTS
)
from pre_process import (
    ModeloSethares, ModeloSetharesVec, ModeloEuler, ModeloArmonicosCriticos,
    PonderacionConsonancia, PonderacionImportanciaPerceptual,
    PonderacionCombinada
)

class UI:
    """Clase que encapsula la interfaz de usuario interactiva para visualización de acordes."""
    
    def __init__(self, laboratorio_total):
        """
        Inicializa la interfaz de usuario con los datos de laboratorio_total.
        
        Args:
            laboratorio_total: Objeto que contiene la lista de acordes y sus propiedades.
        """
        self.laboratorio_total = laboratorio_total
        self.ultimo_resultado = None
        self.favoritos = []
        self.vis_mod = VisualizadorAcordesModular()
        self.setup_ui()

    def setup_ui(self):
        """Configura todos los componentes de la interfaz de usuario."""
        # Dropdowns para selección de opciones
        self.modelo_options = widgets.Dropdown(
            options=MODELO_OPTIONS_LIST,
            description="Modelo de Rugosidad:",
            style={'description_width': 'initial'}
        )
        self.ponderacion_options = widgets.Dropdown(
            options=PONDERACION_OPTIONS_LIST,
            description="Ponderación:",
            style={'description_width': 'initial'}
        )
        self.metrica_options = widgets.Dropdown(
            options=METRICA_OPTIONS_LIST,
            description="Métrica de Distancia:",
            style={'description_width': 'initial'}
        )
        self.reduccion_options = widgets.Dropdown(
            options=REDUCCION_OPTIONS_LIST,
            description="Reducción Dimensional:",
            style={'description_width': 'initial'}
        )

        # Outputs para cada pestaña
        self.out_reporte = widgets.Output()
        self.out_scatter = widgets.Output()
        self.out_heatmap = widgets.Output()
        self.out_shepard = widgets.Output()
        self.out_grafo = widgets.Output()
        self.out_laplacian = widgets.Output()
        self.out_sp_sf = widgets.Output()
        self.out_favoritos = widgets.Output()

        # Tab con 8 pestañas
        self.tab = widgets.Tab(children=[
            self.out_reporte, self.out_scatter, self.out_heatmap, self.out_shepard,
            self.out_grafo, self.out_laplacian, self.out_sp_sf, self.out_favoritos
        ])
        titles = ["Reporte", "Scatter", "Heatmap", "Shepard", "Grafo UMAP", "Laplacian", "Sp–Sf", "Favoritos"]
        for i, title in enumerate(titles):
            self.tab.set_title(i, title)

        # Slider y botones para UMAP y Laplacian
        self.umap_slider = widgets.IntSlider(
            value=3, min=3, max=15, step=1, description='n_neighbors:'
        )
        self.umap_button = widgets.Button(description='Actualizar Grafo UMAP', button_style='info')
        self.umap_button.on_click(self.on_umap_button_clicked)

        self.laplacian_button = widgets.Button(description='Actualizar Laplacian', button_style='info')
        self.laplacian_button.on_click(self.on_laplacian_button_clicked)

        # Controles de favoritos
        self.dropdown_favoritos = widgets.Dropdown(
            options=[(ac.name, i) for i, ac in enumerate(self.laboratorio_total.acordes)],
            description="Acorde:",
            style={'description_width': 'initial'},
            layout={'width': '300px'}
        )
        self.btn_agregar_cercanos = widgets.Button(description="Agregar Vecinos Cercanos", button_style="success")
        self.btn_agregar_lejanos = widgets.Button(description="Agregar Vecinos Lejanos", button_style="warning")
        self.btn_reproducir_favoritos = widgets.Button(description="Reproducir Favoritos", button_style="info")
        self.btn_borrar_favoritos = widgets.Button(description="Borrar Favoritos", button_style="danger")
        self.out_favoritos_tab = widgets.Output()

        self.btn_agregar_cercanos.on_click(self.on_agregar_cercanos_clicked)
        self.btn_agregar_lejanos.on_click(self.on_agregar_lejanos_clicked)
        self.btn_reproducir_favoritos.on_click(self.on_reproducir_favoritos_clicked)
        self.btn_borrar_favoritos.on_click(self.on_borrar_favoritos_clicked)

        # Interactividad
        self.ui_interactivo = widgets.interactive_output(self.ejecutar_experimento_interactivo, {
            'modelo_tipo': self.modelo_options,
            'ponderacion_tipo': self.ponderacion_options,
            'metrica': self.metrica_options,
            'reduccion': self.reduccion_options
        })

        # Organizar la interfaz
        ui_controls = widgets.VBox([
            widgets.HBox([self.modelo_options, self.ponderacion_options]),
            widgets.HBox([self.metrica_options, self.reduccion_options])
        ])
        ui_buttons = widgets.VBox([self.umap_slider, widgets.HBox([self.umap_button, self.laplacian_button])])
        self.ui = widgets.VBox([ui_controls, self.ui_interactivo, self.tab, ui_buttons])

        # Mostrar controles de favoritos inicialmente
        with self.out_favoritos:
            clear_output()
            display(widgets.VBox([
                self.dropdown_favoritos,
                widgets.HBox([
                    self.btn_agregar_cercanos, self.btn_agregar_lejanos,
                    self.btn_borrar_favoritos, self.btn_reproducir_favoritos
                ]),
                self.out_favoritos_tab
            ]))
            self.mostrar_favoritos()

    def on_umap_button_clicked(self, b):
        """Callback para actualizar el grafo UMAP."""
        self.out_grafo.clear_output()
        with self.out_grafo:
            if (self.ultimo_resultado is None or
                not hasattr(self.ultimo_resultado, 'reducer_obj') or
                self.ultimo_resultado.X_original is None):
                print("No hay resultados UMAP disponibles. Ejecuta un experimento primero.")
            else:
                self.vis_mod.grafo_umap(
                    self.ultimo_resultado.reducer_obj,
                    self.ultimo_resultado.X_original,
                    self.laboratorio_total.acordes,
                    self.umap_slider.value
                )

    def on_laplacian_button_clicked(self, b):
        """Callback para actualizar el embedding Laplacian."""
        self.out_laplacian.clear_output()
        with self.out_laplacian:
            if (self.ultimo_resultado is None or not hasattr(self.ultimo_resultado, 'reducer_obj')):
                print("No hay resultados disponibles para Laplacian. Ejecuta un experimento primero.")
            else:
                embedding_laplacian = laplacian_eigenmaps_embedding(self.ultimo_resultado.reducer_obj)
                if embedding_laplacian is None:
                    print("No se pudo calcular el embebido Laplacian.")
                else:
                    self.vis_mod.laplacian(
                        embedding_laplacian,
                        self.laboratorio_total.acordes,
                        self.ultimo_resultado.X_original
                    )

    def ejecutar_experimento_interactivo(self, modelo_tipo, ponderacion_tipo, metrica, reduccion):
        """Ejecuta el experimento interactivo y actualiza las visualizaciones."""
        # Selección del modelo
        if modelo_tipo == "Sethares":
            modelo_elegido = ModeloSethares(config={})
        elif modelo_tipo == "Euler":
            modelo_elegido = ModeloEuler(config={})
        elif modelo_tipo in ["Armónicos Criticos", "ArmonicosCriticos"]:
            modelo_elegido = ModeloArmonicosCriticos(config={})
        else:
            modelo_elegido = ModeloSethares(config={})

        # Selección de la ponderación
        if ponderacion_tipo == "consonancia":
            ponderacion_elegida = PonderacionConsonancia()
        elif ponderacion_tipo == "importancia":
            ponderacion_elegida = PonderacionImportanciaPerceptual()
        elif ponderacion_tipo == "combinada":
            ponderacion_elegida = PonderacionCombinada(PonderacionConsonancia(), PonderacionImportanciaPerceptual())
        else:
            ponderacion_elegida = None

        # Ejecutar el experimento
        resultado = self.laboratorio_total.ejecutar_experimento(modelo_elegido, ponderacion_elegida, metrica=metrica, reduccion=reduccion)
        self.ultimo_resultado = resultado

        # Actualizar visualizaciones
        with self.out_reporte:
            clear_output()
            self.vis_mod.reporte(resultado, resultado.X_original)
        with self.out_scatter:
            clear_output()
            self.vis_mod.scatter(resultado.embeddings, self.laboratorio_total.acordes, resultado.X_original)
        with self.out_heatmap:
            clear_output()
            self.vis_mod.heatmap(resultado.matriz_distancias, self.laboratorio_total.acordes)
        with self.out_shepard:
            clear_output()
            self.vis_mod.shepard(resultado.embeddings, resultado.matriz_distancias)
        with self.out_sp_sf:
            clear_output()
            self.vis_mod.sp_sf(self.laboratorio_total.acordes)

        # Actualizar pestaña de favoritos
        with self.out_favoritos:
            clear_output()
            display(widgets.VBox([
                self.dropdown_favoritos,
                widgets.HBox([self.btn_agregar_cercanos, self.btn_agregar_lejanos, self.btn_borrar_favoritos, self.btn_reproducir_favoritos]),
                self.out_favoritos_tab
            ]))
            self.mostrar_favoritos()
        # Ejemplo: with self.out_reporte: self.vis_mod.reporte(...)

    def find_neighbors(self, index, dist_matrix, k=3, far=False):
        """
        Encuentra los k vecinos más cercanos o lejanos de un acorde.
        
        Args:
            index: Índice del acorde en la matriz de distancias.
            dist_matrix: Matriz de distancias.
            k: Número de vecinos a devolver.
            far: Si True, devuelve los más lejanos; si False, los más cercanos.
        
        Returns:
            Lista de índices de los k vecinos.
        """
        row = dist_matrix[index]
        sorted_indices = np.argsort(-row) if far else np.argsort(row)
        sorted_indices = sorted_indices[sorted_indices != index]
        return sorted_indices[:k].tolist()

    def on_agregar_cercanos_clicked(self, b):
        """Callback para agregar vecinos cercanos a favoritos."""
        if self.ultimo_resultado is None or self.ultimo_resultado.matriz_distancias is None:
            with self.out_favoritos_tab:
                clear_output()
                print("Ejecuta un experimento primero para tener la matriz de distancias.")
            return
        index_sel = self.dropdown_favoritos.value
        dist_mat = self.ultimo_resultado.matriz_distancias
        vecinos_cercanos = self.find_neighbors(index_sel, dist_mat, k=3, far=False)
        existente = next((fav for fav in self.favoritos if fav[0] == index_sel), None)
        if existente:
            existente[1][:] = vecinos_cercanos
        else:
            self.favoritos.append([index_sel, vecinos_cercanos, []])
        with self.out_favoritos_tab:
            clear_output()
            print(f"Se agregó el acorde '{self.laboratorio_total.acordes[index_sel].name}' con vecinos CERCANOS: "
                  f"{[self.laboratorio_total.acordes[v].name for v in vecinos_cercanos]}")
            self.mostrar_favoritos()

    def on_agregar_lejanos_clicked(self, b):
        """Callback para agregar vecinos lejanos a favoritos."""
        if self.ultimo_resultado is None or self.ultimo_resultado.matriz_distancias is None:
            with self.out_favoritos_tab:
                clear_output()
                print("Ejecuta un experimento primero para tener la matriz de distancias.")
            return
        index_sel = self.dropdown_favoritos.value
        dist_mat = self.ultimo_resultado.matriz_distancias
        vecinos_lejanos = self.find_neighbors(index_sel, dist_mat, k=3, far=True)
        existente = next((fav for fav in self.favoritos if fav[0] == index_sel), None)
        if existente:
            existente[2][:] = vecinos_lejanos
        else:
            self.favoritos.append([index_sel, [], vecinos_lejanos])
        with self.out_favoritos_tab:
            clear_output()
            print(f"Se agregó el acorde '{self.laboratorio_total.acordes[index_sel].name}' con vecinos LEJANOS: "
                  f"{[self.laboratorio_total.acordes[v].name for v in vecinos_lejanos]}")
            self.mostrar_favoritos()

    def on_borrar_favoritos_clicked(self, b):
        """Callback para borrar todos los favoritos."""
        self.favoritos.clear()
        with self.out_favoritos_tab:
            clear_output()
            print("Se han borrado todos los favoritos.")

    def mostrar_favoritos(self):
        """Muestra la lista actual de favoritos."""
        if not self.favoritos:
            print("No hay favoritos aún.")
            return
        print("=== Lista de Favoritos ===")
        for i, (principal_idx, cercanos, lejanos) in enumerate(self.favoritos, start=1):
            principal_name = self.laboratorio_total.acordes[principal_idx].name
            cercanos_nombres = [self.laboratorio_total.acordes[v].name for v in cercanos] if cercanos else []
            lejanos_nombres = [self.laboratorio_total.acordes[v].name for v in lejanos] if lejanos else []
            print(f"{i}) Acorde Principal: {principal_name}")
            print(f"    Vecinos Cercanos: {cercanos_nombres}")
            print(f"    Vecinos Lejanos: {lejanos_nombres}")

    def reproducir_acorde(self, acorde, player):
        """
        Reproduce un acorde usando ChordPlayer.
        
        Args:
            acorde: Objeto acorde con atributo frequencies.
            player: Instancia de ChordPlayer.
        
        Returns:
            Objeto de audio o None si hay error.
        """
        freqs = acorde.frequencies
        if isinstance(freqs, str):
            try:
                freqs = ast.literal_eval(freqs)
            except Exception as e:
                print(f"Error al parsear frecuencias de {acorde.name}: {e}")
                return None
        return player.play_sequence(freqs, individual_duration=0.4, chord_duration=1.5, delay=0.1)

    def on_reproducir_favoritos_clicked(self, b):
        """Callback para reproducir todos los favoritos."""
        if not self.favoritos:
            with self.out_favoritos_tab:
                clear_output()
                print("No hay favoritos para reproducir.")
            return
        player = ChordPlayer()
        with self.out_favoritos_tab:
            clear_output()
            print("Reproduciendo Favoritos (acorde principal + vecinos cercanos y lejanos) en secuencia...\n")
            for i, (principal_idx, cercanos, lejanos) in enumerate(self.favoritos, start=1):
                acorde_principal = self.laboratorio_total.acordes[principal_idx]
                print(f"{i}) Acorde Principal: {acorde_principal.name}")
                audio_obj = self.reproducir_acorde(acorde_principal, player)
                display(audio_obj)
                if cercanos:
                    print("   Vecinos Cercanos:")
                    for v in cercanos:
                        ac_vec = self.laboratorio_total.acordes[v]
                        print(f"      {ac_vec.name}")
                        display(self.reproducir_acorde(ac_vec, player))
                if lejanos:
                    print("   Vecinos Lejanos:")
                    for v in lejanos:
                        ac_vec = self.laboratorio_total.acordes[v]
                        print(f"      {ac_vec.name}")
                        display(self.reproducir_acorde(ac_vec, player))
                print("")

    def display_ui(self):
        """Muestra la interfaz de usuario completa."""
        display(self.ui)
