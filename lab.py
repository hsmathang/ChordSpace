"""
lab.py
------
Este módulo contiene la clase LaboratorioAcordes, que gestiona y ejecuta experimentos
de modelado de rugosidad utilizando los datos preprocesados, la reducción dimensional y 
la evaluación de métricas. Su responsabilidad es orquestar el flujo completo del experimento:

  1. Recibir una lista de objetos Acorde (definidos en pre_process.py).
  2. Aplicar un modelo de rugosidad (por ejemplo, ModeloSethares, ModeloEuler, ModeloArmonicosCriticos).
  3. Aplicar, si se requiere, una ponderación al vector de rugosidad.
  4. Calcular la matriz de distancias entre los vectores resultantes.
  5. Realizar la reducción dimensional utilizando diferentes métodos (MDS, UMAP, Kernel MDS, Geometric MDS o RepresentacionSpSf).
  6. Calcular y reportar métricas de evaluación (por ejemplo, stress, trustworthiness, continuity, etc.).
  7. Almacenar los resultados en un objeto ResultadoExperimento.

Inventario Detallado:

• kruskal_stress_1(dist_original, dist_embedded) -> float  
   - **Propósito:** Calcula el Stress-1 (Kruskal) normalizado a partir de dos matrices de distancias.
   - **Uso:** Se utiliza para evaluar la calidad del embedding obtenido mediante reducción dimensional.
   - **Dependencias:** numpy.

• Clase LaboratorioAcordes  
   - **Propósito:**  
     Gestiona la lista de acordes y ejecuta el experimento completo aplicando modelos de rugosidad,
     ponderaciones y reducción dimensional, y finalmente calculando métricas de evaluación.
   - **Método ejecutar_experimento(modelo, ponderacion, metrica, reduccion) -> ResultadoExperimento:**  
     - Orquesta el flujo completo:
       1. Itera sobre los acordes y calcula, para cada uno, el vector de rugosidad (y la rugosidad total).
       2. Aplica, opcionalmente, una ponderación.
       3. Conforma una matriz con los vectores procesados.
       4. Calcula la matriz de distancias usando la métrica indicada.
       5. Realiza la reducción dimensional según el método especificado:
            - MDS: usa sklearn.manifold.MDS.
            - UMAP: usa la biblioteca umap.
            - Kernel MDS y Geometric MDS: se importan de reduction.py.
            - RepresentacionSpSf: usa la función representacion_sp_sf de pre_process.py.
       6. Calcula métricas adicionales (trustworthiness, knn recall, continuity) si es aplicable.
       7. Registra y retorna un objeto ResultadoExperimento (definido en pre_process.py).
   - **Dependencias:**  
     - Tiempo (time), numpy, funciones de distancia (pdist, squareform) de scipy.
     - MDS de sklearn.manifold y umap.
     - Funciones y clases de pre_process.py (por ejemplo, Acorde, ResultadoExperimento, modelos de rugosidad, etc.).
     - Funciones de reducción dimensional: kernel_mds, geometric_mds_vectorized, calculate_stress_vectorized, laplacian_eigenmaps_embedding (importadas desde reduction.py).
     - Funciones de evaluación: compute_trustworthiness, compute_knn_recall, compute_continuity, reporte_parametros_reduccion (importadas desde metrics.py).

Orden de Importación Sugerido:
  1. config.py
  2. pre_process.py
  3. reduction.py
  4. metrics.py
  5. lab.py (este módulo)
  6. ui.py

"""

import time
import numpy as np
from typing import List, Dict, Optional

from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
try:
    import umap  # type: ignore  # Para UMAP
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    umap = None

# Importar funciones y clases desde otros módulos:
# De pre_process.py: Acorde, ResultadoExperimento, y (si se requieren) funciones de preprocesamiento.
from pre_process import Acorde, ResultadoExperimento
# Se asume que los modelos y ponderaciones ya se usan en pre_process.py y se pasan como argumentos.

# Importar funciones de reducción dimensional desde reduction.py
from reduction import kernel_mds, geometric_mds_vectorized, calculate_stress_vectorized, laplacian_eigenmaps_embedding

# Importar funciones de métricas de evaluación desde metrics.py
from metrics import (
    compute_trustworthiness,
    compute_knn_recall,
    compute_continuity,
    reporte_parametros_reduccion
)

# Nota: Las funciones compute_timbre_distance, compute_combined_features y gower_distance_matrix se deben definir
# o importar si son necesarias. Aquí se asume que están disponibles en pre_process.py o en otro módulo.
# Por ejemplo, se podría hacer:
# from pre_process import compute_timbre_distance, compute_combined_features, gower_distance_matrix

def kruskal_stress_1(dist_original: np.ndarray, dist_embedded: np.ndarray) -> float:
    """
    Calcula el Stress-1 (Kruskal) normalizado.
    
    Args:
        dist_original (np.ndarray): Matriz de distancias original (NxN).
        dist_embedded (np.ndarray): Matriz de distancias en el embedding (NxN).
    
    Returns:
        float: Valor del stress normalizado.
    
    Uso:
      - Se utiliza en el método ejecutar_experimento para evaluar la calidad del embedding.
    """
    n = dist_original.shape[0]
    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            numerator += (dist_embedded[i, j] - dist_original[i, j]) ** 2
            denominator += dist_original[i, j] ** 2
    return np.sqrt(numerator / denominator)

class LaboratorioAcordes:
    """
    Gestiona y ejecuta experimentos de modelado de rugosidad sobre una lista de acordes.
    
    Funcionalidades:
      - Itera sobre los acordes para calcular sus vectores de rugosidad utilizando un modelo dado.
      - Aplica, opcionalmente, una ponderación sobre dichos vectores.
      - Calcula la matriz de distancias entre los vectores obtenidos.
      - Realiza la reducción dimensional según el método especificado:
            * MDS: mediante sklearn.manifold.MDS.
            * UMAP: mediante la biblioteca umap.
            * Kernel MDS: usando kernel_mds de reduction.py.
            * Geometric MDS: usando geometric_mds_vectorized de reduction.py.
            * RepresentacionSpSf: usando la función representacion_sp_sf (definida en pre_process.py).
      - Calcula métricas de evaluación (por ejemplo, trustworthiness, continuity, KNN recall, rank correlation).
      - Almacena el resultado en un objeto ResultadoExperimento.
    
    Dependencias:
      - Importa funciones y clases de pre_process.py, reduction.py y metrics.py.
      - Usa MDS y umap para la reducción dimensional.
    """
    def __init__(self, acordes: List[Acorde]):
        self.acordes = acordes
        self.resultados: Dict[str, ResultadoExperimento] = {}

    def ejecutar_experimento(self, modelo, ponderacion, metrica: str = 'cosine', reduccion: str = 'MDS') -> ResultadoExperimento:
        """
        Ejecuta un experimento de modelado de rugosidad.
        
        Parámetros:
          - modelo: Objeto de un modelo de rugosidad (ej. ModeloSethares, ModeloEuler, ModeloArmonicosCriticos).
          - ponderacion: Objeto de ponderación (ej. PonderacionConsonancia, PonderacionImportanciaPerceptual, o combinada).
          - metrica (str): Métrica de distancia a utilizar (ej. 'cosine', 'euclidean', etc.).
          - reduccion (str): Método de reducción dimensional (ej. 'MDS', 'UMAP', 'Kernel MDS', 'Geometric MDS', 'RepresentacionSpSf').
        
        Flujo:
          1. Calcula para cada acorde el vector de rugosidad y su valor total.
          2. Aplica la ponderación si se especifica.
          3. Forma la matriz de vectores y calcula la matriz de distancias.
          4. Realiza la reducción dimensional según el método especificado.
          5. Calcula métricas de evaluación (si es aplicable) y reporta los parámetros del método.
          6. Crea y retorna un objeto ResultadoExperimento con todos los datos.
        
        Retorna:
          - ResultadoExperimento: Objeto que encapsula el nombre del modelo, métricas, embeddings, matriz de distancias, tiempo de ejecución, objeto reductor y los vectores originales.
        """
        start_time = time.time()
        print(f"Iniciando experimento: Modelo={modelo.__class__.__name__}, Ponderación={ponderacion.__class__.__name__ if ponderacion else 'Ninguna'}, Métrica={metrica}, Reducción={reduccion}")

        vectores = []
        total_roughness_list = []  # Almacena la rugosidad total de cada acorde
        for ac in self.acordes:
            # Se espera que modelo.calcular retorne (vector, total_roughness)
            vector_base, total_rug = modelo.calcular(ac)
            if ponderacion:
                vector_ponderado = ponderacion.aplicar(vector_base, ac, modelo)
            else:
                vector_ponderado = vector_base
            vectores.append(vector_ponderado)
            total_roughness_list.append(total_rug)
            ac.total_roughness = total_rug  # Se asigna al acorde el valor total de rugosidad

        vectores = np.array(vectores)
        print("Forma de la matriz de vectores:", vectores.shape)

        # Cálculo de la matriz de distancias
        if metrica == 'custom':
            distancias = squareform(pdist(vectores, lambda u, v: 1 - np.dot(u, v)))
        else:
            distancias = squareform(pdist(vectores, metrica))
        
        reducer_obj = None
        stress = None
        stress_normalizado = None
        if reduccion == 'MDS':
            reducer = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
            embeddings = reducer.fit_transform(distancias)
            stress = reducer.stress_
            dist_embedded = squareform(pdist(embeddings))
            stress_normalizado = kruskal_stress_1(distancias, dist_embedded)
        elif reduccion == 'UMAP':
            if umap is None:
                raise ModuleNotFoundError(
                    "El paquete 'umap-learn' no está instalado. Instálalo con 'pip install umap-learn' "
                    "o selecciona otro método de reducción."
                )
            if metrica.lower() in ['euclidean', 'cosine', 'cityblock', 'manhattan', 'chebyshev']:
                reducer = umap.UMAP(n_components=2, metric=metrica, random_state=42)
                embeddings = reducer.fit_transform(vectores)
            else:
                reducer = umap.UMAP(n_components=2, metric='precomputed', random_state=42)
                embeddings = reducer.fit_transform(distancias)
            reducer_obj = reducer
        elif reduccion == 'Kernel MDS':
            embeddings = kernel_mds(distancias, n_components=2)
        elif reduccion == 'Geometric MDS':
            embeddings = geometric_mds_vectorized(distancias, n_components=2)
            stress = calculate_stress_vectorized(distancias, embeddings)
            dist_embedded = squareform(pdist(embeddings))
            stress_normalizado = kruskal_stress_1(distancias, dist_embedded)
        elif reduccion == 'RepresentacionSpSf':
            from pre_process import representacion_sp_sf
            embeddings = representacion_sp_sf(self.acordes)
            reducer_obj = None
            stress = None
        else:
            raise ValueError(f"Método de reducción dimensional '{reduccion}' no soportado.")

        var_explicada = np.corrcoef(distancias.flatten(), squareform(pdist(embeddings)).flatten())[0, 1] ** 2
        tiempo_total = time.time() - start_time

        reporte_parametros_reduccion(reduccion, reducer_obj)

        resultado = ResultadoExperimento(
            nombre_modelo=modelo.__class__.__name__,
            metricas={
                'stress': stress,
                'stress_normalizado': stress_normalizado,
                'varianza_explicada': var_explicada,
                'tiempo_total': tiempo_total,
                'trustworthiness': compute_trustworthiness(vectores, embeddings) if reduccion == 'UMAP' else None,
                'knn_recall': compute_knn_recall(vectores, embeddings) if reduccion == 'UMAP' else None,
                'continuity': compute_continuity(vectores, embeddings) if reduccion == 'UMAP' else None
            },
            embeddings=embeddings,
            matriz_distancias=distancias,
            tiempo_ejecucion=tiempo_total,
            reducer_obj=reducer_obj,
            X_original=vectores
        )
        # Almacenar el resultado en un diccionario para referencia futura
        key = f"{modelo.__class__.__name__}_{str(ponderacion)}_{metrica}_{reduccion}"
        self.resultados[key] = resultado
        print(f"Experimento completado en {tiempo_total:.2f} segundos.")
        return resultado

print("[ok] lab.py: Clase 'LaboratorioAcordes' actualizada y lista para ejecutar experimentos.")


class MemoriaAcordes(LaboratorioAcordes):
  def __init__(self, acordes: List[Acorde]):
      super().__init__(acordes)

  
  
