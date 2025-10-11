"""
metrics.py
----------
Este módulo contiene funciones para evaluar la calidad de la reducción dimensional de
los embeddings, midiendo en qué medida se preserva la estructura de los datos originales.

Las métricas implementadas incluyen:
  1. Trustworthiness: Evalúa la proporción de vecinos en el espacio embebido que también 
     eran vecinos en el espacio original.
  2. KNN Recall: Mide la fracción de vecinos originales que se mantienen en el espacio embebido.
  3. Continuity: Penaliza la pérdida de vecinos originales en la representación embebida.
  4. Rank Correlation (Spearman): Evalúa la correlación entre las disimilitudes originales 
     y las distancias en el espacio embebido.
  5. compute_additional_metrics: Agrega las métricas anteriores en un diccionario para facilitar 
     su reporte e interpretación.
  6. reporte_parametros_reduccion: Imprime un reporte detallado con los parámetros usados en el 
     método de reducción dimensional.

Dependencias:
  - Se importa la constante EVAL_N_NEIGHBORS desde config.py para definir el número de vecinos.
  - Utiliza funciones de SciPy (pdist, squareform, spearmanr) y scikit-learn (trustworthiness, NearestNeighbors).
  - Se usa numpy para operaciones numéricas.

Inventario Detallado:
  • compute_trustworthiness(X_original, X_embedded, n_neighbors=EVAL_N_NEIGHBORS) -> float  
      - Calcula la métrica de trustworthiness usando la función de scikit-learn.  
      - Uso: Se utiliza en compute_additional_metrics y en reportes finales de experimentos.
  
  • compute_knn_recall(X_original, X_embedded, n_neighbors=EVAL_N_NEIGHBORS) -> float  
      - Calcula la fracción promedio de vecinos originales que se mantienen en el espacio embebido.  
      - Uso: Se usa en compute_additional_metrics.
  
  • compute_continuity(X_original, X_embedded, n_neighbors=EVAL_N_NEIGHBORS) -> float  
      - Mide la continuidad, penalizando la pérdida de vecinos originales en la representación embebida.  
      - Uso: Se usa en compute_additional_metrics.
  
  • compute_rank_correlation(X_original, X_embedded) -> float  
      - Calcula la correlación de rangos (Spearman) entre las disimilitudes originales y las distancias 
        en el espacio embebido.  
      - Uso: Se usa en compute_additional_metrics para evaluar la preservación de la estructura relativa.
  
  • compute_additional_metrics(X_original, X_embedded, n_neighbors=EVAL_N_NEIGHBORS) -> dict  
      - Calcula y retorna un diccionario con las métricas: Trustworthiness, Continuity, KNN Recall y Rank Correlation.  
      - Uso: Se utiliza en la generación de reportes y en la función de ejecución de experimentos.
  
  • reporte_parametros_reduccion(reduccion, reducer_obj=None)  
      - Imprime un reporte con los parámetros específicos del método de reducción dimensional empleado.  
      - Uso: Se invoca tras la reducción dimensional para informar al usuario sobre la configuración usada.

"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Optional

# Importar constante de evaluación desde config.py
from config import EVAL_N_NEIGHBORS

def compute_trustworthiness(X_original: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = EVAL_N_NEIGHBORS) -> float:
    """
    Calcula la métrica 'trustworthiness' utilizando la función de scikit-learn.
    
    Args:
        X_original (np.ndarray): Matriz de datos originales.
        X_embedded (np.ndarray): Matriz de datos en el espacio embebido.
        n_neighbors (int): Número de vecinos a considerar (por defecto EVAL_N_NEIGHBORS).
    
    Returns:
        float: Valor de trustworthiness.
    
    Uso:
      - Se utiliza en compute_additional_metrics y en los reportes finales de experimentos.
    """
    return trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)

def compute_knn_recall(X_original: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = EVAL_N_NEIGHBORS) -> float:
    """
    Calcula el 'KNN recall', evaluando la fracción promedio de vecinos originales que se mantienen 
    como vecinos en el espacio embebido.
    
    Args:
        X_original (np.ndarray): Datos originales.
        X_embedded (np.ndarray): Datos embebidos.
        n_neighbors (int): Número de vecinos a considerar (por defecto EVAL_N_NEIGHBORS).
    
    Returns:
        float: Valor medio del KNN recall.
    
    Uso:
      - Se utiliza en compute_additional_metrics.
    """
    nbrs_orig = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean').fit(X_original)
    _, indices_orig = nbrs_orig.kneighbors(X_original)
    nbrs_emb = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean').fit(X_embedded)
    _, indices_emb = nbrs_emb.kneighbors(X_embedded)

    # Excluir el mismo punto (índice 0) de cada conjunto de vecinos
    indices_orig = indices_orig[:, 1:]
    indices_emb = indices_emb[:, 1:]

    recall_list = [len(set(indices_orig[i]).intersection(set(indices_emb[i]))) / n_neighbors 
                   for i in range(X_original.shape[0])]
    return np.mean(recall_list)

def compute_continuity(X_original: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = EVAL_N_NEIGHBORS) -> float:
    """
    Calcula la métrica 'continuity', que penaliza la pérdida de vecinos originales en el espacio embebido.
    
    Args:
        X_original (np.ndarray): Matriz de datos originales.
        X_embedded (np.ndarray): Matriz de datos embebidos.
        n_neighbors (int): Número de vecinos a considerar (por defecto EVAL_N_NEIGHBORS).
    
    Returns:
        float: Valor de continuidad.
    
    Uso:
      - Se utiliza en compute_additional_metrics para evaluar la calidad de la reducción dimensional.
    """
    n = X_original.shape[0]
    nbrs_orig = NearestNeighbors(n_neighbors=n, metric='euclidean').fit(X_original)
    _, indices_orig = nbrs_orig.kneighbors(X_original)
    nbrs_emb = NearestNeighbors(n_neighbors=n, metric='euclidean').fit(X_embedded)
    _, indices_emb = nbrs_emb.kneighbors(X_embedded)

    continuity_sum = 0.0
    for i in range(n):
        orig_neighbors = set(indices_orig[i, 1:EVAL_N_NEIGHBORS + 1])
        emb_neighbors = set(indices_emb[i, 1:EVAL_N_NEIGHBORS + 1])
        diff = emb_neighbors - orig_neighbors

        for j in diff:
            # Buscar el rango en el conjunto original
            rank = np.where(indices_orig[i] == j)[0][0] if j in indices_orig[i] else n
            continuity_sum += (rank - EVAL_N_NEIGHBORS)

    denominator = n * EVAL_N_NEIGHBORS * (2 * n - 3 * EVAL_N_NEIGHBORS - 1) / 2.0
    return 1 - (continuity_sum / denominator)

def compute_rank_correlation(X_original: np.ndarray, X_embedded: np.ndarray) -> float:
    """
    Calcula la correlación de rangos (Spearman) entre las disimilitudes originales y 
    las distancias en el espacio embebido.
    
    Args:
        X_original (np.ndarray): Datos originales.
        X_embedded (np.ndarray): Datos embebidos.
    
    Returns:
        float: Coeficiente de correlación de Spearman.
    
    Uso:
      - Se utiliza en compute_additional_metrics para evaluar la preservación de la 
        estructura relativa de las distancias.
    """
    d_orig = squareform(pdist(X_original))
    d_emb = squareform(pdist(X_embedded))
    triu_idx = np.triu_indices_from(d_orig, k=1)
    orig_vals = d_orig[triu_idx]
    emb_vals = d_emb[triu_idx]
    rho, _ = spearmanr(orig_vals, emb_vals)
    return rho

def compute_additional_metrics(X_original: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = EVAL_N_NEIGHBORS) -> Dict[str, float]:
    """
    Calcula un conjunto de métricas para evaluar la calidad de la reducción dimensional:
      - Trustworthiness
      - Continuity
      - KNN Recall
      - Rank Correlation (Spearman)
    
    Args:
        X_original (np.ndarray): Datos originales.
        X_embedded (np.ndarray): Datos embebidos.
        n_neighbors (int): Número de vecinos a considerar (por defecto EVAL_N_NEIGHBORS).
    
    Returns:
        dict: Diccionario con las métricas calculadas.
    
    Uso:
      - Se utiliza en la generación de reportes y en la función de ejecución de experimentos.
    """
    trust = compute_trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)
    cont = compute_continuity(X_original, X_embedded, n_neighbors=n_neighbors)
    knn_recall = compute_knn_recall(X_original, X_embedded, n_neighbors=n_neighbors)
    rank_corr = compute_rank_correlation(X_original, X_embedded)
    return {
        "Trustworthiness": trust,
        "Continuity": cont,
        "Neighborhood Hit Rate": knn_recall,
        "Rank Correlation (Spearman)": rank_corr
    }

def reporte_parametros_reduccion(reduccion: str, reducer_obj: Optional[object] = None):
    """
    Imprime un reporte informativo sobre los parámetros del método de reducción dimensional utilizado.
    
    Args:
        reduccion (str): Nombre del método de reducción dimensional (ej. 'MDS', 'UMAP', etc.).
        reducer_obj (Optional[object]): Objeto reductor, si aplica, para extraer sus parámetros.
    
    Uso:
      - Se llama tras la reducción dimensional para informar al usuario sobre la configuración del método.
    """
    print("=== Detalles del Método de Reducción ===")
    if reduccion == "MDS":
        print("MDS: Se usó dissimilarity='precomputed'.")
    elif reduccion == "t-SNE":
        print("t-SNE: Perplexity = 15, Random State = 42")
    elif reduccion == "UMAP":
        print("UMAP: n_components=2, metric configurada según la métrica elegida, Random State=42")
        if reducer_obj is not None:
            print("  Parámetros UMAP:")
            if hasattr(reducer_obj, "n_neighbors"):
                print("    n_neighbors:", reducer_obj.n_neighbors)
            if hasattr(reducer_obj, "min_dist"):
                print("    min_dist:", reducer_obj.min_dist)
            if hasattr(reducer_obj, "metric"):
                print("    metric:", reducer_obj.metric)
    elif reduccion == "Kernel MDS":
        print("Kernel MDS: Se usó kernel RBF con validaciones adicionales.")
    else:
        print("Método de reducción no reconocido.")
