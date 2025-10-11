"""
reduction.py
------------
Este módulo contiene funciones para realizar la reducción dimensional de los datos,
con el fin de transformar vectores de alta dimensión (por ejemplo, vectores de rugosidad)
o matrices de distancia en embeddings de baja dimensión (usualmente 2D) para su visualización y análisis.

Inventario Detallado:

1. kernel_mds(distancias: np.ndarray, n_components: int = KERNEL_MDS_N_COMPONENTS) -> np.ndarray  
   - **Propósito:** Implementar el método Kernel Multidimensional Scaling (Kernel MDS) usando un kernel RBF.  
   - **Uso:** Se emplea para generar embeddings a partir de una matriz de distancias precalculada.  
   - **Dependencias:**  
     - Librerías: numpy, logging.  
     - Constantes: KERNEL_MDS_N_COMPONENTS (importada desde config.py).  
   - **Referencia:** Llamado en la fase de reducción dimensional cuando se selecciona "Kernel MDS".

2. geometric_mds_vectorized(distancias_orig: np.ndarray, n_components: int, iterations: int = 100, tol: float = 1e-6) -> np.ndarray  
   - **Propósito:** Implementar una versión vectorizada del algoritmo Geometric MDS para obtener un embedding.  
   - **Uso:** Se puede utilizar como alternativa a Kernel MDS, especialmente cuando se requiere una implementación sin bucles explícitos.  
   - **Dependencias:**  
     - Librerías: numpy.  
   - **Referencia:** Alternativa para la reducción dimensional, invocable desde el flujo de experimentos.

3. calculate_stress_vectorized(distancias_orig: np.ndarray, embedding: np.ndarray) -> float  
   - **Propósito:** Calcular el Stress-1 de forma vectorizada, lo que permite evaluar la calidad del embedding obtenido.  
   - **Uso:** Se utiliza para monitorear la convergencia y calidad del embedding en métodos de reducción dimensional.  
   - **Dependencias:**  
     - Librerías: numpy, scipy.spatial.distance (pdist).  
   - **Referencia:** Puede integrarse en reportes o como métrica auxiliar en la evaluación de métodos de reducción.

4. laplacian_eigenmaps_embedding(umap_reducer, n_components: int = UMAP_N_COMPONENTS) -> Optional[np.ndarray]  
   - **Propósito:** Calcular un embedding utilizando Laplacian Eigenmaps a partir del grafo obtenido de un objeto UMAP.  
   - **Uso:** Ofrece una representación alternativa basada en la estructura del grafo generado por UMAP.  
   - **Dependencias:**  
     - Librerías: networkx, scipy.sparse, scipy.sparse.linalg (eigsh).  
     - Constantes: UMAP_N_COMPONENTS (importada desde config.py).  
   - **Referencia:** Se utiliza cuando se elige el método de reducción "Laplacian Eigenmaps" en el flujo de experimentos.



"""

import numpy as np
import logging
import networkx as nx
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from typing import Optional

# Importar constantes de reducción dimensional desde config.py
from config import UMAP_N_COMPONENTS, KERNEL_MDS_N_COMPONENTS

# Configurar logger (opcional)
logger = logging.getLogger(__name__)

def kernel_mds(distancias: np.ndarray, n_components: int = KERNEL_MDS_N_COMPONENTS) -> np.ndarray:
    """
    Implementa Kernel MDS usando un kernel RBF.
    
    Args:
        distancias (np.ndarray): Matriz de distancias precalculada.
        n_components (int): Número de componentes para el embedding. Por defecto, KERNEL_MDS_N_COMPONENTS.
    
    Returns:
        np.ndarray: Embeddings de dimensión (n_samples, n_components).
    
    Uso:
      - Se invoca cuando se selecciona el método "Kernel MDS" en la reducción dimensional.
    """
    # Seleccionar distancias no nulas para evitar problemas en el cálculo de la mediana
    d_nonzero = distancias[distancias > 0]
    sigma = np.median(d_nonzero) if len(d_nonzero) > 0 else 1.0
    gamma = 1 / (2 * sigma ** 2)
    logger.info(f"Kernel MDS: sigma = {sigma:.4f}, gamma = {gamma:.4f}")
    
    # Calcular el kernel RBF
    K = np.exp(-gamma * (distancias ** 2))
    if np.any(np.isnan(K)):
        logger.warning("El kernel calculado contiene NaNs.")
    
    # Centrar el kernel
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Calcular autovalores y autovectores
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    logger.info(f"Eigenvalores (descendente): {eigvals}")
    
    pos_idx = eigvals > 1e-8  # Seleccionar autovalores positivos para evitar problemas con la raíz
    if np.sum(pos_idx) < n_components:
        logger.warning("No se hallaron suficientes componentes positivas en Kernel MDS.")
        n_components = np.sum(pos_idx)
    
    # Calcular el embedding
    L = np.diag(np.sqrt(eigvals[pos_idx][:n_components]))
    Y = eigvecs[:, pos_idx][:, :n_components] @ L
    return Y

def geometric_mds_vectorized(distancias_orig: np.ndarray, n_components: int, iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Implementa el algoritmo Geometric MDS de forma vectorizada.
    
    Args:
        distancias_orig (np.ndarray): Matriz de distancias originales (m x m).
        n_components (int): Dimensionalidad deseada para el embedding.
        iterations (int): Número máximo de iteraciones (por defecto 100).
        tol (float): Tolerancia para la convergencia (por defecto 1e-6).
    
    Returns:
        np.ndarray: Embedding de dimensión (m, n_components).
    
    Uso:
      - Se puede utilizar como método alternativo para la reducción dimensional.
    """
    m = distancias_orig.shape[0]
    # Inicialización aleatoria del embedding
    embedding = np.random.rand(m, n_components)
    
    for it in range(iterations):
        diff = embedding[:, None, :] - embedding[None, :, :]  # Diferencias entre puntos
        normas = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(normas, 1.0)  # Evitar división por cero
        ratio = distancias_orig / normas
        np.fill_diagonal(ratio, 0.0)  # Excluir la diagonal
        A = embedding[None, :, :] + diff * ratio[:, :, None]
        nuevo_embedding = np.sum(A, axis=1) / (m - 1)
        
        # Calcular el stress para monitoreo
        diff_actual = embedding[:, None, :] - embedding[None, :, :]
        normas_actual = np.linalg.norm(diff_actual, axis=2)
        np.fill_diagonal(normas_actual, 0.0)
        stress = np.sum((distancias_orig - normas_actual) ** 2) / 2
        
        if np.linalg.norm(nuevo_embedding - embedding) < tol:
            print(f"Convergencia alcanzada en la iteración {it+1}")
            embedding = nuevo_embedding
            break
        
        embedding = nuevo_embedding
    return embedding

def calculate_stress_vectorized(distancias_orig: np.ndarray, embedding: np.ndarray) -> float:
    """
    Calcula el Stress-1 de forma vectorizada.
    
    Args:
        distancias_orig (np.ndarray): Matriz de distancias originales.
        embedding (np.ndarray): Embedding obtenido.
    
    Returns:
        float: Valor del stress.
    
    Uso:
      - Se utiliza para monitorear la calidad de la reducción dimensional.
    """
    diff = embedding[:, None, :] - embedding[None, :, :]
    normas = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(normas, 0.0)
    stress = np.sum((distancias_orig - normas) ** 2) / 2
    return stress

def laplacian_eigenmaps_embedding(umap_reducer, n_components: int = UMAP_N_COMPONENTS) -> Optional[np.ndarray]:
    """
    Calcula un embedding utilizando Laplacian Eigenmaps a partir del grafo obtenido 
    de un objeto UMAP.
    
    Args:
        umap_reducer: Objeto UMAP que contiene el atributo 'graph_'.
        n_components (int): Número de componentes para el embedding (por defecto UMAP_N_COMPONENTS).
    
    Returns:
        Optional[np.ndarray]: Embedding de dimensión (n_samples, n_components) o None si falla.
    
    Uso:
      - Se utiliza cuando se selecciona un método basado en el grafo (Laplacian Eigenmaps) para la reducción dimensional.
    """
    if not hasattr(umap_reducer, "graph_"):
        print("Error: El objeto UMAP no tiene el atributo 'graph_'. No se pueden calcular Laplacian Eigenmaps.")
        return None

    graph_sparse = umap_reducer.graph_
    # Crear el grafo a partir de la matriz dispersa
    G = nx.from_scipy_sparse_array(graph_sparse) if hasattr(nx, "from_scipy_sparse_array") else nx.from_scipy_sparse_matrix(graph_sparse)
    
    # Convertir el grafo a una matriz dispersa
    if hasattr(nx, "to_scipy_sparse_array"):
        A = nx.to_scipy_sparse_array(G, format='csr')
    else:
        A = nx.to_scipy_sparse_matrix(G, format='csr')
    
    # Calcular la matriz laplaciana normalizada
    L = csgraph.laplacian(A, normed=True)
    eigenvalues, eigenvectors = eigsh(L, k=n_components + 1, which='SM')
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Excluir el primer autovector (correspondiente al valor 0)
    embedding = eigenvectors[:, 1:n_components + 1]
    return embedding


