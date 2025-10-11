import numpy as np
import matplotlib.pyplot as plt

# --- Definición de la Función 1 (de pre_process.py) ---
# Usando las constantes de config.py
SETHARES_D_STAR = 0.24
SETHARES_S1 = 0.0207
SETHARES_S2 = 18.96
SETHARES_C1 = 5
SETHARES_C2 = -5
SETHARES_A1 = -3.51
SETHARES_A2 = -5.75

def calcular_disonancia_pairwise_from_module(f1, f2, a1, a2):
    Fmin = np.minimum(f1, f2)
    S = SETHARES_D_STAR / (SETHARES_S1 * Fmin + SETHARES_S2)
    Fdif = np.abs(f2 - f1)
    a = np.minimum(a1, a2)
    return a * (SETHARES_C1 * np.exp(SETHARES_A1 * S * Fdif) + SETHARES_C2 * np.exp(SETHARES_A2 * S * Fdif))

# --- Definición de la Función 2 (del Notebook rugosidad_model) ---
def dissmeasure_from_notebook(fvec, amp, model='min'):
    sort_idx = np.argsort(fvec)
    am_sorted = np.asarray(amp)[sort_idx]
    fr_sorted = np.asarray(fvec)[sort_idx]

    # Constantes hard-coded dentro de la función del notebook
    Dstar = 0.24
    S1 = 0.0207
    S2 = 18.96
    C1 = 5
    C2 = -5
    A1 = -3.51
    A2 = -5.75

    idx = np.transpose(np.triu_indices(len(fr_sorted), 1))
    fr_pairs = fr_sorted[idx]
    am_pairs = am_sorted[idx]

    Fmin = fr_pairs[:, 0]
    S = Dstar / (S1 * Fmin + S2)
    Fdif = fr_pairs[:, 1] - fr_pairs[:, 0]

    if model == 'min':
        a = np.amin(am_pairs, axis=1)
    elif model == 'product':
        a = np.prod(am_pairs, axis=1)
    else:
        raise ValueError('model should be "min" or "product"')
    SFdif = S * Fdif
    D = np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)))
    return D

# --- Generación de Datos para las Curvas ---
BASE_FREQ = 261.63 # C4
f2_range = np.linspace(BASE_FREQ, 2 * BASE_FREQ, 500)

# Curva 1
dissonance_module = [calcular_disonancia_pairwise_from_module(BASE_FREQ, f2, 1.0, 1.0) for f2 in f2_range]

# Curva 2
dissonance_notebook = [dissmeasure_from_notebook(fvec=[BASE_FREQ, f2], amp=[1.0, 1.0]) for f2 in f2_range]

# --- Creación de la Gráfica Comparativa ---
plt.figure(figsize=(12, 7))
plt.plot(f2_range, dissonance_module, label='Curva desde pre_process.py (_calcular_disonancia_pairwise)', color='blue', linewidth=4, alpha=0.7)
plt.plot(f2_range, dissonance_notebook, label='Curva desde Notebook (dissmeasure)', color='darkorange', linestyle='--', linewidth=2)
plt.title('Comparación de Curvas de Disonancia de Sethares')
plt.xlabel('Frecuencia de la segunda nota (Hz)')
plt.ylabel('Disonancia Calculada')
plt.legend()
plt.grid(True)
plt.savefig('comparacion_curvas.png')

print("Gráfica comparativa generada y guardada como comparacion_curvas.png")
