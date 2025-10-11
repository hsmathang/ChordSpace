import numpy as np
import matplotlib.pyplot as plt

# --- Definición de la Función 1 (de pre_process.py) ---
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
    else:
        raise ValueError('model should be "min" or "product"')
    SFdif = S * Fdif
    D = np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)))
    return D

# --- Generación de Datos para las Curvas ---
BASE_FREQ = 261.63 # C4
semitone_range = np.linspace(0, 12, 500) # Eje X: 0 a 12 semitonos

dissonance_module = []
dissonance_notebook = []

for semitone in semitone_range:
    f2 = BASE_FREQ * (2**(semitone / 12.0))
    d_mod = calcular_disonancia_pairwise_from_module(BASE_FREQ, f2, 1.0, 1.0)
    d_nb = dissmeasure_from_notebook(fvec=[BASE_FREQ, f2], amp=[1.0, 1.0])
    dissonance_module.append(d_mod)
    dissonance_notebook.append(d_nb)

# --- Creación de la Gráfica Comparativa ---
plt.figure(figsize=(12, 7))
plt.plot(semitone_range, dissonance_module, label='Curva desde pre_process.py (_calcular_disonancia_pairwise)', color='blue', linewidth=4, alpha=0.7)
plt.plot(semitone_range, dissonance_notebook, label='Curva desde Notebook (dissmeasure)', color='darkorange', linestyle='--', linewidth=2)
plt.title('Comparación de Curvas de Disonancia (Eje X en Semitonos)')
plt.xlabel('Intervalo (Semitonos)')
plt.ylabel('Disonancia Calculada')
plt.xticks(np.arange(0, 13, 1)) # Marcas para cada semitono
plt.grid(True)
plt.legend()
plt.savefig('comparacion_curvas_semitonos.png')

print("Gráfica comparativa por semitonos generada y guardada como comparacion_curvas_semitonos.png")
