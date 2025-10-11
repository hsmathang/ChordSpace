import numpy as np
import matplotlib.pyplot as plt

# --- Función de Disonancia Principal (del Notebook) ---
def dissmeasure(fvec, amp, model='min'):
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
    if len(idx) == 0:
        return 0.0
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

# --- Parámetros Globales de la Simulación ---
ji_intervals = {
    '1/1': 1/1, '16/15': 16/15, '9/8': 9/8, '6/5': 6/5, '5/4': 5/4, '4/3': 4/3,
    '7/5': 7/5, '3/2': 3/2, '8/5': 8/5, '5/3': 5/3, '9/5': 9/5, '15/8': 15/8, '2/1': 2/1
}
FUNDAMENTAL = 500
AMPLITUDE_DECAY = 0.88
N_POINTS = 3000
ALPHA_RANGE = (1.0, 2.0)
alphas = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], N_POINTS)

# --- Función para calcular una curva de disonancia para un número de armónicos ---
def calculate_dissonance_curve(n_harmonics):
    """Calcula una curva de disonancia completa para un timbre específico."""
    if n_harmonics == 1:
        # Caso especial para ondas puras (la función `dissmeasure` no está optimizada para esto)
        diss_curve = np.empty(N_POINTS)
        for i, alpha in enumerate(alphas):
            f1 = FUNDAMENTAL
            f2 = alpha * FUNDAMENTAL
            # Usamos la fórmula directamente para el caso de 1 vs 1 parciales
            Fmin = np.minimum(f1, f2)
            S = 0.24 / (0.0207 * Fmin + 18.96)
            Fdif = np.abs(f2 - f1)
            diss_curve[i] = 1.0 * (5 * np.exp(-3.51 * S * Fdif) + -5 * np.exp(-5.75 * S * Fdif))
        return diss_curve

    freq_base = FUNDAMENTAL * np.arange(1, n_harmonics + 1)
    amp_base = AMPLITUDE_DECAY**np.arange(n_harmonics)
    diss_curve = np.empty(N_POINTS)
    
    for i, alpha in enumerate(alphas):
        freq_combined = np.concatenate((freq_base, alpha * freq_base))
        amp_combined = np.concatenate((amp_base, amp_base))
        diss_curve[i] = dissmeasure(freq_combined, amp_combined)
        
    return diss_curve

# --- Cálculos de las Curvas ---
print("Calculando curva para 1 armónico (ondas puras)...")
diss_1_harmonic = calculate_dissonance_curve(1)

print("Calculando curva para 3 armónicos...")
diss_3_harmonics = calculate_dissonance_curve(3)

print("Calculando curva para 6 armónicos...")
diss_6_harmonics = calculate_dissonance_curve(6)

print("Calculando curva para 9 armónicos...")
diss_9_harmonics = calculate_dissonance_curve(9)

# --- Creación de la Gráfica ---
print("Generando la gráfica...")
plt.figure(figsize=(15, 10))

# Dibujar las 4 curvas
plt.plot(alphas, diss_1_harmonic, label='Disonancia (1 Armónico - Pura)', color='orange', linestyle='--')
plt.plot(alphas, diss_3_harmonics, label='Disonancia (3 Armónicos)', color='green')
plt.plot(alphas, diss_6_harmonics, label='Disonancia (6 Armónicos)', color='blue', linewidth=2.5)
plt.plot(alphas, diss_9_harmonics, label='Disonancia (9 Armónicos)', color='red')

# --- Resaltar puntos de interés ---
def annotate_point(ratio, name, diss_curve, color, y_offset):
    idx = np.argmin(np.abs(alphas - ratio))
    x_val = alphas[idx]
    y_val = diss_curve[idx]
    plt.scatter(x_val, y_val, color=color, s=100, zorder=5, ec='black')
    plt.text(x_val, y_val + y_offset, f'{name}\n({x_val:.2f}, {y_val:.2f})', ha='center', va='bottom', 
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

# Anotaciones para la curva de 3 armónicos (verde)
annotate_point(5/4, '3ra Mayor', diss_3_harmonics, 'green', 0.2)
annotate_point(3/2, '5ta Justa', diss_3_harmonics, 'green', 0.2)

# Anotaciones para la curva de 6 armónicos (azul)
annotate_point(5/4, '3ra Mayor', diss_6_harmonics, 'blue', 0.2)
annotate_point(3/2, '5ta Justa', diss_6_harmonics, 'blue', 0.2)

# Anotaciones para la curva de 9 armónicos (roja)
annotate_point(5/4, '3ra Mayor', diss_9_harmonics, 'red', 0.2)
annotate_point(3/2, '5ta Justa', diss_9_harmonics, 'red', 0.2)

# --- Configuración final de la Gráfica ---
plt.xscale('log')
plt.xlim(ALPHA_RANGE)
plt.xlabel('Relación de Frecuencia (Frequency Ratio)')
plt.ylabel('Disonancia Sensorial')
plt.title('Impacto del Número de Armónicos en la Curva de Disonancia')

for ratio in ji_intervals.values():
    plt.axvline(ratio, color='gray', linestyle=':', alpha=0.6)

plt.xticks(list(ji_intervals.values()), list(ji_intervals.keys()), rotation=90)
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

# Guardar la gráfica
output_filename = 'comparacion_armonicos.png'
plt.savefig(output_filename)

print(f"Gráfica generada como '{output_filename}'")