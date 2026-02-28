"""
Experimento: Sustitución Armónica — sus4-fund en la población de estructuras extendidas (C3)
=============================================================================================
Población: 69 estructuras (idéntica a exp_estructuras_extendidas_c3.py)
  - 12 díadas (m2..P8) ancladas a C3
  - 6 tríadas × 3 inversiones = 18
  - 6 tétradas × 4 inversiones = 24
  - 3 nonales  × 5 inversiones = 15
  Todas con nota MÁS GRAVE = MIDI 48 (C3)
Consulta: sus4-fund = [48, 53, 55]
Métricas: Euclidiana y Coseno en Phi_raw ∈ R^12
K=8 sustitutos (más cercanos) + K=8 anti-sustitutos (más lejanos)
"""
import os
import numpy as np

_D,_S1,_S2 = 0.24,0.0207,18.96
_C1,_C2,_A1,_A2 = 5.0,-5.0,-3.5,-5.75
N_H,DECAY,A4 = 6,0.88,440.0
EPS = 1e-12

def midi_to_freq(n):
    return A4 * 2**((n-69)/12.0)

def _pair_roughness(f1, f2):
    K = np.arange(1,N_H+1,dtype=float)
    A = DECAY**(K-1)
    P1,P2 = f1*K, f2*K
    Fm = np.minimum(P1[:,None],P2[None,:])
    Df = np.abs(P2[None,:]-P1[:,None])
    S = _D/(_S1*Fm+_S2)
    Ap = A[:,None]*A[None,:]
    return float(np.sum(Ap*(_C1*np.exp(_A1*S*Df)+_C2*np.exp(_A2*S*Df))))

def phi_raw(midi_notes):
    freqs = sorted(midi_to_freq(float(n)) for n in midi_notes)
    n = len(freqs)
    if n < 2: return np.zeros(12)
    st = [0.0]+[12.0*np.log2(freqs[i]/freqs[0]) for i in range(1,n)]
    h = np.zeros(12)
    for i in range(n-1):
        for j in range(i+1,n):
            iv = int(round(st[j]-st[i]))%12
            r = _pair_roughness(freqs[i],freqs[j])
            h[(iv-1)%12] += r
    return h

BASE = 48  # C3

def get_inversions(name, semitones, base=48):
    out = []
    for inv in range(len(semitones)):
        if inv == 0:
            inv_name = f"{name}-fund"
            notas = [base+s for s in semitones]
        else:
            inv_name = f"{name}-inv{inv}"
            shifted = sorted(semitones[inv:]+[s+12 for s in semitones[:inv]])
            notas = [base+(s-shifted[0]) for s in shifted]
        out.append((inv_name, notas))
    return out

corpus = []
for nm,ivs in [("Diada-m2",[0,1]),("Diada-M2",[0,2]),("Diada-m3",[0,3]),("Diada-M3",[0,4]),
               ("Diada-P4",[0,5]),("Diada-TT",[0,6]),("Diada-P5",[0,7]),("Diada-m6",[0,8]),
               ("Diada-M6",[0,9]),("Diada-m7",[0,10]),("Diada-M7",[0,11]),("Diada-P8",[0,12])]:
    corpus.append({"name":nm,"midi":[BASE+s for s in ivs],"cat":"Diada"})
for nm,ivs in [("Maj",[0,4,7]),("Min",[0,3,7]),("Dim",[0,3,6]),
               ("Aug",[0,4,8]),("sus2",[0,2,7]),("sus4",[0,5,7])]:
    for inv_name,notas in get_inversions(nm,ivs,BASE):
        corpus.append({"name":inv_name,"midi":notas,"cat":nm})
for nm,ivs in [("Maj7",[0,4,7,11]),("Min7",[0,3,7,10]),("Dom7",[0,4,7,10]),
               ("m7b5",[0,3,6,10]),("dim7",[0,3,6,9]),("mM7",[0,3,7,11])]:
    for inv_name,notas in get_inversions(nm,ivs,BASE):
        corpus.append({"name":inv_name,"midi":notas,"cat":nm})
for nm,ivs in [("Maj9",[0,4,7,11,14]),("Dom9",[0,4,7,10,14]),("Min9",[0,3,7,10,14])]:
    for inv_name,notas in get_inversions(nm,ivs,BASE):
        corpus.append({"name":inv_name,"midi":notas,"cat":nm})

N = len(corpus)
print(f"Poblacion: {N} estructuras")

PHI = np.array([phi_raw(c["midi"]) for c in corpus])
diff = PHI[:,None,:]-PHI[None,:,:]
D_euc = np.sqrt((diff**2).sum(-1))
norms = np.linalg.norm(PHI,axis=1,keepdims=True)+EPS
D_cos = np.clip(1.0-(PHI/norms)@(PHI/norms).T, 0.0, 2.0)

qi = next(i for i,c in enumerate(corpus) if c["name"]=="sus4-fund")
print(f"Consulta: {corpus[qi]['name']}  MIDI={corpus[qi]['midi']}")

K = 8

def top_near(D,idx,k):
    r=D[idx].copy(); r[idx]=np.inf
    return [(int(j),float(D[idx,j])) for j in np.argsort(r)[:k]]

def top_far(D,idx,k):
    r=D[idx].copy(); r[idx]=-np.inf
    return [(int(j),float(D[idx,j])) for j in np.argsort(r)[::-1][:k]]

near_euc = top_near(D_euc,qi,K)
near_cos = top_near(D_cos,qi,K)
far_euc  = top_far(D_euc,qi,K)
far_cos  = top_far(D_cos,qi,K)

def fmt(pairs):
    return [(corpus[j]['name'], d) for j,d in pairs]

print("\nSUSTITUTOS (cercanos):")
for k,(n,d) in enumerate(zip(fmt(near_euc), fmt(near_cos))):
    print(f"  {k+1}  Euc: {n[0]:<18} {n[1]:.5f}   Cos: {d[0]:<18} {d[1]:.5f}")

print("\nANTI-SUSTITUTOS (lejanos):")
for k,(n,d) in enumerate(zip(fmt(far_euc), fmt(far_cos))):
    print(f"  {k+1}  Euc: {n[0]:<18} {n[1]:.5f}   Cos: {d[0]:<18} {d[1]:.5f}")

# Guardar para el LaTeX
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "exp_sustitucion_csus4_c3_results.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write(f"N={N}, consulta=sus4-fund MIDI={corpus[qi]['midi']}\n\n")
    f.write("SUSTITUTOS\n")
    for k in range(K):
        je,de = near_euc[k]; jc,dc = near_cos[k]
        f.write(f"{k+1}  EUC: {corpus[je]['name']:<18} {de:.5f}   COS: {corpus[jc]['name']:<18} {dc:.5f}\n")
    f.write("\nANTI-SUSTITUTOS\n")
    for k in range(K):
        je,de = far_euc[k]; jc,dc = far_cos[k]
        f.write(f"{k+1}  EUC: {corpus[je]['name']:<18} {de:.5f}   COS: {corpus[jc]['name']:<18} {dc:.5f}\n")
print(f"Guardado: {out}")
