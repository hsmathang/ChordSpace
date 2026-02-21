# Comparative Analysis: `metodologia_version_repo.tex`

**Date:** 2026-02-15
**Subject:** Comparison of New Methodology Draft vs. Existing Thesis Documents
**Context:** Verification of Code Alignment and Substitution Logic

---

## 1. Overview

The new file `metodologia_version_repo.tex` represents a significant shift in tone and content compared to the previous `03Seccion03.tex`. It acts more as a **Technical Specification** of the software artifact (`ChordSpace`) than a traditional **Experimental Methodology** chapter.

| Feature | `metodologia_version_repo.tex` (New) | `03Seccion03.tex` (Previous) |
| :--- | :--- | :--- |
| **Focus** | Software Architecture & Implementation Details | Mathematical Models & Experimental Design |
| **Code Alignment** | **High.** Explicit references to specific files/functions (e.g., `services/combinatorial_generator.py`). | **Low.** Abstract descriptions of algorithms without implementation details. |
| **Experimental Scope** | Describes the **tools** to run experiments. | Describes the **specific experiments** (Exp 1-6) performed. |
| **Substitution Algo** | Mentioned briefly as k-NN application (Sec 6.2). | Discussed conceptually but vaguely. |
| **Math Depth** | Operational/Implementation formulas (Sethares/MDS). | Theoretical definitions (Sets/Vectors). |

---

## 2. Skill-Based Evaluation

### 2.1 ScholarEval Framework (Methodological Rigor)
*   **Reproducibility (Score: 5/5):** This is the strongest point of the new draft. By citing specific Python modules and global constants (e.g., `SETHARES_DECAY` in `config.py`), it makes the methodology genuinely reproducible for a developer.
*   **Completeness (Score: 3/5):** While it describes *how* to generate chords and measure roughness, it lacks the specific **substitution logic** detailed in `substitution_metrics.md` and `metodologia_temporal/cuestiones_sustitucion_primera_fase.md`.
    *   *Missing:* It ignores the weighted metric formula ($0.6 \cdot JSD + 0.4 \cdot Jaccard$) and the rationale behind using $\sqrt{JSD}$.
    *   *Missing:* It does not detail the "profiles" (Basic vs. Functional) that are currently being designed.

### 2.2 Scientific Critical Thinking (Evidence & Gaps)
*   **Gap Identification:** The new text accurately describes the *current code state* (MVP), admitting that substitution is restricted to the same cardinality. However, it fails to justify *why* this limitation exists from a musical perspective, unlike `cuestiones_sustitucion...md` which argues it's for "control" in the first phase.
*   **Experimental Context:** A thesis methodology usually describes *what was done* (The Experiments), not just *what tool was used*. The new text describes the "Generation Pipeline" effectively but removes the logical narrative of the 6 Experiments found in `04Seccion04.tex`. **Crucial:** You cannot present Results for Exp 1-6 if the Methodology doesn't define them.

---

## 3. Comparison with Repository Reality

### 3.1 What is accurate?
*   **Generation:** Accurately reflects the dual mode (Combinatorial vs Structural) and the specific filters in `population_filter.py`.
*   **Sethares:** Correctly identifies the vectorized implementation (`ModeloSetharesVec`).
*   **Dimensionality Reduction:** Correctly lists the specific library calls (`sklearn.manifold.MDS`) and parameters.

### 3.2 What is missing? (The "Substitution Algorithms")
The user specifically asked about missing "substitution algorithms". The new text (`Sec 6.2`) is extremely brief:

> "Como aplicación, el sistema calcula vecinos cercanos (k-NN)... típicamente restringidos a la misma cardinalidad (decisión de diseño del MVP)."

This completely omits the **Scientific Core** of the substitution proposal currently documented in `substitution_metrics.md`:
1.  **The Metric Definition:** It doesn't define the composite distance $D_{w}$.
2.  **The Features:** It doesn't mention *which* features feed the k-NN (is it just Roughness? Is it PC-set? Is it Voice Leading?).
3.  **The Logic:** It doesn't explain the "60/40" weighting logic or the "Profile" concept.

---

## 4. Recommendations for Integration

To solve the user's request ("missing things like substitution algorithms"), the new draft needs to be expanded using the content from `metodologia_temporal/*.md`.

**Action Items:**
1.  **Expand Section 6.2 (Sustituciones):**
    *   Import the mathematical definition of the **Substitution Metric** from `substitution_metrics.md` (Section 4: Similaridad Compuesta) or `cuestiones_sustitucion...md`.
    *   Explicitly define the "Current Profile" (MVP) used in the thesis: Weighted sum of Roughness JSD and Structural Jaccard.
2.  **link Experiments:**
    *   Add a **Section 9: Experimental Design** to this new file.
    *   Move the text defining "Exp 1: Diatonic Triads", "Exp 6: Massive Analysis", etc., from `03Seccion03.tex` into this new section.
    *   This bridges the gap between "Tool Description" and "Scientific Application".
3.  **Keep the Code Refs:** The code references are excellent. Keep them. They prove the thesis is implemented.

## 5. Conclusion
`metodologia_version_repo.tex` is a superior **Technical Methodology** but an inferior **Scientific Narrative**. It explains the *how* perfectly but loses the *why* and *what specifically* (experiments). It needs to be a hybrid: **Technical Precision** of the new file + **Experimental Roadmap** of the old file + **Substitution Math** of the markdown notes.
