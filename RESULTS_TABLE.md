# Probing Results — Models × Variable Perturbations

**Task:** Binary token classification — detect index/key variables  
**Metric:** Best-layer Test Macro F1  
**Dataset:** XLCoST Python programs (500, program-level)  
**Probe:** Logistic Regression (class-balanced, C=1.0)

Fill in each cell with: `F1 (layer)` e.g. `0.959 (L8)`

---

## Table 1 — Python (XLCoST)

| Model | Baseline | Random Nouns | Single Chars | All Same (`x`) | Numeric (`v1,v2…`) | Misleading |
|-------|:--------:|:------------:|:------------:|:--------------:|:------------------:|:----------:|
| Qwen2.5-1.5B | 0.959 (L8) | 0.926 (L13) | | | | |
| CodeBERT | | | | | | |
| RoBERTa | | | | | | |
| Qwen2.5-0.5B | | | | | | |

---

## Table 2 — Python Renamed Variables (XLCoST)

| Model | Baseline | Random Nouns | Single Chars | All Same (`x`) | Numeric (`v1,v2…`) | Misleading |
|-------|:--------:|:------------:|:------------:|:--------------:|:------------------:|:----------:|
| Qwen2.5-1.5B | 0.926 (L13) | — | | | | |
| CodeBERT | | | | | | |
| RoBERTa | | | | | | |
| Qwen2.5-0.5B | | | | | | |

---

## Table 3 — CodeComplex (Java, Codeforces)

| Model | In-Domain F1 | XLCoST→CC Transfer | CC→XLCoST Transfer |
|-------|:------------:|:------------------:|:------------------:|
| Qwen2.5-1.5B | 0.959 (L7) | 0.911 (L13) | 0.893 (L20) |
| CodeBERT | | | |
| RoBERTa | | | |
| Qwen2.5-0.5B | | | |

---

## Table 4 — Cross-Language Generalization (Python-trained → target language)

Best-layer test accuracy shown. Python baseline in first column for reference.

| Model | Python (train) | Java | C++ | C | C# | JavaScript |
|-------|:--------------:|:----:|:---:|:-:|:--:|:----------:|
| Qwen2.5-1.5B | 0.991 | 0.985 | 0.984 | 0.973 | 0.985 | 0.988 |
| CodeBERT | | | | | | |
| RoBERTa | | | | | | |
| Qwen2.5-0.5B | | | | | | |

---

## Table 5 — Cross-Model Transfer F1

Probe trained on **row model**, evaluated on **column model's** hidden states.  
(All at each model's best layer; baseline Python experiment.)

| Train ↓ / Test → | Qwen2.5-1.5B | CodeBERT | RoBERTa | Qwen2.5-0.5B |
|------------------|:------------:|:--------:|:-------:|:------------:|
| Qwen2.5-1.5B | — | | | |
| CodeBERT | | — | | |
| RoBERTa | | | — | |
| Qwen2.5-0.5B | | | | — |

---

## Table 6 — Cross-Perturbation Transfer F1 (Qwen2.5-1.5B)

Probe trained on **row strategy**, evaluated on **column strategy's** hidden states.

| Train ↓ / Test → | Baseline | Rand Nouns | Single Chars | All Same | Numeric | Misleading |
|------------------|:--------:|:----------:|:------------:|:--------:|:-------:|:----------:|
| Baseline | — | | | | | |
| Random Nouns | | — | | | | |
| Single Chars | | | — | | | |
| All Same | | | | — | | |
| Numeric | | | | | — | |
| Misleading | | | | | | — |

---

## Summary: Δ F1 vs Baseline (best layer, Qwen2.5-1.5B)

| Perturbation | Python F1 | Δ vs Baseline | Interpretation |
|---|:---:|:---:|---|
| Baseline | 0.959 | — | Original names |
| Random Nouns | 0.926 | −0.033 | Neutral replacement |
| Single Chars | | | |
| All Same (`x`) | | | |
| Numeric (`v1…`) | | | |
| Misleading | | | Adversarial names |

---

## Notes

- **Baseline** = original variable names from XLCoST (e.g. `i`, `key`, `arr`)
- **Misleading** = index/key vars renamed to accumulator-sounding names (`total`, `count`, `result`); non-index vars renamed to index-sounding names (`i`, `j`, `k`, `idx`)
- **All Same** = every variable renamed to `x` — tests if identity alone carries role signal
- **Cross-model transfer** uses source model's best layer index; if target model has fewer layers, nearest available layer is used
- Results sourced from notebooks: `probing_variable_roles.ipynb`, `probing_renamed_variables.ipynb`, `probing_codecomplex.ipynb`, `probing_more_perturbations.ipynb`
