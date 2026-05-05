# Probing Code LLMs — Variable Role Detection

Investigates whether large language models encode syntactic variable roles (e.g. index/key variables) in their hidden states, using the [XLCoST](https://github.com/reddy-lab-code-research/XLCoST) dataset across 7 programming languages.

---

## Setup

```bash

# create and activate a virtual environment
python3 -m venv algoverse
source algoverse/bin/activate

# install dependencies
pip install -r requirements.txt
```

**Download the dataset** — get `XLCoST_data.zip` and unzip it into the repo root:

```bash
unzip XLCoST_data.zip
```

---

## How to Use

All experiments are in `notebooks/`. Run them in order or independently — each is self-contained.

### 1. Baseline probing (original variable names)

```bash
jupyter notebook notebooks/probing_variable_roles.ipynb
```

Trains a logistic regression probe on hidden states of `Qwen2.5-1.5B` to detect index/key variables in Python code. Also evaluates cross-language transfer to Java, C++, C, C#, JavaScript.

### 2. Renamed variable probing

```bash
jupyter notebook notebooks/probing_renamed_variables.ipynb
```

Same experiment but all variable names are replaced with random nouns (e.g. `i` → `drum`, `key` → `leaf`). Tests whether the model relies on surface names or syntactic position.

### 3. CodeComplex generalization

```bash
jupyter notebook notebooks/probing_codecomplex.ipynb
```

Applies probes trained on XLCoST to [CodeComplex](https://huggingface.co/datasets/codeparrot/codecomplex) — a separate dataset of real Java competitive programming submissions. Tests out-of-distribution generalization.

### 4. Multiple perturbation strategies

```bash
jupyter notebook notebooks/probing_more_perturbations.ipynb
```

Runs the probing experiment under 6 variable perturbation strategies:

| Strategy | Description |
|----------|-------------|
| `baseline` | Original names |
| `random_nouns` | Replaced with everyday nouns |
| `single_chars` | Replaced with a, b, c… |
| `all_same` | Everything renamed to `x` |
| `numeric_vars` | Replaced with v1, v2, v3… |
| `misleading` | Index vars get accumulator names; non-index get index names |

### 5. Directional misalignment analysis

```bash
jupyter notebook notebooks/probing_directional_analysis.ipynb
```

Investigates *why* cross-language transfer accuracy is high (~97%) despite low probe weight cosine similarity (~0.15–0.30). Tests four hypotheses via margin analysis, subspace principal angles, class-mean direction alignment, and feature dimension overlap.

---

## Results

- Results and plots are saved to `results/` (organized by experiment)
- Numeric results are tracked in `RESULTS_TABLE.md`
- A full written summary is in `RESULTS.md`

---

## Project Structure

```
├── notebooks/
│   ├── probing_variable_roles.ipynb        # baseline
│   ├── probing_renamed_variables.ipynb     # renamed vars
│   ├── probing_codecomplex.ipynb           # cross-dataset
│   ├── probing_more_perturbations.ipynb    # 6 perturbation strategies
│   └── probing_directional_analysis.ipynb # why misalignment happens
├── results/
│   ├── baseline/
│   ├── renamed/
│   ├── codecomplex/
│   ├── more_perturbated_variables/
│   └── directional_analysis/
├── XLCoST_data/                            # dataset (not tracked in git)
├── RESULTS.md                              # full results writeup
├── RESULTS_TABLE.md                        # fill-in table for all models
└── requirements.txt
```

---

## Model

All notebooks default to `Qwen/Qwen2.5-1.5B`. To swap models, change `MODEL_NAME` at the top of any notebook — CodeBERT, RoBERTa, and Qwen2.5-0.5B comparison cells are already included at the bottom of each notebook.
