# Preference Elicitation Portal
### SURA 2026 · IIT Delhi

A Streamlit web app for collecting human pairwise preferences and learning
an interpretable ML model (RuleFit) that captures the annotator's
decision-making patterns.

---

## What it does

1. **Presents pairwise scenarios** — two options (A and B) shown side by side,
   each described by a set of parameters loaded from a CSV file
2. **Records your choices** — saved per-user to `responses/<username>_responses.csv`
3. **Trains an interpretable model** — learns IF-THEN rules from your decisions
4. **Explains the model** — rule charts, confidence scores, decision boundaries,
   counterfactual explanations, sensitivity analysis, and consistency checking

---

## Project structure

```
Front_end/
├── app.py                          ← main application
├── run.py                          ← cross-platform launcher (Mac/Windows/Linux)
├── requirements.txt                ← pip dependencies
├── environment.yml                 ← conda environment definition
├── organ_allocation_scenarios.csv  ← scenarios CSV (auto-loaded on startup)
├── users.json                      ← session state per user (auto-created)
├── responses/                      ← per-user response files (auto-created)
│   ├── alice_responses.csv
│   └── john_responses.csv
├── venv/                           ← virtual environment (not shared)
└── olds/                           ← previous versions of app.py
```

---

## Libraries used

### Core app
| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.32.0 | Web UI framework — all pages, tabs, widgets |
| `pandas` | ≥1.5.0 | CSV loading, data manipulation, response saving |
| `numpy` | ≥1.24.0 | Numerical operations, feature engineering |

### Machine learning
| Library | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.3.0 | `LogisticRegression`, `DecisionTreeClassifier`, `cross_val_score`, `balanced_accuracy_score`, `StandardScaler` |
| `imodels` | ≥1.3.0 | `RuleFitClassifier` — learns interpretable IF-THEN rules from decisions. **Requires Python 3.9–3.11** |

### Visualisation
| Library | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | ≥3.7.0 | All charts — rule coefficients, lollipop, decision boundary heatmap, radar, confidence strip, sensitivity plot, waterfall |

### Standard library (no install needed)
`json`, `os`, `warnings`, `datetime`

---

## CSV format

The app auto-loads `organ_allocation_scenarios.csv` from the project folder.
Columns must be `A_<param>` and `B_<param>` for each parameter.

```
A_age, A_years_waiting, A_health_score, B_age, B_years_waiting, B_health_score
35, 3, 8, 60, 9, 5
28, 1, 9, 45, 7, 6
```

- Any number of parameters
- Any parameter names (after `A_` / `B_`)
- All values must be numeric

The included `organ_allocation_scenarios.csv` has 20 scenarios
across 6 parameters: `age`, `years_waiting`, `health_score`,
`dependents`, `prior_transplants`, `urgency_score`.

---

## ⚠️ Python version requirement

Most features work on **any Python version (3.9+)**.

The **RuleFit model** (Rules tab, Charts tab, Examples, Analysis) requires
**Python 3.9–3.11** due to an imodels/sklearn compatibility issue with Python 3.12.

If you run on Python 3.12, the app still works — the scenarios,
response saving, and session management all function normally.
Only the model-related tabs show a warning.

---

## Setup — Mac / Linux

### Option A: venv (recommended, no conda needed)

**Step 1 — Install Python 3.11**
```bash
brew install python@3.11
```

**Step 2 — Create virtual environment**
```bash
cd /path/to/Front_end
/opt/homebrew/bin/python3.11 -m venv venv
```

**Step 3 — Activate and install**
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install imodels
```

**Step 4 — Run**
```bash
streamlit run app.py
# or
python3 run.py
```

**Every time after that:**
```bash
cd /path/to/Front_end
source venv/bin/activate
streamlit run app.py
```

---

### Option B: conda

**Step 1 — Create environment from file**
```bash
conda env create -f environment.yml
```

**Step 2 — Activate and run**
```bash
conda activate imodels_env
streamlit run app.py
```

---

## Setup — Windows

**Step 1 — Install Python 3.11**

Download from [python.org/downloads/release/python-3119](https://www.python.org/downloads/release/python-3119/)

During install, check **"Add Python to PATH"**.

**Step 2 — Open Command Prompt in the project folder**
```cmd
cd C:\path\to\Front_end
```

**Step 3 — Create virtual environment**
```cmd
python -m venv venv
```

If `python` is not found, try:
```cmd
py -3.11 -m venv venv
```

**Step 4 — Activate**
```cmd
venv\Scripts\activate
```

If you get a permissions error in PowerShell, run this once as administrator:
```powershell
Set-ExecutionPolicy RemoteSigned
```

**Step 5 — Install packages**
```cmd
pip install -r requirements.txt
pip install imodels
```

**Step 6 — Run**
```cmd
streamlit run app.py
```

**Every time after that:**
```cmd
cd C:\path\to\Front_end
venv\Scripts\activate
streamlit run app.py
```

---

## Using run.py (cross-platform launcher)

`run.py` automatically finds and uses the venv Python:

```bash
python3 run.py    # Mac/Linux
python run.py     # Windows
```

If the venv does not exist yet, it prints setup instructions.

---

## App flow

```
Auto-load CSV → Login → Answer scenarios → View model & results
```

1. App starts and auto-loads `organ_allocation_scenarios.csv`
2. Enter a username — returning users resume where they left off
3. Answer pairwise comparisons — each decision saves immediately
4. After answering at least 6 scenarios, navigate to **Model & Results**

---

## Model & Results tabs

| Tab | Contents |
|-----|----------|
| 📄 Model Card | Documentation — training data, accuracy, symmetry, limitations |
| 📋 Rules | Learned IF-THEN rules with coverage and strength |
| 📊 Charts | 4 charts — rule coefficients, importance + confidence strip, decision boundary heatmap, parameter radar |
| 🎯 Predict New Pair | Enter two patients, get prediction + confidence + counterfactual + sensitivity slider |
| 🔍 Examples | 3 agreements vs 3 disagreements, each with model explanation |
| 🧠 Analysis | Scenario difficulty overview + consistency checker |

---

## Output files

| File | Description |
|------|-------------|
| `responses/<username>_responses.csv` | One row per decision — auto-appended on every answer |
| `users.json` | Session progress per user — used to resume incomplete sessions |

Both files are created automatically on first use.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: imodels` | Activate venv first: `source venv/bin/activate` |
| `No module named streamlit` | `pip install streamlit` inside activated venv |
| `python3.11` not found on Mac | `brew install python@3.11` then use `/opt/homebrew/bin/python3.11` |
| venv has no `activate` script | `rm -rf venv` then recreate with `/opt/homebrew/bin/python3.11 -m venv venv` |
| PowerShell activation blocked | Run `Set-ExecutionPolicy RemoteSigned` as administrator |
| RuleFit tabs show warning | Use Python 3.11 venv — imodels is incompatible with Python 3.12 |
| CSV not found on startup | Ensure `organ_allocation_scenarios.csv` is in the same folder as `app.py` |
| `ensurepip` error when creating venv | Run `brew reinstall python@3.11` then recreate venv |

---

## .gitignore

Add this file to avoid committing machine-specific or generated files:

```
venv/
__pycache__/
*.pyc
.DS_Store
responses/
users.json
```

---

## References

- **RuleFit:** Friedman & Popescu (2008). *Predictive Learning via Rule Ensembles*. Annals of Applied Statistics.
- **imodels:** [github.com/csinva/imodels](https://github.com/csinva/imodels)
- **Symphony:** Bäuerle et al. (2022). *Symphony: Composing Interactive Interfaces for Machine Learning*. CHI 2022.
- **Optimal preference elicitation:** [arxiv.org/abs/2404.13895](https://arxiv.org/abs/2404.13895)
- **Streamlit:** [streamlit.io](https://streamlit.io)
