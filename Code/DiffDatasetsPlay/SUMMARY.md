# Synthetic Patient-Comparison Benchmark Datasets

These eleven CSV files share the schema of the uploaded reference dataset:

| Column      | Meaning                                        |
| ----------- | ---------------------------------------------- |
| `x1a`–`x4a` | Four features for Patient **A**                |
| `x1b`–`x4b` | Same four features for Patient **B**           |
| `y`         | `0` if Patient A wins, `1` if Patient B wins   |

Each row is **800 examples, perfectly class-balanced (400 / 400)**. Feature
marginal distributions match the empirical distributions of the original
reference data (x1 ∈ {0..4}, x2 ∈ {0..2}, x3 ∈ {−1..3}, x4 ∈ {0..2}).

## How each row is generated

1. Sample features for A and B independently from the matched distribution.
2. Apply a scoring rule `f(x)` to each patient → `score_A`, `score_B`.
3. Add Gaussian noise `N(0, σ²)` independently to each score.
4. `y = 1` if `score_B + noise_B > score_A + noise_A`, else `y = 0`.
5. Drop ties; subsample to a 50/50 class balance.

Different files use different rules `f` — the rule is the underlying
preference structure a model has to recover from pairwise comparisons.

---

## Dataset catalog

| # | Filename | Type | Scoring rule `f(x)` | Noise σ | Difficulty for linear models |
|---|----------|------|---------------------|---------|------------------------------|
| 1 | `linear_weighted.csv`         | **Linear**                | `1.5·x1 + 2.0·x2 + 1.0·x3 + 0.5·x4`                                                              | 0.5  | **Easy** — LR ≈ 0.97 |
| 2 | `linear_equal_weights.csv`    | **Linear**                | `x1 + x2 + x3 + x4`                                                                              | 0.4  | **Easy** — LR ≈ 0.95 |
| 3 | `linear_noisy.csv`            | **Linear (high noise)**   | `1.0·x1 + 1.5·x2 + 0.8·x3 + 1.2·x4`                                                              | 2.5  | **Medium** — noise caps accuracy near 0.75 |
| 4 | `polynomial_interaction.csv`  | **Nonlinear**             | `x1² + 0.5·x2·x3 + x4 − 0.3·x3²`                                                                 | 0.8  | **Medium** — squared/cross terms |
| 5 | `threshold_based.csv`         | **Nonlinear (rule)**      | step rules on x1≥3, x2≥2, x3≤0, and an (x4=1 ∧ x1≥2) combo                                       | 0.6  | **Hard** — GBM beats LR by ~10pp |
| 6 | `multiplicative_relation.csv` | **Nonlinear**             | `x1·x2 + x3·x4` (pure interactions, no main effects)                                             | 0.7  | **Hard** — GBM beats LR by ~8pp |
| 7 | `exponential_relation.csv`    | **Nonlinear**             | `exp(0.6·x1) + exp(0.4·x2) + 0.3·x3`                                                             | 1.5  | **Medium** — monotonic, mostly linearizable |
| 8 | `logarithmic_relation.csv`    | **Nonlinear**             | `2·log(1+x1) + log(2+x3) + √(x2+1) − 0.2·x4`                                                     | 0.3  | **Medium** — concave, monotonic |
| 9 | `piecewise_decision.csv`      | **Nonlinear (rule)**      | clinical-decision-tree-style: five mutually-exclusive branches with different per-branch scoring | 0.7  | **Hard** — GBM beats LR by ~7pp |
| 10 | `hidden_nonlinear.csv`       | **Nonlinear (NN-like)**   | `2·tanh(w₁·x) − 1.5·tanh(w₂·x) + 0.5·tanh(w₁·x)·tanh(w₂·x)`                                      | 0.25 | **Medium** — smooth, partly linearizable |
| 11 | `xor_like.csv`               | **Nonlinear (XOR)**       | `5` if exactly one of (x1≥2, x2≥1) holds, `−1` otherwise; plus small main effects                | 0.8  | **Very hard** — GBM beats LR by ~17pp |

---

## Empirical validation (5-fold CV accuracy)

I trained two reference models on each dataset to confirm the linear/nonlinear
character of each. **Logistic Regression** is purely linear in the inputs;
**Gradient Boosting** can capture arbitrary nonlinear structure.

| Dataset                          | LogReg | GBM   | Gap (GBM − LR) |
| -------------------------------- | -----: | ----: | -------------: |
| `linear_weighted.csv`            | 0.970  | 0.922 |     −0.048     |
| `linear_equal_weights.csv`       | 0.945  | 0.912 |     −0.032     |
| `linear_noisy.csv`               | 0.756  | 0.725 |     −0.031     |
| `logarithmic_relation.csv`       | 0.901  | 0.887 |     −0.014     |
| `exponential_relation.csv`       | 0.796  | 0.780 |     −0.016     |
| `hidden_nonlinear.csv`           | 0.931  | 0.900 |     −0.031     |
| `polynomial_interaction.csv`     | 0.880  | 0.900 |     +0.020     |
| `piecewise_decision.csv`         | 0.784  | 0.858 |     +0.074     |
| `multiplicative_relation.csv`    | 0.794  | 0.876 |     +0.083     |
| `threshold_based.csv`            | 0.776  | 0.874 |     +0.098     |
| `xor_like.csv`                   | 0.603  | 0.775 |     **+0.173** |

**Reading the table:**

- **Negative gaps (LR ≥ GBM)** → the structure is fundamentally linear or
  near-linear; the simpler linear model is sufficient and the boosted model
  is just adding variance. The first six rows confirm those datasets carry
  linear preference structure.
- **Positive gaps (GBM > LR)** → the structure has irreducibly nonlinear
  components that a linear model cannot capture. `xor_like` is the textbook
  case: LR drops to barely-above-chance (60%) while GBM recovers most of the
  signal.

This gives you a clean diagnostic axis: when evaluating a candidate model
against this benchmark, the gap pattern across datasets tells you what kind
of relationships the model can and cannot recover.

---

## Suggested benchmark protocol

For each candidate model, report per-dataset 5-fold CV accuracy, then
compute:

1. **Linear competence:** mean accuracy on `linear_weighted`,
   `linear_equal_weights`, `linear_noisy`.
2. **Smooth-nonlinear competence:** mean accuracy on `exponential_relation`,
   `logarithmic_relation`, `hidden_nonlinear`, `polynomial_interaction`.
3. **Hard-nonlinear competence:** mean accuracy on `threshold_based`,
   `multiplicative_relation`, `piecewise_decision`, `xor_like`.

A model that scores well on (1) but poorly on (3) is linear-only; a model
that scores well on (3) but poorly on (1) is overfitting; a model that
scores well on all three is genuinely flexible.
