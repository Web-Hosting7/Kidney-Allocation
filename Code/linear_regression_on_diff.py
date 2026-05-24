"""
=============================================================================
 PAIRWISE RANKING VIA LINEAR REGRESSION — FROM SCRATCH
 Kidney Recipient Priority Scoring
=============================================================================

OVERVIEW
--------
We learn a scalar priority score  s(x) = wᵀx  for every patient x ∈ ℝ⁴.
A patient with a *higher* score is considered *higher priority* for a
kidney transplant.  The score is learned entirely from pairwise preference
labels: given (patient_A, patient_B) → outcome ∈ {0,1}, where 0 means A
is preferred and 1 means B is preferred.

=============================================================================
MATHEMATICAL FORMULATION
=============================================================================

1. SCORING FUNCTION
   s(xᵢ) = wᵀ xᵢ = w₁·alco + w₂·dep + w₃·life + w₄·crim
   where w ∈ ℝ⁴ is the weight vector we want to learn.

2. CONVERTING PAIRWISE → REGRESSION
   For a comparison (A, B) with outcome y ∈ {0,1}:
     · y = 0  means A wins  → we want s(A) > s(B)  → wᵀ(xA - xB) > 0
     · y = 1  means B wins  → we want s(B) > s(A)  → wᵀ(xA - xB) < 0

   Define:
     d  = xA − xB        (difference feature vector, shape 4)
     t  = 1 − 2y         (signed target: +1 if A wins, −1 if B wins)

   We then regress:  wᵀ d ≈ t

   In matrix form, stacking N pairs:
     D ∈ ℝᴺˣ⁴  (rows are difference vectors)
     t ∈ ℝᴺ    (signed targets)
     Objective: minimise  ||Dw − t||²

3. SOFT / VOTE-WEIGHTED TARGET (optional, used in second model)
   Instead of hard ±1 we use the empirical vote proportion:
     t_soft = (A_votes − B_votes) / total_responses  ∈ [−1, +1]
   This encodes the *strength* of the preference and reduces noise from
   single-respondent comparisons.

4. LOSS FUNCTION — Mean Squared Error
   L(w) = (1/N) Σᵢ (wᵀdᵢ − tᵢ)²
        = (1/N) ||Dw − t||²

5. GRADIENT OF THE LOSS
   ∂L/∂w = (2/N) Dᵀ (Dw − t)

   This is the direction of steepest ascent.  We step in the *negative*
   gradient direction to reduce the loss.

6. GRADIENT DESCENT UPDATE RULE
   w ← w − α · (2/N) Dᵀ(Dw − t)
   where α is the learning rate (step size).

7. CLOSED-FORM (ORDINARY LEAST SQUARES) SOLUTION
   Setting the gradient to zero:
     Dᵀ(Dw − t) = 0  →  DᵀD w = Dᵀt
     w* = (DᵀD)⁻¹ Dᵀt

   The Moore-Penrose pseudo-inverse is used when DᵀD is singular.

=============================================================================
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 0. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
FEATURE_COLS = ["alco", "dep", "life", "crim"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(profiles_path: str, comparisons_path: str):
    """Load patient profiles and pairwise comparisons from CSV files."""
    profiles = pd.read_csv(profiles_path)
    comparisons = pd.read_csv(comparisons_path)

    print("=" * 65)
    print("DATASET OVERVIEW")
    print("=" * 65)
    print(f"  Patients : {len(profiles)}")
    print(f"  Pairs    : {len(comparisons)}")
    print(f"  Features : {FEATURE_COLS}")
    print()
    print("Feature summary:")
    print(profiles[FEATURE_COLS].describe().round(3).to_string())
    print()
    print(f"  Outcome balance — A-wins (0): {(comparisons['outcome']==0).sum()}"
          f"   B-wins (1): {(comparisons['outcome']==1).sum()}")
    print()
    return profiles, comparisons


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

class StandardScaler:
    """
    Z-score normalisation: x̂ = (x − μ) / σ
    Computed only on training patients to avoid data leakage.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0   # guard against zero-variance columns
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def build_feature_matrix(profiles: pd.DataFrame) -> tuple:
    """
    Build a dict mapping patient_id → feature vector (numpy row).
    Returns:
        profile_dict  : {patient_id: raw feature vector}
        X_raw         : (N, 4) raw feature matrix
        patient_ids   : list of patient IDs in order
    """
    profile_dict = {}
    patient_ids  = []
    rows         = []

    for _, row in profiles.iterrows():
        pid  = row["patient_id"]
        feat = row[FEATURE_COLS].values.astype(float)
        profile_dict[pid] = feat
        patient_ids.append(pid)
        rows.append(feat)

    return profile_dict, np.array(rows), patient_ids


def build_pairwise_dataset(comparisons: pd.DataFrame,
                            profile_dict: dict,
                            use_soft_targets: bool = False) -> tuple:
    """
    Convert pairwise comparison rows into regression instances.

    For each row (A, B, outcome):
      d = x_A − x_B           (difference vector, ℝ⁴)
      t = 1 − 2·outcome       (hard target ∈ {+1, −1})
         OR
      t = (A_votes−B_votes)/total   (soft target ∈ [−1, +1])

    Both hard AND soft target columns are returned for comparison.

    Returns:
      D      : (N, 4) difference-vector matrix
      t_hard : (N,)  hard ±1 targets
      t_soft : (N,)  vote-weighted targets
      weights: (N,)  sample weights (total_responses; useful in WLS)
    """
    diff_rows = []
    t_hard    = []
    t_soft    = []
    weights   = []

    for _, row in comparisons.iterrows():
        pid_a = row["patient_A"]
        pid_b = row["patient_B"]

        if pid_a not in profile_dict or pid_b not in profile_dict:
            continue  # skip comparisons for patients without profiles

        xa = profile_dict[pid_a]
        xb = profile_dict[pid_b]
        d  = xa - xb                                 # difference vector

        y      = int(row["outcome"])
        hard_t = 1 - 2 * y                           # +1 if A wins, −1 if B wins

        total  = row["total_responses"]
        av     = row["A_votes"]
        bv     = row["B_votes"]
        soft_t = (av - bv) / total if total > 0 else hard_t

        diff_rows.append(d)
        t_hard.append(hard_t)
        t_soft.append(soft_t)
        weights.append(float(total))

    D       = np.array(diff_rows, dtype=float)        # (N, 4)
    t_hard  = np.array(t_hard, dtype=float)
    t_soft  = np.array(t_soft, dtype=float)
    weights = np.array(weights, dtype=float)

    return D, t_hard, t_soft, weights


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def train_test_split_pairs(D, t_hard, t_soft, weights, test_ratio=0.20):
    """
    Random 80/20 split on pairwise rows.

    We split at the *pair* level (not patient level) to keep the API simple.
    A patient-level split (holding out all pairs of certain patients for test)
    is a stricter alternative discussed in the Improvements section.
    """
    N = D.shape[0]
    idx = np.random.permutation(N)
    n_test  = int(N * test_ratio)
    n_train = N - n_test

    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    return (D[train_idx], t_hard[train_idx], t_soft[train_idx], weights[train_idx],
            D[test_idx],  t_hard[test_idx],  t_soft[test_idx],  weights[test_idx])


# ─────────────────────────────────────────────────────────────────────────────
# 4. LINEAR REGRESSION MODELS (FROM SCRATCH)
# ─────────────────────────────────────────────────────────────────────────────

# ── 4a. Closed-Form OLS ──────────────────────────────────────────────────────

def ols_solve(D: np.ndarray, t: np.ndarray, W: np.ndarray = None) -> np.ndarray:
    """
    Ordinary (or Weighted) Least Squares — closed form.

    Unweighted:  w* = (DᵀD)⁻¹ Dᵀt
    Weighted:    w* = (DᵀWD)⁻¹ DᵀWt  where W = diag(weights)

    Uses the pseudo-inverse (via SVD) for numerical stability when DᵀD
    is ill-conditioned.

    Parameters
    ----------
    D : (N, p) design matrix
    t : (N,)   target vector
    W : (N,)   optional sample weights (not a full matrix — we apply them
                by scaling rows of D and t before solving)

    Returns
    -------
    w : (p,) weight vector
    """
    if W is not None:
        # Scale rows: D̃ = diag(√w)·D,  t̃ = diag(√w)·t
        # Then standard OLS on D̃, t̃ gives the WLS solution.
        sqrt_w = np.sqrt(W / W.sum() * len(W))   # normalise so scale doesn't explode
        D_scaled = D * sqrt_w[:, None]
        t_scaled = t * sqrt_w
    else:
        D_scaled = D
        t_scaled = t

    # w* = (DᵀD)⁻¹ Dᵀt  via pseudo-inverse
    w = np.linalg.lstsq(D_scaled, t_scaled, rcond=None)[0]
    return w


# ── 4b. Gradient Descent ─────────────────────────────────────────────────────

def gradient_descent(D: np.ndarray,
                     t: np.ndarray,
                     learning_rate: float = 0.01,
                     max_epochs:   int   = 5000,
                     tol:          float = 1e-7,
                     verbose:      bool  = True) -> tuple:
    """
    Batch Gradient Descent to minimise MSE loss.

    Loss:     L(w) = (1/N) ||Dw − t||²
    Gradient: ∇L   = (2/N) Dᵀ(Dw − t)
    Update:   w ← w − α · ∇L

    Convergence criterion: ||∇L||₂ < tol  OR  loss change < tol

    Parameters
    ----------
    D             : (N, p) design matrix (difference vectors)
    t             : (N,)   target vector
    learning_rate : step size α
    max_epochs    : maximum number of iterations
    tol           : convergence tolerance
    verbose       : print loss every 500 epochs

    Returns
    -------
    w          : (p,) learned weight vector
    loss_history : list of MSE values per epoch
    """
    N, p = D.shape
    w    = np.zeros(p)           # initialise weights at zero
    loss_history = []

    for epoch in range(1, max_epochs + 1):
        residuals  = D @ w - t                     # (N,)   Dw − t
        loss       = float(np.mean(residuals**2))  # scalar MSE
        gradient   = (2.0 / N) * (D.T @ residuals) # (p,)  ∇L

        loss_history.append(loss)

        if verbose and epoch % 500 == 0:
            grad_norm = np.linalg.norm(gradient)
            print(f"  Epoch {epoch:>5d} | MSE = {loss:.6f} | ||∇|| = {grad_norm:.6f}")

        # Convergence check
        if np.linalg.norm(gradient) < tol:
            if verbose:
                print(f"  → Converged at epoch {epoch} (gradient norm < {tol})")
            break
        if epoch > 1 and abs(loss_history[-2] - loss) < tol:
            if verbose:
                print(f"  → Converged at epoch {epoch} (loss change < {tol})")
            break

        w = w - learning_rate * gradient            # update step

    return w, loss_history


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def predict_outcome(w: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Predict pairwise outcome from difference vectors.

    Score difference:   Δ = wᵀ(xA − xB)
    Decision rule:      Δ > 0  → A wins (predict 0)
                        Δ ≤ 0  → B wins (predict 1)
    """
    delta = D @ w                              # (N,)  score difference
    return (delta <= 0).astype(int)            # 0 if A wins, 1 if B wins


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly predicted outcomes."""
    return float(np.mean(y_true == y_pred))


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    2×2 confusion matrix:
      [[TP_A  FN_A]
       [FP_A  TP_B]]
    where class 0 = A wins, class 1 = B wins.
    """
    cm = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true.astype(int), y_pred.astype(int)):
        cm[actual][predicted] += 1
    return cm


def precision_recall_f1(cm: np.ndarray) -> dict:
    """
    Compute per-class precision, recall, F1 and macro average.

    Precision_c = TP_c / (TP_c + FP_c)
    Recall_c    = TP_c / (TP_c + FN_c)
    F1_c        = 2 · (P_c · R_c) / (P_c + R_c)
    """
    metrics = {}
    for c in [0, 1]:
        tp = cm[c][c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[c] = {"precision": p, "recall": r, "f1": f1}
    macro_f1 = np.mean([metrics[c]["f1"] for c in [0, 1]])
    metrics["macro_f1"] = macro_f1
    return metrics


def compute_patient_scores(w: np.ndarray,
                            X_scaled: np.ndarray,
                            patient_ids: list) -> pd.DataFrame:
    """
    Compute the learned scalar priority score for every patient.

    s(xᵢ) = wᵀ x̂ᵢ    (x̂ = standardised features)
    """
    scores = X_scaled @ w
    df = pd.DataFrame({"patient_id": patient_ids, "priority_score": scores})
    df["rank"] = df["priority_score"].rank(ascending=False).astype(int)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


def spearman_rank_correlation(rank1: np.ndarray, rank2: np.ndarray) -> float:
    """
    Spearman ρ = 1 − 6Σdᵢ² / (n(n²−1))
    Measures agreement between two ranking orderings.
    """
    n  = len(rank1)
    d  = rank1 - rank2
    return 1.0 - (6.0 * np.sum(d**2)) / (n * (n**2 - 1))


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - targets)))


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISE CONVERGENCE (ASCII)
# ─────────────────────────────────────────────────────────────────────────────

def ascii_loss_curve(loss_history: list, width: int = 55, height: int = 10):
    """Print an ASCII plot of the training loss curve."""
    losses = np.array(loss_history)
    lo, hi = losses.min(), losses.max()
    step   = max(1, len(losses) // width)
    sample = losses[::step][:width]

    print(f"\n  MSE ▲  [{hi:.4f}]")
    for row in range(height, -1, -1):
        threshold = lo + (hi - lo) * row / height
        line = "  "
        for val in sample:
            line += "█" if val >= threshold else " "
        print(line)
    print(f"       └{'─'*len(sample)}► epoch")
    print(f"         0{' '*(len(sample)//2-1)}→{' '*(len(sample)//2-1)}{len(losses)}")
    print(f"  Final MSE: {losses[-1]:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 7.1 Load data ────────────────────────────────────────────────────────
    profiles, comparisons = load_data(
        "patient_profiles.csv",
        "pairwise_comparisons.csv"
    )

    # ── 7.2 Build feature matrix ─────────────────────────────────────────────
    profile_dict, X_raw, patient_ids = build_feature_matrix(profiles)

    # Standardise features using ALL patients
    # (acceptable here since we split at the *pair* level, not patient level)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Update profile_dict to use scaled features
    scaled_dict = {pid: X_scaled[i] for i, pid in enumerate(patient_ids)}

    print("=" * 65)
    print("FEATURE STANDARDISATION")
    print("=" * 65)
    for f, mu, sigma in zip(FEATURE_COLS, scaler.mean_, scaler.std_):
        print(f"  {f:6s}  μ = {mu:.4f}   σ = {sigma:.4f}")
    print()

    # ── 7.3 Build pairwise regression dataset ────────────────────────────────
    D, t_hard, t_soft, sample_weights = build_pairwise_dataset(
        comparisons, scaled_dict
    )
    print(f"Pairwise dataset: D.shape = {D.shape},  "
          f"t_hard range = [{t_hard.min():.1f}, {t_hard.max():.1f}],  "
          f"t_soft range = [{t_soft.min():.3f}, {t_soft.max():.3f}]")
    print()

    # ── 7.4 Train / Test split ───────────────────────────────────────────────
    (D_tr, t_hard_tr, t_soft_tr, w_tr,
     D_te, t_hard_te, t_soft_te, w_te) = train_test_split_pairs(
        D, t_hard, t_soft, sample_weights, test_ratio=0.20
    )

    # Convert signed targets back to binary (0/1) for accuracy evaluation
    # t_hard = 1 − 2·outcome  →  outcome = (1 − t_hard) / 2
    y_true_tr = ((1 - t_hard_tr) / 2).astype(int)
    y_true_te = ((1 - t_hard_te) / 2).astype(int)

    print("=" * 65)
    print("TRAIN / TEST SPLIT")
    print("=" * 65)
    print(f"  Training pairs : {D_tr.shape[0]}")
    print(f"  Test pairs     : {D_te.shape[0]}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL A — OLS with hard targets
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("MODEL A: OLS — Hard Targets (±1)")
    print("=" * 65)
    w_ols_hard = ols_solve(D_tr, t_hard_tr)
    print(f"  Learned weights (scaled space): {dict(zip(FEATURE_COLS, w_ols_hard.round(4)))}")

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL B — OLS with soft (vote-weighted) targets
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("MODEL B: Weighted OLS — Soft Targets (vote proportions)")
    print("=" * 65)
    w_ols_soft = ols_solve(D_tr, t_soft_tr, W=w_tr)
    print(f"  Learned weights (scaled space): {dict(zip(FEATURE_COLS, w_ols_soft.round(4)))}")

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL C — Gradient Descent with hard targets
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("MODEL C: Gradient Descent — Hard Targets")
    print("=" * 65)
    # Auto-tune learning rate: use 1 / spectral_norm(DᵀD)
    # For safe convergence: α < 1 / λ_max(DᵀD / N)
    eigvals    = np.linalg.eigvalsh(D_tr.T @ D_tr / D_tr.shape[0])
    safe_lr    = 0.9 / eigvals.max()
    print(f"  Spectral norm of (DᵀD/N): {eigvals.max():.4f}  →  safe α = {safe_lr:.5f}")
    print()
    w_gd, loss_history = gradient_descent(
        D_tr, t_hard_tr,
        learning_rate=safe_lr,
        max_epochs=10000,
        tol=1e-8,
        verbose=True
    )
    ascii_loss_curve(loss_history)
    print(f"\n  Learned weights (scaled space): {dict(zip(FEATURE_COLS, w_gd.round(4)))}")
    print(f"  ||w_OLS − w_GD||₂ = {np.linalg.norm(w_ols_hard - w_gd):.2e}")

    # ─────────────────────────────────────────────────────────────────────────
    # 8. EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    models = {
        "A – OLS Hard"     : w_ols_hard,
        "B – WLS Soft"     : w_ols_soft,
        "C – GD Hard"      : w_gd,
    }

    print()
    print("=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    for name, w in models.items():
        print(f"\n  ── {name} ──────────────────────────────────────")
        for split_name, D_s, y_true, t_hard_s in [
            ("Train", D_tr, y_true_tr, t_hard_tr),
            ("Test ", D_te, y_true_te, t_hard_te),
        ]:
            y_pred  = predict_outcome(w, D_s)
            acc     = accuracy(y_true, y_pred)
            cm      = confusion_matrix_manual(y_true, y_pred)
            metrics = precision_recall_f1(cm)

            delta_pred = D_s @ w
            m_mse = mse(delta_pred, t_hard_s)
            m_mae = mae(delta_pred, t_hard_s)

            print(f"  {split_name}: Acc = {acc:.4f}  |  MSE = {m_mse:.4f}  |  MAE = {m_mae:.4f}"
                  f"  |  Macro-F1 = {metrics['macro_f1']:.4f}")

        # Full confusion matrix (test set)
        y_pred_te = predict_outcome(w, D_te)
        cm        = confusion_matrix_manual(y_true_te, y_pred_te)
        print(f"  Confusion matrix (test):")
        print(f"                Pred A-wins  Pred B-wins")
        print(f"  Actual A-wins   {cm[0][0]:>5d}         {cm[0][1]:>5d}")
        print(f"  Actual B-wins   {cm[1][0]:>5d}         {cm[1][1]:>5d}")

    # ─────────────────────────────────────────────────────────────────────────
    # 9. PATIENT RANKING
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("PATIENT PRIORITY RANKINGS  (best model = WLS Soft)")
    print("=" * 65)

    ranking_df = compute_patient_scores(w_ols_soft, X_scaled, patient_ids)
    ranking_df = ranking_df.merge(profiles[["patient_id"] + FEATURE_COLS], on="patient_id")

    print(ranking_df[["rank", "patient_id", "priority_score"] + FEATURE_COLS].to_string(index=False))

    # Compare rankings from all three models (Spearman ρ)
    print()
    print("  Spearman ρ between model rankings:")
    rank_dict = {}
    for name, w in models.items():
        scores = X_scaled @ w
        ranks  = (-scores).argsort().argsort() + 1
        rank_dict[name] = ranks

    model_names = list(rank_dict.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            rho = spearman_rank_correlation(
                rank_dict[model_names[i]].astype(float),
                rank_dict[model_names[j]].astype(float)
            )
            print(f"    ρ({model_names[i]}  vs  {model_names[j]}) = {rho:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 10. COEFFICIENT INTERPRETATION
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("COEFFICIENT INTERPRETATION  (WLS Soft model)")
    print("=" * 65)
    print("  Each weight is in the standardised feature space.")
    print("  A POSITIVE weight means that feature INCREASES priority score.")
    print("  A NEGATIVE weight means that feature DECREASES priority score.")
    print()
    print(f"  {'Feature':6s}  {'Raw Weight':>12s}  {'|Weight|':>10s}  Direction")
    print(f"  {'──────':6s}  {'──────────':>12s}  {'────────':>10s}  ─────────")
    for feat, wval in sorted(zip(FEATURE_COLS, w_ols_soft),
                              key=lambda x: abs(x[1]), reverse=True):
        direction = "↑ higher = MORE priority" if wval > 0 else "↓ higher = LESS priority"
        print(f"  {feat:6s}  {wval:+12.4f}  {abs(wval):10.4f}  {direction}")

    # ─────────────────────────────────────────────────────────────────────────
    # 11. LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION (LOPO-CV)")
    print("=" * 65)
    print("  In each fold, all pairs involving one patient are held out as test.")
    print("  This is a stricter protocol that tests generalisation to new patients.")
    print()

    lopo_accs = []
    for pid in patient_ids:
        # Identify rows where held-out patient appears
        is_test = np.array([
            (row["patient_A"] == pid or row["patient_B"] == pid)
            for _, row in comparisons.iterrows()
        ])
        D_lopo_tr = D[~is_test];  t_lopo_tr = t_soft[~is_test];  w_lopo_tr = sample_weights[~is_test]
        D_lopo_te = D[is_test];   t_lopo_te = t_hard[is_test]

        if D_lopo_tr.shape[0] < 5:
            continue  # skip if too few training examples

        w_lopo = ols_solve(D_lopo_tr, t_lopo_tr, W=w_lopo_tr)
        y_lopo_true = ((1 - t_lopo_te) / 2).astype(int)
        y_lopo_pred = predict_outcome(w_lopo, D_lopo_te)
        lopo_accs.append(accuracy(y_lopo_true, y_lopo_pred))

    print(f"  Mean LOPO Accuracy : {np.mean(lopo_accs):.4f}")
    print(f"  Std  LOPO Accuracy : {np.std(lopo_accs):.4f}")
    print(f"  Min  LOPO Accuracy : {np.min(lopo_accs):.4f}")
    print(f"  Max  LOPO Accuracy : {np.max(lopo_accs):.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 12. SUGGESTIONS FOR IMPROVEMENT
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("SUGGESTIONS FOR IMPROVEMENT")
    print("=" * 65)
    suggestions = [
        "1. Logistic / Bradley-Terry model: Replace MSE with cross-entropy loss on\n"
        "   P(B wins | A,B) = σ(w·(xB−xA)). Produces calibrated probabilities.\n",

        "2. L2 regularisation (Ridge Regression): Add λ||w||² to the loss to\n"
        "   reduce overfitting on sparse pairwise data:\n"
        "   w* = (DᵀD + λI)⁻¹ Dᵀt\n",

        "3. Patient-level train/test split: Ensure held-out pairs involve patients\n"
        "   whose profiles were never seen during training — a much harder test.\n",

        "4. Feature interactions: Add cross-terms (alco×dep, life×crim, etc.)\n"
        "   to capture non-additive effects in priority scoring.\n",

        "5. RankNet / Neural ranking: A small 2-layer network can model non-linear\n"
        "   interactions while still producing an interpretable ranking.\n",

        "6. Transitivity enforcement: If A>B and B>C, we should have A>C.\n"
        "   Checking and filtering inconsistent triples can clean noisy labels.\n",

        "7. Confidence weighting: Pairs with very few total_responses (e.g. 1–2)\n"
        "   are noisy. Downweighting them (as in Model B) is beneficial.\n",
    ]
    for s in suggestions:
        print("  " + s.replace("\n", "\n  "))

    print("=" * 65)
    print("DONE")
    print("=" * 65)

    return models, ranking_df, loss_history, w_ols_soft


if __name__ == "__main__":
    models_out, ranking_df, loss_hist, best_weights = main()