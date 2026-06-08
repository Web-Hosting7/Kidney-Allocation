"""
Preference Elicitation Portal
SURA 2026 · IIT Delhi

Run:  streamlit run app.py
Install: pip install streamlit pandas numpy scikit-learn matplotlib imodels

CSV file must be in the same folder as app.py.
Responses are saved to responses/<username>_responses.csv
"""




import streamlit as st
import pandas as pd
import numpy as np
import json, os, warnings
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Preference Elicitation", page_icon="🫘", layout="wide"
)

# CSV file must sit in the same folder as app.py
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join(APP_DIR, "responses")
USERS_FILE    = os.path.join(APP_DIR, "users.json")
SEED          = 42

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = "#0d1117"
BG2      = "#161b22"
BG3      = "#21262d"
BORDER   = "#30363d"
A_WIN    = "#3fb950"
B_WIN    = "#f85149"
TEXT     = "#e6edf3"
TEXT_DIM = "#7d8590"
COL_A    = "#C0392B"
COL_B    = "#2C5F8A"

# ── Session defaults ──────────────────────────────────────────────────────────
DEFAULTS = {
    "logged_in": False,
    "username":  "",
    "scenarios": [],
    "params":    [],
    "sc_index":  0,
    "decisions": [],
    "page":      "login",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSV auto-loader ───────────────────────────────────────────────────────────
def find_and_load_csv():
    path = os.path.join(APP_DIR, "organ_allocation_scenarios.csv")
    if not os.path.exists(path):
        raise ValueError(
            f"File not found: {path}\n"
            "Make sure organ_allocation_scenarios.csv is in the same folder as app.py."
        )
    df     = pd.read_csv(path)
    params, scenarios = parse_csv(df)
    return params, scenarios, "organ_allocation_scenarios.csv"


def parse_csv(df):
    """
    Validate and parse a scenarios CSV.
    Columns must be A_<name> and B_<name>.
    Returns (params, scenarios) or raises ValueError.
    """
    a_cols = [c for c in df.columns if str(c).startswith("A_")]
    if not a_cols:
        raise ValueError(
            "No columns starting with 'A_' found.\n"
            "Format: A_param1, A_param2, ..., B_param1, B_param2, ..."
        )
    params  = [c[2:] for c in a_cols]
    missing = [f"B_{p}" for p in params if f"B_{p}" not in df.columns]
    if missing:
        raise ValueError(f"Missing B_ columns: {', '.join(missing)}")
    for col in a_cols + [f"B_{p}" for p in params]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[a_cols].isnull().any().any():
        raise ValueError("Some values are not numeric. Check your CSV.")
    scenarios = []
    for _, row in df.iterrows():
        scenarios.append({
            "A": {p: float(row[f"A_{p}"]) for p in params},
            "B": {p: float(row[f"B_{p}"]) for p in params},
        })
    return params, scenarios


# ── Persistence ───────────────────────────────────────────────────────────────
def load_users():
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_users(u):
    with open(USERS_FILE, "w") as f:
        json.dump(u, f, indent=2)


def save_session():
    u = load_users()
    u[st.session_state.username] = {
        "sc_index":  st.session_state.sc_index,
        "decisions": st.session_state.decisions,
    }
    save_users(u)


def record_decision(choice):
    """Save current scenario decision and advance index."""
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    idx = st.session_state.sc_index
    sc  = st.session_state.scenarios[idx]
    row = {
        "username":  st.session_state.username,
        "scenario":  idx + 1,
        "choice":    choice,
        "timestamp": datetime.now().isoformat(),
    }
    for p in st.session_state.params:
        row[f"A_{p}"] = sc["A"][p]
        row[f"B_{p}"] = sc["B"][p]

    # Append to <username>_responses.csv inside responses/
    user_file = os.path.join(
        RESPONSES_DIR,
        f"{st.session_state.username}_responses.csv"
    )
    new_row_df = pd.DataFrame([row])
    if os.path.exists(user_file):
        existing = pd.read_csv(user_file)
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df
    combined.to_csv(user_file, index=False)

    st.session_state.decisions.append(row)
    st.session_state.sc_index += 1
    save_session()


# ── Auto-load CSV at startup ──────────────────────────────────────────────────
if not st.session_state.scenarios:
    try:
        params, scenarios, csv_name = find_and_load_csv()
        st.session_state.params    = params
        st.session_state.scenarios = scenarios
        st.session_state._csv_name = csv_name
    except ValueError as _csv_err:
        st.error(str(_csv_err))
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SYMMETRIC FEATURE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(decisions, params):
    """
    Build symmetric feature library from pairwise decisions.
    Returns (F, feat_names).
    Antisymmetric: A-B, A²-B², A³-B³  (flip sign when A↔B swapped)
    Symmetric:     weaker, stronger, combined, gap  (same when A↔B swapped)
    """
    a_vals = np.array(
        [[d.get(f"A_{p}", 0) for p in params] for d in decisions], dtype=float
    )
    b_vals = np.array(
        [[d.get(f"B_{p}", 0) for p in params] for d in decisions], dtype=float
    )
    feats = {}
    for i, p in enumerate(params):
        a  = a_vals[:, i]
        b  = b_vals[:, i]
        pn = p.replace("_", " ").title()
        feats[f"{pn}: A-B"]      = a - b
        feats[f"{pn}: A²-B²"]   = a**2 - b**2
        feats[f"{pn}: A³-B³"]   = a**3 - b**3
        feats[f"{pn}: weaker"]   = np.minimum(a, b)
        feats[f"{pn}: stronger"] = np.maximum(a, b)
        feats[f"{pn}: combined"] = a + b
        feats[f"{pn}: gap"]      = np.abs(a - b)
    F = pd.DataFrame(feats)
    return F, list(F.columns)


def augment(F, y):
    """Double the dataset by swapping A↔B (negate antisymmetric features)."""
    F_swap = F.copy()
    for col in F.columns:
        if any(x in col for x in ["A-B", "A²-B²", "A³-B³"]):
            F_swap[col] = -F[col]
    return (
        pd.concat([F, F_swap], ignore_index=True),
        np.concatenate([y, 1 - y])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RULEFIT TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def train_rulefit(decisions_json, params):
    """
    Train RuleFit on symmetric augmented features.
    Cached so it doesn't retrain on every rerender.
    Returns (rulefit, rules_df, stats, feat_names, error_msg)
    """
    try:
        import sklearn.ensemble._gradient_boosting  # triggers py3.12 error early
        from imodels import RuleFitClassifier
    except (ImportError, AttributeError):
        return None, None, None, None, (
            "RuleFit requires Python 3.9–3.11. "
            "Run the app from your imodels_env conda environment.\n"
            "BTL model and basic tree still work on any Python version."
        )

    decisions = json.loads(decisions_json)
    F, feat_names = build_features(decisions, params)
    y = np.array([1 if d["choice"] == "A" else 0 for d in decisions])
    F_aug, y_aug = augment(F, y)

    MIN_IMPORTANCE = 0.03
    MIN_SUPPORT    = 0.05
    rulefit  = None
    rules_df = None

    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        clf = RuleFitClassifier(
            max_rules=30, n_estimators=10, tree_size=4,
            random_state=SEED, alpha=alpha, include_linear=False
        )
        clf.fit(F_aug.values, y_aug, feature_names=feat_names)
        rows = []
        for rule, coef in zip(clf.rules_, clf.coef):
            imp = abs(float(coef)) * float(rule.support)
            rows.append({
                "rule":       rule.rule,
                "support":    float(rule.support),
                "coef":       float(coef),
                "importance": imp,
            })
        df_try = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
        df_try = df_try[
            (df_try["importance"] >= MIN_IMPORTANCE) &
            (df_try["support"]    >= MIN_SUPPORT)
        ]
        df_try = df_try.sort_values("importance", ascending=False)
        df_try = df_try.head(10).reset_index(drop=True)
        if len(df_try) > 0:
            rulefit  = clf
            rules_df = df_try
            break

    if rulefit is None:
        return None, None, None, feat_names, "No rules found. Answer more scenarios."

    preds = rulefit.predict(F_aug.values)
    acc   = balanced_accuracy_score(y_aug, preds)
    F_swap = F.copy()
    for col in F.columns:
        if any(x in col for x in ["A-B", "A²-B²", "A³-B³"]):
            F_swap[col] = -F[col]
    p_orig = rulefit.predict(F.values)
    p_swap = rulefit.predict(F_swap.values)
    sym    = ((p_orig + p_swap) == 1).mean()

    stats = {"acc": acc, "sym": sym, "n_rules": len(rules_df)}
    return rulefit, rules_df, stats, feat_names, None


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def dark_rcparams():
    plt.rcParams.update({
        "font.family": "monospace", "text.color": TEXT,
        "axes.labelcolor": TEXT, "xtick.color": TEXT_DIM,
        "ytick.color": TEXT, "axes.facecolor": BG2,
        "figure.facecolor": BG, "axes.edgecolor": BORDER,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": BORDER, "grid.alpha": 0.4, "grid.linewidth": 0.5,
    })


def light_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif", "text.color": "#111827",
        "axes.labelcolor": "#6b7280", "xtick.color": "#6b7280",
        "ytick.color": "#6b7280", "axes.facecolor": "white",
        "figure.facecolor": "white", "axes.edgecolor": "#d1d5db",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def chart_rulefit_coef(rules_df):
    dark_rcparams()
    n   = len(rules_df)
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7)), facecolor=BG)
    ax.set_facecolor(BG2)
    for i in range(n):
        ax.axhspan(i - 0.45, i + 0.45,
                   color=BG3 if i % 2 == 0 else BG2, zorder=0)
    ax.axvline(0, color=BORDER, linewidth=1.5, zorder=2)
    for i, (_, row) in enumerate(rules_df.iterrows()):
        c     = row["coef"]
        color = A_WIN if c > 0 else B_WIN
        ax.barh(i, c, height=0.55, color=color, alpha=0.85, zorder=3, linewidth=0)
        ha   = "left"  if c > 0 else "right"
        xpos = c + (0.012 if c > 0 else -0.012)
        ax.text(xpos, i, row["rule"], va="center", ha=ha,
                fontsize=8, color=TEXT, zorder=5)
        ax.text(c, i - 0.28, f"{c:+.3f}", va="top", ha="center",
                fontsize=6.5, color=color, alpha=0.9, zorder=5)
    ax.set_yticks([])
    cab = float(rules_df["coef"].abs().max())
    ax.set_xlim(-cab * 1.6, cab * 1.6)
    ax.set_xlabel("← Favours A         Coefficient         Favours B →",
                  fontsize=9, color=TEXT_DIM, labelpad=8)
    ax.set_title("RuleFit · Rule Coefficients",
                 fontsize=13, color=TEXT, pad=14, fontweight="bold", loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=A_WIN, label="Predicts A preferred"),
        mpatches.Patch(color=B_WIN, label="Predicts B preferred"),
    ], fontsize=8, facecolor=BG3, edgecolor=BORDER,
       labelcolor=TEXT, loc="lower right")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    plt.tight_layout(pad=1.2)
    return fig


def chart_rulefit_lollipop(rules_df):
    dark_rcparams()
    n       = len(rules_df)
    max_imp = float(rules_df["importance"].max())
    fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7)), facecolor=BG)
    ax.set_facecolor(BG2)
    for i in range(n):
        ax.axhspan(i - 0.45, i + 0.45,
                   color=BG3 if i % 2 == 0 else BG2, zorder=0)
    for i, (_, row) in enumerate(rules_df.iterrows()):
        imp   = row["importance"]
        color = A_WIN if row["coef"] > 0 else B_WIN
        ax.plot([0, imp], [i, i], color=color, alpha=0.4, linewidth=2, zorder=2)
        ax.scatter(imp, i, s=160, color=color, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.text(max_imp * 1.03, i, f"support {row['support']:.0%}",
                va="center", ha="left", fontsize=7, color=TEXT_DIM)
        ax.text(-max_imp * 0.02, i, row["rule"],
                va="center", ha="right", fontsize=8, color=TEXT)
    ax.set_yticks([])
    ax.set_xlim(-max_imp * 0.85, max_imp * 1.35)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_xlabel("Importance  ( |coef| × support )",
                  fontsize=9, color=TEXT_DIM, labelpad=8)
    ax.set_title("RuleFit · Rule Importance Ranking",
                 fontsize=13, color=TEXT, pad=14, fontweight="bold", loc="left")
    ax.legend(handles=[
        mpatches.Patch(color=A_WIN, label="Predicts A preferred"),
        mpatches.Patch(color=B_WIN, label="Predicts B preferred"),
    ], fontsize=8, facecolor=BG3, edgecolor=BORDER,
       labelcolor=TEXT, loc="lower right")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    plt.tight_layout(pad=1.2)
    return fig


def chart_rulefit_bubble(rules_df):
    light_rcparams()
    top = rules_df.head(10).copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    ax.grid(True, color="#e5e7eb", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    med_x    = top["support"].median()
    med_y    = top["coef"].abs().median()
    ylim_max = float(top["coef"].abs().max()) * 1.4
    xlim_max = float(top["support"].max()) * 1.3
    ax.axvline(med_x, color="#d1d5db", linewidth=1, linestyle="--", zorder=1)
    ax.axhline(med_y, color="#d1d5db", linewidth=1, linestyle="--", zorder=1)
    for xp, yp, label in [
        (0.01,        ylim_max * 0.93, "rare + strong"),
        (med_x + 0.01, ylim_max * 0.93, "common + strong ★"),
        (0.01,        ylim_max * 0.05, "rare + weak"),
        (med_x + 0.01, ylim_max * 0.05, "common + weak"),
    ]:
        ax.text(xp, yp, label, fontsize=8, color="#9ca3af",
                style="italic", va="bottom")
    for i, (_, row) in enumerate(top.iterrows()):
        color = A_WIN if row["coef"] > 0 else B_WIN
        ax.scatter(row["support"], abs(row["coef"]),
                   s=row["importance"] * 1000 + 10, color=color, alpha=0.75,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(row["support"], abs(row["coef"]), str(i + 1),
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=4)
    ax.set_xlabel("Coverage — fraction of decisions this rule applies to",
                  fontsize=9, color="#6b7280", labelpad=8)
    ax.set_ylabel("Strength — |coefficient|",
                  fontsize=9, color="#6b7280", labelpad=8)
    ax.set_title(f"Top {len(top)} rules: coverage vs strength",
                 fontsize=12, color="#111827", pad=12,
                 loc="left", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlim(left=0, right=xlim_max)
    ax.set_ylim(bottom=0, top=ylim_max)
    sh = [
        plt.scatter([], [], s=0.05*5000+120, color="#9ca3af",
                    alpha=0.6, edgecolors="white", label="low importance"),
        plt.scatter([], [], s=0.20*5000+120, color="#9ca3af",
                    alpha=0.6, edgecolors="white", label="high importance"),
        plt.scatter([], [], s=120, color=A_WIN, alpha=0.75,
                    edgecolors="white", label="A preferred"),
        plt.scatter([], [], s=120, color=B_WIN, alpha=0.75,
                    edgecolors="white", label="B preferred"),
    ]
    ax.legend(handles=sh, title="size = importance", fontsize=8, title_fontsize=8,
              frameon=True, framealpha=0.9, edgecolor="#e5e7eb", loc="upper right")
    plt.tight_layout(pad=1.2)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫘 Preference Portal")
    st.caption("SURA 2026 · IIT Delhi")


    st.divider()

    # Login
    st.markdown("**Login**")
    if not st.session_state.logged_in:
        uname_input = st.text_input(
            "Username", placeholder="e.g. participant_01",
            label_visibility="collapsed", key="uname_input"
        )
        if st.button("Enter →", type="primary",
                     use_container_width=True, key="login_btn"):
            raw = uname_input.strip()
            if len(raw) < 2:
                st.error("Username must be at least 2 characters.")
            else:
                users = load_users()
                st.session_state.username  = raw
                st.session_state.logged_in = True
                if raw in users:
                    ud = users[raw]
                    st.session_state.sc_index  = ud.get("sc_index",  0)
                    st.session_state.decisions = ud.get("decisions", [])
                else:
                    st.session_state.sc_index  = 0
                    st.session_state.decisions = []
                    users[raw] = {"sc_index": 0, "decisions": []}
                    save_users(users)
                st.session_state.page = "scenarios"
                st.rerun()
    else:
        st.markdown(f"**👤 {st.session_state.username}**")
        n_total = len(st.session_state.scenarios)
        st.progress(min(st.session_state.sc_index / max(n_total, 1), 1.0))
        st.caption(
            f"Scenario {min(st.session_state.sc_index + 1, n_total)}/{n_total}"
        )
        st.divider()
        for label, key in [
            ("🧪 Scenarios",       "scenarios"),
            ("🌳 Model & Results", "model"),
        ]:
            if st.button(
                label, use_container_width=True, key=f"nav_{key}",
                type="primary" if st.session_state.page == key else "secondary"
            ):
                st.session_state.page = key
                st.rerun()
        st.divider()
        if st.button("🚪 Log out", use_container_width=True, key="logout_btn"):
            save_session()
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — login gate
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.title("🫘 Preference Elicitation Portal")
    st.info("Enter your username in the sidebar to begin.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "scenarios":

    n_total = len(st.session_state.scenarios)

    # All done
    if st.session_state.sc_index >= n_total:
        st.success(f"✅ All {n_total} scenarios complete!")
        n_ans = len([d for d in st.session_state.decisions
                     if d.get("choice") in ("A", "B")])
        st.markdown(f"**{n_ans}** answered · **{n_total - n_ans}** skipped.")
        if st.button("🌳 See Model & Results",
                     type="primary", use_container_width=False):
            st.session_state.page = "model"
            st.rerun()
        st.stop()

    current_idx = st.session_state.sc_index
    sc          = st.session_state.scenarios[current_idx]
    params      = st.session_state.params

    st.markdown(f"### Scenario {current_idx + 1} of {n_total}")
    st.progress(current_idx / n_total)
    st.markdown("## Which option should be preferred?")
    st.divider()

    col_a, col_b = st.columns(2, gap="large")
    for col, label, data, color in [
        (col_a, "A", sc["A"], COL_A),
        (col_b, "B", sc["B"], COL_B),
    ]:
        with col:
            st.markdown(
                f"<div style='border:2px solid {color};border-radius:10px;"
                f"padding:1.2rem'>"
                f"<p style='color:{color};font-weight:700;font-size:15px;"
                f"margin:0 0 12px'>OPTION {label}</p>",
                unsafe_allow_html=True,
            )
            for p in params:
                st.markdown(
                    f"<div style='margin-bottom:10px'>"
                    f"<div style='font-size:11px;color:#888;text-transform:uppercase;"
                    f"letter-spacing:.05em'>{p.replace('_', ' ')}</div>"
                    f"<div style='font-size:26px;font-weight:600;"
                    f"font-family:monospace'>{data[p]:g}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("✅  Option A is better", use_container_width=True,
                     type="primary", key=f"btn_A_{current_idx}"):
            record_decision("A")
            st.rerun()
    with c2:
        if st.button("✅  Option B is better", use_container_width=True,
                     type="primary", key=f"btn_B_{current_idx}"):
            record_decision("B")
            st.rerun()
    with c3:
        if st.button("⏭ Skip", use_container_width=True,
                     key=f"btn_skip_{current_idx}"):
            record_decision("skip")
            st.rerun()

    n_done = len([d for d in st.session_state.decisions
                  if d.get("choice") in ("A", "B")])
    st.caption(f"📝 {n_done} answered so far")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "model":
    st.markdown("## 🌳 Model & Results")

    params    = st.session_state.params
    decisions = [d for d in st.session_state.decisions
                 if d.get("choice") in ("A", "B")]

    if len(decisions) < 6:
        st.warning(
            f"Need at least 6 answered scenarios (you have {len(decisions)})."
        )
        if st.button("← Back to Scenarios"):
            st.session_state.page = "scenarios"
            st.rerun()
        st.stop()

    # Train RuleFit
    decisions_json = json.dumps(decisions)
    with st.spinner("Training model…"):
        rulefit, rules_df, rf_stats, feat_names, rf_error = train_rulefit(
            decisions_json, params
        )

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisions answered", len(decisions))
    c2.metric("Parameters",         len(params))
    if rf_error is None and rf_stats:
        c3.metric("Model accuracy",  f"{rf_stats['acc']*100:.0f}%")
        c4.metric("Symmetry",        f"{rf_stats['sym']*100:.0f}%",
                  help="% of predictions that correctly flip when A and B are swapped")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Rules", "📊 Charts", "🎯 Predict New Pair"])

    # ── Tab 1: Rule text ──────────────────────────────────────────────────────
    with tab1:
        if rf_error:
            st.error(rf_error)
        elif rules_df is None or len(rules_df) == 0:
            st.warning("No rules found. Answer more scenarios.")
        else:
            st.markdown(
                f"**{len(rules_df)} rules learned from your decisions:**"
            )
            for i, (_, row) in enumerate(rules_df.iterrows()):
                direction = "→ A preferred" if row["coef"] > 0 else "→ B preferred"
                color     = COL_A           if row["coef"] > 0 else COL_B
                st.markdown(
                    f"<div style='border-left:3px solid {color};"
                    f"padding:8px 12px;margin:8px 0;"
                    f"background:#f9fafb;border-radius:4px'>"
                    f"<b>{i+1}.</b> IF &nbsp;<code>{row['rule']}</code><br>"
                    f"<span style='color:{color};font-weight:600'>"
                    f"{direction}</span>"
                    f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                    f"coverage <b>{row['support']:.0%}</b>"
                    f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                    f"strength <b>{abs(row['coef']):.2f}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Charts ─────────────────────────────────────────────────────────
    with tab2:
        if rf_error or rules_df is None or len(rules_df) == 0:
            st.warning("RuleFit unavailable — see Rules tab.")
        else:
            st.markdown("#### Chart 1 — Rule Coefficients")
            st.caption(
                "Bar length = how strongly each rule predicts. "
                "Direction = which option it favours."
            )
            fig1 = chart_rulefit_coef(rules_df)
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)

            st.divider()
            st.markdown("#### Chart 2 — Importance Ranking")
            st.caption(
                "Dot position = importance. "
                "Support % = fraction of decisions the rule applies to."
            )
            fig2 = chart_rulefit_lollipop(rules_df)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.divider()
            st.markdown("#### Chart 3 — Coverage vs Strength")
            st.caption(
                "Top-right = rules that are both common AND strong. "
                "Numbers match the rule list in the Rules tab."
            )
            fig3 = chart_rulefit_bubble(rules_df)
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

    # ── Tab 3: Predict new pair ───────────────────────────────────────────────
    with tab3:
        st.markdown("#### Predict outcome for a new patient pair")
        st.caption(
            "Enter values for two new patients — the model will predict "
            "which option is preferred based on your past decisions."
        )

        if rf_error or rulefit is None:
            st.warning("RuleFit unavailable. Run from imodels_env to use predictions.")
            st.stop()

        col_a2, col_b2 = st.columns(2)
        patient_a_input = {}
        patient_b_input = {}

        with col_a2:
            st.markdown(f"**Patient A**")
            for p in params:
                patient_a_input[p] = st.number_input(
                    p.replace("_", " ").title(),
                    key=f"pred_a_{p}",
                    value=0.0, step=1.0
                )
        with col_b2:
            st.markdown(f"**Patient B**")
            for p in params:
                patient_b_input[p] = st.number_input(
                    p.replace("_", " ").title(),
                    key=f"pred_b_{p}",
                    value=0.0, step=1.0
                )

        if st.button("🎯 Predict", type="primary", use_container_width=False,
                     key="predict_btn"):
            # Build feature vector for the new pair
            a_vals   = np.array([patient_a_input[p] for p in params], dtype=float)
            b_vals   = np.array([patient_b_input[p] for p in params], dtype=float)
            feat_vec = {}
            for i, p in enumerate(params):
                a  = a_vals[i]; b = b_vals[i]
                pn = p.replace("_", " ").title()
                feat_vec[f"{pn}: A-B"]      = a - b
                feat_vec[f"{pn}: A\u00b2-B\u00b2"] = a**2 - b**2
                feat_vec[f"{pn}: A\u00b3-B\u00b3"] = a**3 - b**3
                feat_vec[f"{pn}: weaker"]   = min(a, b)
                feat_vec[f"{pn}: stronger"] = max(a, b)
                feat_vec[f"{pn}: combined"] = a + b
                feat_vec[f"{pn}: gap"]      = abs(a - b)

            F_new = pd.DataFrame([feat_vec])
            # Align columns to training feature order
            F_new  = F_new.reindex(columns=feat_names, fill_value=0)
            pred   = rulefit.predict(F_new.values)[0]
            prob   = rulefit.predict_proba(F_new.values)[0] if hasattr(rulefit, "predict_proba") else None

            # ── Win probability gauge ──────────────────────────────────────
            winner = "A preferred" if pred == 1 else "B preferred"
            w_color = COL_A if pred == 1 else COL_B
            p_a = prob[1] if prob is not None else (1.0 if pred == 1 else 0.0)
            p_b = 1.0 - p_a

            st.divider()
            st.markdown(
                f"<div style=\"border:2px solid {w_color};border-radius:10px;"
                f"padding:16px;text-align:center;margin:12px 0\">"
                f"<span style=\"font-size:22px;font-weight:700;"
                f"color:{w_color}\">{winner}</span>"
                f"<br><span style=\"font-size:13px;color:#6b7280\">"
                f"Confidence: {max(p_a, p_b):.1%}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Gauge bar
            light_rcparams()
            fig_g, ax = plt.subplots(figsize=(8, 2))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            ax.barh(0.5, 1.0, height=0.4, color="#f3f4f6", left=0, zorder=1)
            ax.barh(0.5, p_a, height=0.4, color=A_WIN, left=0, zorder=2, alpha=0.9)
            ax.barh(0.5, p_b, height=0.4, color=B_WIN, left=p_a, zorder=2, alpha=0.9)
            if p_a > 0.08:
                ax.text(p_a / 2, 0.5, f"A  {p_a:.0%}",
                        ha="center", va="center", fontsize=11,
                        color="white", fontweight="bold", zorder=3)
            if p_b > 0.08:
                ax.text(p_a + p_b / 2, 0.5, f"B  {p_b:.0%}",
                        ha="center", va="center", fontsize=11,
                        color="white", fontweight="bold", zorder=3)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

            # ── Parameter breakdown table ──────────────────────────────────
            st.markdown("**Parameter breakdown:**")
            rows_bd = []
            for p in params:
                a = float(patient_a_input[p])
                b = float(patient_b_input[p])
                rows_bd.append({
                    "Parameter": p.replace("_", " ").title(),
                    "A value":   a,
                    "B value":   b,
                    "A − B":     round(a - b, 2),
                    "Advantage": "A" if a > b else ("B" if b > a else "Tie"),
                })
            st.dataframe(
                pd.DataFrame(rows_bd),
                use_container_width=True,
                hide_index=True,
            )

            # ── Which rules fired ──────────────────────────────────────────
            if rules_df is not None and len(rules_df) > 0:
                # Check which rules actually evaluate to True for this pair
                # by evaluating each rule condition against F_new values
                fired = []
                for _, row in rules_df.iterrows():
                    try:
                        # rule is a string like "x2: A-B > 0.5 and x4: A-B <= 1.0"
                        # evaluate it against the feature vector
                        rule_str = row["rule"]
                        # replace feature names with their values
                        eval_str = rule_str
                        for feat, val in zip(feat_names, F_new.values[0]):
                            eval_str = eval_str.replace(feat, str(round(float(val), 4)))
                        result = eval(eval_str)
                        if result:
                            fired.append(row)
                    except Exception:
                        pass

                if fired:
                    st.markdown("**Rules that fired for this pair:**")
                    for row in fired:
                        direction = "→ A preferred" if row["coef"] > 0 else "→ B preferred"
                        color = COL_A if row["coef"] > 0 else COL_B
                        st.markdown(
                            f"<div style=\"border-left:3px solid {color};"
                            f"padding:6px 12px;margin:4px 0;"
                            f"background:#f9fafb;border-radius:4px;font-size:13px\">"
                            f"<code>{row['rule']}</code> "
                            f"<span style=\"color:{color};font-weight:600\">"
                            f"{direction}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No individual rules fired — prediction based on combined model score.")