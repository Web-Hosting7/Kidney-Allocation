"""
Preference Elicitation Portal
SURA 2026 · IIT Delhi

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import random
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from fft_model import train_fft, summarize_model_changes, explain_thinking_evolution_llm
from fft_viz import fft_viz
from fft_component import fft_svg_explained, DEFAULT_FFT_PALETTE, PARAM_DIRECTION_LABELS

st.set_page_config(
    page_title="Organ Allocation Preference Study",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ────────────────────────────────────────────────────────────
# Single source of truth for every color used in the app's own CSS/markup.
# fft_component.DEFAULT_FFT_PALETTE (used by the tree SVGs) is kept value-
# identical to the entries below — same hex, different dict shape — so the
# interactive tree, the "explain simply" view, and the rest of the page never
# drift out of sync. Light mode only, by design: this app has no dark theme.
COLORS = {
    "bg":             "#ffffff",
    "surface":        "#f8fafc",
    "surface_alt":    "#f1f5f9",
    "border":         "#e2e8f0",
    "border_strong":  "#cbd5e1",
    "text":           "#0f172a",
    "text_secondary": "#475569",
    "text_muted":     "#94a3b8",
    "accent":         "#2563eb",
    "accent_hover":   "#1d4ed8",
    "accent_soft":    "#eff6ff",
    "a":              "#b91c1c",
    "a_soft":         "#fef2f2",
    "b":              "#1d4ed8",
    "b_soft":         "#eff6ff",
    "success":        "#15803d",
    "success_soft":   "#f0fdf4",
    "warning":        "#b45309",
    "warning_soft":   "#fffbeb",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer {{ display: none !important; }}
[data-testid="stToolbar"]    {{ display: none !important; }}
[data-testid="stSidebar"]    {{ display: none !important; }}
[data-testid="stHeader"]     {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* Force light mode regardless of the visitor's OS/browser theme */
:root {{ color-scheme: light !important; }}
html, body, .stApp {{ background-color: {COLORS["bg"]} !important; }}

/* Main container — full width, tight vertical padding */
.main .block-container {{
  padding-top: 2.25rem !important;
  padding-bottom: 3rem !important;
  padding-left: 2.5rem !important;
  padding-right: 2.5rem !important;
  max-width: 1100px !important;
}}

/* Base font */
html, body, [class*="css"] {{
  font-size: 16px;
  color: {COLORS["text"]};
  font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
}}

/* Headings */
h1 {{ font-size: 2rem !important;   font-weight: 700 !important; line-height: 1.2 !important; color: {COLORS["text"]} !important; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.5rem !important; font-weight: 600 !important; line-height: 1.3 !important; color: {COLORS["text"]} !important; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.25rem !important; font-weight: 600 !important; line-height: 1.4 !important; color: {COLORS["text"]} !important; }}

/* Paragraph & list text */
.stMarkdown p  {{ font-size: 16px !important; line-height: 1.7 !important; color: {COLORS["text"]}; }}
.stMarkdown li {{ font-size: 16px !important; line-height: 1.7 !important; color: {COLORS["text"]}; }}
.stMarkdown strong {{ color: {COLORS["text"]}; }}

/* All buttons */
.stButton > button {{
  font-size: 16px !important;
  padding: 11px 22px !important;
  min-height: 46px !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.05s ease !important;
}}
.stButton > button:active {{ transform: scale(0.99); }}
.stButton > button[kind="primary"] {{
  background-color: {COLORS["accent"]} !important;
  color: {COLORS["bg"]} !important;
  border: 1px solid {COLORS["accent"]} !important;
}}
.stButton > button[kind="primary"]:hover  {{ background-color: {COLORS["accent_hover"]} !important; border-color: {COLORS["accent_hover"]} !important; }}
.stButton > button[kind="secondary"] {{
  background-color: {COLORS["bg"]} !important;
  color: {COLORS["text"]} !important;
  border: 1px solid {COLORS["border"]} !important;
}}
.stButton > button[kind="secondary"]:hover {{
  background-color: {COLORS["surface"]} !important;
  border-color: {COLORS["border_strong"]} !important;
}}

/* Text input / text area */
.stTextInput > div > div > input,
.stTextArea textarea {{
  font-size: 16px !important;
  padding: 13px 16px !important;
  border-radius: 10px !important;
  border: 1.5px solid {COLORS["border"]} !important;
  background: {COLORS["surface"]} !important;
  color: {COLORS["text"]} !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {{
  border-color: {COLORS["accent"]} !important;
  box-shadow: 0 0 0 3px {COLORS["accent_soft"]} !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
  border-radius: 10px !important;
  border: 1.5px solid {COLORS["border"]} !important;
  background: {COLORS["surface"]} !important;
}}

/* Number input */
.stNumberInput > div > div > input {{
  border-radius: 10px !important;
  border: 1.5px solid {COLORS["border"]} !important;
  background: {COLORS["surface"]} !important;
}}

/* Progress bar */
[data-testid="stProgressBar"] > div {{
  height: 6px !important;
  border-radius: 3px !important;
  background: {COLORS["surface_alt"]} !important;
}}
[data-testid="stProgressBar"] > div > div {{ background: {COLORS["accent"]} !important; }}

/* Divider */
hr {{ border: none !important; border-top: 1px solid {COLORS["border"]} !important; margin: 1.5rem 0 !important; }}

/* Expander */
[data-testid="stExpander"] {{
  border: 1px solid {COLORS["border"]} !important;
  border-radius: 12px !important;
  background: {COLORS["bg"]} !important;
}}
[data-testid="stExpander"] summary {{ font-weight: 600 !important; color: {COLORS["text"]} !important; }}

/* Alerts (info / success / warning / error) — keep flat and on-brand */
[data-testid="stAlert"] {{ border-radius: 10px !important; }}

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {{ color: {COLORS["text_muted"]} !important; }}

/* Right-panel feedback card */
.fb-card {{
  background: {COLORS["surface"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 14px;
  padding: 28px 26px;
}}

/* Generic content card, used for scenario/review panels */
.content-card {{
  background: {COLORS["surface"]};
  border: 1px solid {COLORS["border"]};
  border-radius: 12px;
  padding: 18px 20px;
}}

/* Small uppercase eyebrow label */
.eyebrow {{
  font-size: 13px;
  color: {COLORS["text_muted"]};
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 600;
}}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join(APP_DIR, "responses")
USERS_FILE    = os.path.join(APP_DIR, "users.json")

# Human-readable descriptions for the landing page
PARAM_DESCRIPTIONS = {
    "age":               "The patient's age in years.",
    "years_waiting":     "How long the patient has been on the transplant waiting list.",
    "health_score":      "Overall medical health score (higher is better, scale 1–10).",
    "dependents":        "Number of people who depend on this patient (family, caregivers).",
    "prior_transplants": "Number of organ transplants the patient has previously received.",
    "urgency_score":     "Medical urgency level — how critical their need is (scale 1–10).",
}

def _clear_edit_widget_state():
    for j in range(20):
        for key in (f"feat_{j}", f"op_{j}", f"thr_{j}", f"cls_{j}",
                    f"rf_{j}", f"rop_{j}", f"rthr_{j}", f"rpref_{j}"):
            st.session_state.pop(key, None)
    st.session_state.pop("def_cls", None)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "page":            "login",
    "username":        "",
    "scenarios":       [],
    "params":          [],
    "sc_index":        0,
    "decisions":       [],
    "alignment_score": None,
    "wants_edit":      None,
    "_saved_msg":      False,
    "pending_tree":         None,
    # Part 2 — the 20-question follow-up survey
    "survey_scenarios":       [],
    "survey_index":           0,
    "survey_decisions":       [],
    "survey_alignment_score": None,

}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSV loading ───────────────────────────────────────────────────────────────
def load_csv():
    path = os.path.join(APP_DIR, "organ_allocation_scenarios.csv")
    if not os.path.exists(path):
        st.error(f"Missing data file: organ_allocation_scenarios.csv")
        st.stop()
    df     = pd.read_csv(path)
    a_cols = [c for c in df.columns if str(c).startswith("A_")]
    params = [c[2:] for c in a_cols]
    for col in a_cols + [f"B_{p}" for p in params]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    scenarios = [
        {
            "A": {p: float(row[f"A_{p}"]) for p in params},
            "B": {p: float(row[f"B_{p}"]) for p in params},
        }
        for _, row in df.iterrows()
    ]
    return params, scenarios

if not st.session_state.scenarios:
    st.session_state.params, st.session_state.scenarios = load_csv()

def generate_survey_scenarios(params, base_scenarios, n=20, seed=None):
    """
    Build a second batch of pairwise comparisons for the Part 2 follow-up
    survey ("does the model still feel right after a few more?"). Sampled
    uniformly within the same per-parameter ranges seen in the main CSV
    (integer-valued parameters stay integers), so they feel consistent with
    the original scenarios without needing extra input data.
    """
    rng = random.Random(seed)
    ranges = {}
    for p in params:
        vals = [s["A"][p] for s in base_scenarios] + [s["B"][p] for s in base_scenarios]
        lo, hi = min(vals), max(vals)
        is_int = all(float(v).is_integer() for v in vals)
        ranges[p] = (lo, hi, is_int)
    out = []
    for _ in range(n):
        a, b = {}, {}
        for p in params:
            lo, hi, is_int = ranges[p]
            if is_int:
                a[p] = float(rng.randint(int(lo), int(hi)))
                b[p] = float(rng.randint(int(lo), int(hi)))
            else:
                a[p] = round(rng.uniform(lo, hi), 1)
                b[p] = round(rng.uniform(lo, hi), 1)
        out.append({"A": a, "B": b})
    return out

# ── User persistence ──────────────────────────────────────────────────────────
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
    u.setdefault(st.session_state.username, {})
    u[st.session_state.username].update({
        "sc_index":         st.session_state.sc_index,
        "decisions":        st.session_state.decisions,
        # Persisting the actual generated survey_scenarios (not just a seed) so a
        # resumed session replays the *same* pairs the participant already saw —
        # regenerating from a seed isn't safe since Python's string hashing is
        # randomized per process.
        "survey_scenarios": st.session_state.survey_scenarios,
        "survey_index":     st.session_state.survey_index,
        "survey_decisions": st.session_state.survey_decisions,
    })
    save_users(u)

def save_fft_override(username, tree_dict):
    u = load_users()
    u.setdefault(username, {})
    u[username]["fft_override"] = tree_dict
    save_users(u)

def load_fft_override(username):
    return load_users().get(username, {}).get("fft_override")

def migrate_fft_override(username):
    """Convert any legacy override tree (exit_class only) to the new use_abs format."""
    override = load_fft_override(username)
    if not override:
        return
    changed = False
    for node in override.get("nodes", []):
        if node.get("use_abs"):
            continue
        op  = node.get("op", ">=")
        thr = float(node.get("threshold", 0))
        ec  = int(node.get("exit_class", 1))
        if op == ">=" and thr > 0:
            ph = (ec == 1)
        elif op == "<=" and thr < 0:
            ph = (ec == 0)
        elif op == ">=" and thr <= 0:
            ph = (ec == 1)
        else:
            ph = (ec == 0)
        node["use_abs"]       = True
        node["prefer_higher"] = ph
        node["op"]            = ">="
        node["threshold"]     = abs(thr)
        changed = True
    if changed:
        save_fft_override(username, override)
        train_fft_cached.clear()

def record_decision(choice):
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

    user_file = os.path.join(RESPONSES_DIR, f"{st.session_state.username}_responses.csv")
    new_df    = pd.DataFrame([row])
    if os.path.exists(user_file):
        combined = pd.concat([pd.read_csv(user_file), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(user_file, index=False)

    st.session_state.decisions.append(row)
    st.session_state.sc_index += 1
    save_session()

def record_survey_decision(choice):
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    idx = st.session_state.survey_index
    sc  = st.session_state.survey_scenarios[idx]
    row = {
        "username":  st.session_state.username,
        "scenario":  idx + 1,
        "choice":    choice,
        "timestamp": datetime.now().isoformat(),
    }
    for p in st.session_state.params:
        row[f"A_{p}"] = sc["A"][p]
        row[f"B_{p}"] = sc["B"][p]

    user_file = os.path.join(RESPONSES_DIR, f"{st.session_state.username}_survey_responses.csv")
    new_df    = pd.DataFrame([row])
    if os.path.exists(user_file):
        combined = pd.concat([pd.read_csv(user_file), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(user_file, index=False)

    st.session_state.survey_decisions.append(row)
    st.session_state.survey_index += 1
    save_session()

@st.cache_data(show_spinner=False)
def train_fft_cached(decisions_json, params_tuple, override_json):
    return train_fft(decisions_json, list(params_tuple), override_json or None)


def render_pairwise_choice_page(scenarios, idx, n_total, qparams, heading,
                                 subheading, on_record, key_prefix):
    """
    Shared UI for a single pairwise A-vs-B comparison screen. Used by both the
    Part 1 questionnaire and the Part 2 follow-up survey so the two stay
    pixel-identical and any future tweak only needs to happen in one place.
    """
    _, qcol, _ = st.columns([1, 5, 1])
    with qcol:
        st.progress(idx / n_total)
        st.markdown(
            f"<div style='color:{COLORS['text_muted']};font-size:15px;font-weight:500;"
            f"margin-top:12px;margin-bottom:26px'>Scenario {idx + 1} of {n_total}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"## {heading}")
        if subheading:
            st.markdown(
                f"<div style='color:{COLORS['text_secondary']};font-size:15px;"
                f"margin-bottom:16px;line-height:1.6'>{subheading}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        sc = scenarios[idx]
        col_a, col_b = st.columns(2, gap="large")
        for col, label, data, color, soft in [
            (col_a, "Patient A", sc["A"], COLORS["a"], COLORS["a_soft"]),
            (col_b, "Patient B", sc["B"], COLORS["b"], COLORS["b_soft"]),
        ]:
            with col:
                st.markdown(
                    f"<div style='background:{soft};border-radius:12px;"
                    f"border-left:4px solid {color};padding:24px 24px 26px 22px'>"
                    f"<div style='color:{color};font-weight:700;font-size:15px;"
                    f"letter-spacing:.04em;text-transform:uppercase;margin-bottom:24px'>"
                    f"{label}</div>",
                    unsafe_allow_html=True,
                )
                for p in qparams:
                    pname = p.replace("_", " ").title()
                    val   = data[p]
                    st.markdown(
                        f"<div style='margin-bottom:18px'>"
                        f"<div style='font-size:14px;color:{COLORS['text_secondary']};"
                        f"text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px'>"
                        f"{pname}</div>"
                        f"<div style='font-size:32px;font-weight:600;font-family:monospace;"
                        f"line-height:1.15;color:{COLORS['text']}'>{val:g}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        btn_a, btn_b = st.columns(2)
        with btn_a:
            if st.button("Choose Patient A", use_container_width=True,
                         type="primary", key=f"{key_prefix}_A_{idx}"):
                on_record("A"); st.rerun()
        with btn_b:
            if st.button("Choose Patient B", use_container_width=True,
                         type="primary", key=f"{key_prefix}_B_{idx}"):
                on_record("B"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "login":
    # Centre the card on the wide canvas
    _, card, _ = st.columns([1, 2, 1])
    with card:
        st.markdown(
            f"<div class='eyebrow' style='margin-bottom:10px'>SURA 2026 · IIT Delhi</div>",
            unsafe_allow_html=True,
        )
        st.markdown("## Organ allocation preference study")
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.markdown(
            "You'll see **pairs of patients** who each need an organ transplant, and choose "
            "who you think should receive it. There's no right or wrong answer here — we're "
            "learning how *you* reason through it. Afterwards, we'll show you a simple model "
            "that captures the pattern in your choices, and you can tell us if it feels right."
        )

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='eyebrow'>Each patient is described by six factors</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        #st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        lp = st.session_state.params
        gc1, gc2 = st.columns(2, gap="large")
        for i, p in enumerate(lp):
            desc  = PARAM_DESCRIPTIONS.get(p, p.replace("_", " ").title() + ".")
            label = p.replace("_", " ").title()
            (gc1 if i % 2 == 0 else gc2).markdown(
                f"<div style='margin-bottom:10px'><strong>{label}</strong> — "
                f"<span style='color:{COLORS['text_secondary']}'>{desc}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown("**What should we call you?**")
        uname = st.text_input(
            "Name", placeholder="Enter your name to begin",
            label_visibility="collapsed", key="login_name_input",
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Start →", type="primary", use_container_width=True, key="login_btn"):
            raw = uname.strip()
            if len(raw) < 2:
                st.error("Enter at least 2 characters to continue.")
            else:
                users = load_users()
                st.session_state.username = raw
                if raw in users:
                    ud = users[raw]
                    st.session_state.sc_index         = ud.get("sc_index", 0)
                    st.session_state.decisions        = ud.get("decisions", [])
                    st.session_state.survey_scenarios = ud.get("survey_scenarios", [])
                    st.session_state.survey_index     = ud.get("survey_index", 0)
                    st.session_state.survey_decisions = ud.get("survey_decisions", [])
                    if "alignment_score" in ud:
                        st.session_state.alignment_score = ud["alignment_score"]
                    if "survey_alignment_score" in ud:
                        st.session_state.survey_alignment_score = ud["survey_alignment_score"]
                else:
                    st.session_state.sc_index  = 0
                    st.session_state.decisions = []
                    users[raw] = {"sc_index": 0, "decisions": []}
                    save_users(users)
                st.session_state.page = "questionnaire"
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: QUESTIONNAIRE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "questionnaire":
    qparams = st.session_state.params
    n_total = len(st.session_state.scenarios)

    if st.session_state.sc_index >= n_total:
        st.session_state.page = "results"
        st.rerun()

    render_pairwise_choice_page(
        scenarios=st.session_state.scenarios,
        idx=st.session_state.sc_index,
        n_total=n_total,
        qparams=qparams,
        heading="Which patient should receive the organ?",
        subheading=None,
        on_record=record_decision,
        key_prefix="btn",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SURVEY QUESTIONNAIRE  (Part 2 — 20 more, after seeing the model)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "survey_questionnaire":
    qparams = st.session_state.params
    n_total = len(st.session_state.survey_scenarios)

    if st.session_state.survey_index >= n_total:
        st.session_state.page = "survey_results"
        st.rerun()

    render_pairwise_choice_page(
        scenarios=st.session_state.survey_scenarios,
        idx=st.session_state.survey_index,
        n_total=n_total,
        qparams=qparams,
        heading="Part 2: A few more scenarios",
        subheading=(
            "Now that you've seen your model, here are 20 more pairs in the same "
            "format. This checks whether the model still matches how you'd actually "
            "choose."
        ),
        on_record=record_survey_decision,
        key_prefix="svy",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":
    migrate_fft_override(st.session_state.username)
    rparams   = st.session_state.params
    decisions = [d for d in st.session_state.decisions if d.get("choice") in ("A", "B")]

    if len(decisions) < 6:
        st.warning(
            f"We need at least 6 answered scenarios to build a model — you've answered "
            f"{len(decisions)} so far."
        )
        if st.button("← Back to scenarios"):
            st.session_state.page = "questionnaire"
            st.rerun()
        st.stop()

    # Train the FFT
    rdecisions_json = json.dumps(decisions)
    _override       = load_fft_override(st.session_state.username)
    _override_json  = json.dumps(_override) if _override else ""

    with st.spinner("Building your preference model…"):
        fft, nodes_df, fft_stats, feat_names, fft_error = train_fft_cached(
            rdecisions_json, tuple(rparams), _override_json
        )

    if fft_error or fft_stats is None:
        st.warning(
            "We couldn't build a model from these answers — try going back and answering "
            "with a bit more variety."
        )
        if st.button("← Back to scenarios"):
            st.session_state.page = "questionnaire"
            st.rerun()
        st.stop()

    rtree_dict = fft_stats["tree"]
    editing    = (st.session_state.wants_edit is True)

    # ── Full-width: model + feedback below ───────────────────────────────────
    st.markdown("## Your preference model")
    st.markdown(
        f"<div style='color:{COLORS['text_secondary']};font-size:15px;margin-bottom:20px;"
        f"line-height:1.6'>Based on your decisions, here's the decision tree that captures "
        f"your thinking. Each step is a single check the tree applies, in order.</div>",
        unsafe_allow_html=True,
    )

    if editing:
        if st.session_state.pending_tree is None:
            st.session_state.pending_tree = {**rtree_dict, "nodes": list(rtree_dict["nodes"])}
        edited = fft_viz(
            tree=st.session_state.pending_tree,
            editing=True,
            params=rparams,
            labels=PARAM_DIRECTION_LABELS,
            key="fft_edit",
        )
        if edited is not None:
            save_fft_override(st.session_state.username, edited)
            train_fft_cached.clear()
            st.session_state.wants_edit   = None
            st.session_state.pending_tree = None
            st.session_state._saved_msg   = True
            st.rerun()
    else:
        st.session_state.pending_tree = None
        st.markdown(fft_svg_explained(
            tree=rtree_dict,
            palette=DEFAULT_FFT_PALETTE,
            node_explanations=fft_stats.get("node_explanations"),
            summary_explanation=fft_stats.get("summary_explanation"),
        ), unsafe_allow_html=True)

    if st.session_state.get("_saved_msg", False):
        st.success("Changes saved. The model now reflects your edits.")
        st.session_state._saved_msg = False

    # ── Feedback section below the model ──────────────────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### How well does this match your thinking?")
    st.markdown(
        f"<div style='color:{COLORS['text_secondary']};font-size:15px;margin-bottom:16px'>"
        f"1 = not at all &nbsp;&nbsp;·&nbsp;&nbsp; 5 = perfectly</div>",
        unsafe_allow_html=True,
    )

    score_cols    = st.columns(5)
    current_score = st.session_state.alignment_score
    for i, rcol in enumerate(score_cols):
        with rcol:
            score    = i + 1
            btn_type = "primary" if current_score == score else "secondary"
            if st.button(str(score), key=f"align_{score}",
                         type=btn_type, use_container_width=True):
                st.session_state.alignment_score = score
                ru = load_users()
                ru.setdefault(st.session_state.username, {})
                ru[st.session_state.username]["alignment_score"] = score
                save_users(ru)
                st.rerun()

    # Editing status note
    if editing:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.info("Editing mode is on. Edit the tree above, then click **Apply changes**.")

    # Correction prompt — only after rating, only when not yet decided
    elif st.session_state.alignment_score is not None and st.session_state.wants_edit is None:
        st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Want to correct anything?")
        btn_yes_col, btn_no_col = st.columns(2)
        with btn_yes_col:
            if st.button("Yes, edit it", type="primary",
                         use_container_width=True, key="edit_yes"):
                st.session_state.wants_edit = True
                st.rerun()
        with btn_no_col:
            if st.button("No, it's good", use_container_width=True, key="edit_no"):
                st.session_state.wants_edit   = False
                st.session_state.pending_tree = None
                st.rerun()

    if st.session_state.wants_edit is False:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.success("Part 1 is done and saved.")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:15px;margin:10px 0 16px;"
            f"line-height:1.6'>One more thing — we'd like to show you 20 more scenarios, "
            f"then ask once more whether the model still feels right.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Continue to Part 2 →", type="primary",
                     use_container_width=True, key="go_survey"):
            if not st.session_state.survey_scenarios:
                st.session_state.survey_scenarios = generate_survey_scenarios(
                    rparams, st.session_state.scenarios, n=20
                )
                save_session()
            st.session_state.page = "survey_questionnaire"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SURVEY RESULTS  (Part 2 — final check, after 40 total decisions)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "survey_results":
    rparams = st.session_state.params
    all_decisions = (
        [d for d in st.session_state.decisions if d.get("choice") in ("A", "B")]
        + [d for d in st.session_state.survey_decisions if d.get("choice") in ("A", "B")]
    )

    if len(all_decisions) < 6:
        st.warning("Not enough answered scenarios to rebuild the model yet.")
        st.stop()

    _override      = load_fft_override(st.session_state.username)
    _override_json = json.dumps(_override) if _override else ""

    with st.spinner("Updating your preference model with the new scenarios…"):
        fft, nodes_df, fft_stats, feat_names, fft_error = train_fft_cached(
            json.dumps(all_decisions), tuple(rparams), _override_json
        )

    if fft_error or fft_stats is None:
        st.warning("We couldn't rebuild the model — your Part 1 model is still saved.")
        st.stop()

    rtree_dict = fft_stats["tree"]

    # Part 1's tree, for the "what changed" review below. Same cache key as the
    # results page used, so this is a cache hit, not a second real train.
    part1_decisions = [d for d in st.session_state.decisions if d.get("choice") in ("A", "B")]
    _, _, part1_stats, _, part1_error = train_fft_cached(
        json.dumps(part1_decisions), tuple(rparams), _override_json
    )
    review_text = (
        summarize_model_changes(part1_stats["tree"], rtree_dict)
        if (part1_stats and not part1_error) else None
    )

    _, mcol, _ = st.columns([1, 4, 1])
    with mcol:
        st.markdown(f"<div class='eyebrow' style='margin-bottom:8px'>Part 2 · complete</div>",
                    unsafe_allow_html=True)
        st.markdown("## Here's your updated model")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:15px;margin-bottom:20px;"
            f"line-height:1.6'>This reflects all 40 of your choices — the original 20 plus "
            f"the 20 you just answered. Take a look, then let us know if it still feels right."
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(fft_svg_explained(
            tree=rtree_dict,
            palette=DEFAULT_FFT_PALETTE,
            node_explanations=fft_stats.get("node_explanations"),
            summary_explanation=fft_stats.get("summary_explanation"),
        ), unsafe_allow_html=True)

        if review_text:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='content-card'>"
                f"<div style='font-size:15px;font-weight:600;color:{COLORS['text']};"
                f"margin-bottom:8px'>📋 Model review — what changed since Part 1</div>"
                f"<div style='font-size:15px;color:{COLORS['text_secondary']};line-height:1.6'>"
                f"{review_text}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### After the extra scenarios, does this still feel right?")
        st.markdown(
            f"<div style='color:{COLORS['text_secondary']};font-size:15px;margin-bottom:16px'>"
            f"1 = not at all &nbsp;&nbsp;·&nbsp;&nbsp; 5 = perfectly</div>",
            unsafe_allow_html=True,
        )

        score_cols    = st.columns(5)
        current_score = st.session_state.survey_alignment_score
        for i, rcol in enumerate(score_cols):
            with rcol:
                score    = i + 1
                btn_type = "primary" if current_score == score else "secondary"
                if st.button(str(score), key=f"svy_align_{score}",
                             type=btn_type, use_container_width=True):
                    st.session_state.survey_alignment_score = score
                    ru = load_users()
                    ru.setdefault(st.session_state.username, {})
                    ru[st.session_state.username]["survey_alignment_score"] = score
                    save_users(ru)
                    st.rerun()

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        comment = st.text_area(
            "Anything you'd add? (optional)",
            key="survey_comment_input",
            placeholder="e.g. which step feels off, or what you'd change",
        )
        if st.button("Submit review", key="submit_survey_comment"):
            ru = load_users()
            ru.setdefault(st.session_state.username, {})
            ru[st.session_state.username]["survey_comment"] = comment
            save_users(ru)
            st.success("Thanks — your comments were saved.")

        if st.session_state.survey_alignment_score is not None:
            # LLM thinking-shift paragraph — generated once and cached in users.json
            ru = load_users()
            cached_shift = ru.get(st.session_state.username, {}).get("thinking_shift_text")
            if not cached_shift and part1_stats and not part1_error:
                with st.spinner("Generating your thinking-shift summary…"):
                    cached_shift = explain_thinking_evolution_llm(
                        part1_stats["tree"], rtree_dict
                    )
                ru.setdefault(st.session_state.username, {})
                ru[st.session_state.username]["thinking_shift_text"] = cached_shift
                save_users(ru)

            if cached_shift:
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='content-card'>"
                    f"<div style='font-size:15px;font-weight:600;color:{COLORS['text']};"
                    f"margin-bottom:8px'>🔄 How your thinking evolved</div>"
                    f"<div style='font-size:15px;color:{COLORS['text_secondary']};line-height:1.6'>"
                    f"{cached_shift}</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.success(
                "You've completed both parts of the study — thank you. "
                "Your responses have been saved."
            )
