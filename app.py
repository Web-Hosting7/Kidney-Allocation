"""
Preference Elicitation Portal — Flask edition
SURA 2026 · IIT Delhi

Run:  gunicorn app:app --bind 0.0.0.0:5001 --workers 1 --timeout 120
"""

import json
import os
import random
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

import db
from fft_model import train_fft
from fft_component import fft_svg_explained, DEFAULT_FFT_PALETTE, PARAM_DIRECTION_LABELS

APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.environ.get("RESPONSES_DIR") or os.path.join(APP_DIR, "responses")

# Only these five features are used in this cut of the study.
FEATURES = ["dependents", "age", "years_waiting", "urgency_score", "health_score"]

PARAM_DESCRIPTIONS = {
    "age":           "The patient's age in years.",
    "years_waiting": "How long the patient has been on the transplant waiting list.",
    "health_score":  "Overall medical health score (higher is better, scale 1–10).",
    "dependents":    "Number of people who depend on this patient (family, caregivers).",
    "urgency_score": "Medical urgency level — how critical their need is (scale 1–10).",
}

PART1_N = 20
PART2_N = 10

# Create the SQL schema (and import users.json once, if present) at startup.
db.init_db(FEATURES)

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

_train_cache = {}


def train_fft_cached(decisions, params, override):
    key = (json.dumps(decisions, sort_keys=True), tuple(params), json.dumps(override or {}))
    if key not in _train_cache:
        _train_cache[key] = train_fft(
            json.dumps(decisions), list(params), json.dumps(override) if override else None
        )
    return _train_cache[key]


# ── CSV / scenario generation ────────────────────────────────────────────────

def load_base_scenarios():
    path = os.path.join(APP_DIR, "organ_allocation_scenarios.csv")
    df = pd.read_csv(path)
    for p in FEATURES:
        df[f"A_{p}"] = pd.to_numeric(df[f"A_{p}"], errors="coerce")
        df[f"B_{p}"] = pd.to_numeric(df[f"B_{p}"], errors="coerce")
    return [
        {
            "A": {p: float(row[f"A_{p}"]) for p in FEATURES},
            "B": {p: float(row[f"B_{p}"]) for p in FEATURES},
        }
        for _, row in df.iterrows()
    ]


BASE_SCENARIOS = load_base_scenarios()


def generate_scenarios(n, base_scenarios=BASE_SCENARIOS, seed=None):
    """
    n random A/B pairs. Sampled uniformly within the per-feature ranges seen
    in the base CSV (integer features stay integers).
    """
    rng = random.Random(seed)
    ranges = {}
    for p in FEATURES:
        vals = [s["A"][p] for s in base_scenarios] + [s["B"][p] for s in base_scenarios]
        lo, hi = min(vals), max(vals)
        is_int = all(float(v).is_integer() for v in vals)
        ranges[p] = (lo, hi, is_int)
    out = []
    for _ in range(n):
        a, b = {}, {}
        for p in FEATURES:
            lo, hi, is_int = ranges[p]
            if is_int:
                a[p] = float(rng.randint(int(lo), int(hi)))
                b[p] = float(rng.randint(int(lo), int(hi)))
            else:
                a[p] = round(rng.uniform(lo, hi), 1)
                b[p] = round(rng.uniform(lo, hi), 1)
        out.append({"A": a, "B": b})
    return out


def shuffle_scenarios(scenarios):
    """
    Randomise the order in which a set of scenarios is presented.

    A fresh, entropy-seeded RNG is drawn on every call (rather than reusing a
    module-level `random` instance), so the presentation order is independently
    randomised each time this runs — per participant, per run of the app.

    Only the order changes: the scenario content and the generation logic that
    produced it are left untouched. The shuffled order is persisted with the
    participant's record, so the sequence stays stable if they resume a session
    and the recorded `scenario` number keeps pointing at the same pair.
    """
    order = list(scenarios)
    random.SystemRandom().shuffle(order)
    return order


# ── User persistence ──────────────────────────────────────────────────────────

def load_users():
    return db.load_all_users()


def save_users(u):
    db.save_all_users(u)


def get_user_record():
    username = session.get("username")
    if not username:
        return None, None
    users = load_users()
    return users, users.setdefault(username, {})


def save_user_record(record):
    db.save_user_record(session["username"], record)


def save_fft_override(username, tree_dict):
    db.set_fft_override(username, tree_dict)
    _train_cache.clear()


def load_fft_override(username):
    return db.get_fft_override(username)


def clear_fft_override(username):
    db.delete_fft_override(username)
    _train_cache.clear()


def record_decision(field, choice, idx, scenarios, csv_suffix):
    username = session["username"]
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    sc = scenarios[idx]
    row = {
        "username": username,
        "scenario": idx + 1,
        "choice": choice,
        "timestamp": datetime.now().isoformat(),
    }
    for p in FEATURES:
        row[f"A_{p}"] = sc["A"][p]
        row[f"B_{p}"] = sc["B"][p]

    user_file = os.path.join(RESPONSES_DIR, f"{username}_{csv_suffix}.csv")
    new_df = pd.DataFrame([row])
    if os.path.exists(user_file):
        combined = pd.concat([pd.read_csv(user_file), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(user_file, index=False)

    users, rec = get_user_record()
    rec.setdefault(field, []).append(row)
    rec[f"{field}_index"] = idx + 1
    save_user_record(rec)
    return rec


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def start():
    session.clear()
    return render_template(
        "start.html",
        features=FEATURES,
        descriptions=PARAM_DESCRIPTIONS,
    )


@app.route("/start", methods=["POST"])
def do_start():
    name = (request.form.get("username") or "").strip()
    if len(name) < 2:
        return render_template(
            "start.html", features=FEATURES, descriptions=PARAM_DESCRIPTIONS,
            error="Enter at least 2 characters to continue.",
        )
    session["username"] = name

    users, rec = get_user_record()
    if "part1_scenarios" not in rec:
        rec["part1_scenarios"] = shuffle_scenarios(generate_scenarios(PART1_N))
        rec["part1"] = []
        rec["part1_index"] = 0
    if "part2_scenarios" not in rec:
        rec["part2_scenarios"] = shuffle_scenarios(generate_scenarios(PART2_N))
        rec["part2"] = []
        rec["part2_index"] = 0
    save_user_record(rec)

    return redirect(url_for("questionnaire", part=1))


def _require_user():
    if "username" not in session:
        return redirect(url_for("start"))
    return None


@app.route("/questionnaire/<int:part>")
def questionnaire(part):
    guard = _require_user()
    if guard:
        return guard
    users, rec = get_user_record()
    scenarios = rec.get(f"part{part}_scenarios", [])
    idx = rec.get(f"part{part}_index", 0)
    n_total = len(scenarios)

    if idx >= n_total:
        return redirect(url_for("results") if part == 1 else url_for("final"))

    heading = "Which patient should receive the organ?" if part == 1 else "Part 2: A few more scenarios"
    subheading = None if part == 1 else (
        "Now that you've seen your model, here are a few more pairs in the same "
        "format. This checks whether the model still matches how you'd actually choose."
    )
    return render_template(
        "questionnaire.html",
        part=part, idx=idx, n_total=n_total,
        sc=scenarios[idx], features=FEATURES,
        heading=heading, subheading=subheading,
    )


@app.route("/questionnaire/<int:part>/answer", methods=["POST"])
def answer(part):
    guard = _require_user()
    if guard:
        return guard
    users, rec = get_user_record()
    idx = rec.get(f"part{part}_index", 0)
    scenarios = rec.get(f"part{part}_scenarios", [])
    choice = request.form.get("choice")
    if choice in ("A", "B") and idx < len(scenarios):
        record_decision(f"part{part}", choice, idx, scenarios,
                         csv_suffix="responses" if part == 1 else "survey_responses")
    return redirect(url_for("questionnaire", part=part))


def _train_for(rec, decisions_field, override):
    decisions = [d for d in rec.get(decisions_field, []) if d.get("choice") in ("A", "B")]
    if len(decisions) < 6:
        return None, decisions
    fft, nodes_df, stats, feat_names, error = train_fft_cached(decisions, FEATURES, override)
    if error or stats is None:
        return None, decisions
    return stats, decisions


def _scenarios_for_audit(decisions):
    """
    Compact A/B/choice-only view of past decisions, in the order they were
    answered, for the client-side "check the model against your answers"
    panel. Only what's needed to re-evaluate the tree in the browser.
    """
    out = []
    for d in decisions:
        out.append({
            "A": {p: d.get(f"A_{p}") for p in FEATURES},
            "B": {p: d.get(f"B_{p}") for p in FEATURES},
            "choice": d.get("choice"),
        })
    return out


@app.route("/results")
def results():
    guard = _require_user()
    if guard:
        return guard
    users, rec = get_user_record()
    override = load_fft_override(session["username"])
    stats, decisions = _train_for(rec, "part1", override)

    if stats is None:
        return render_template("not_ready.html", n_decisions=len(decisions))

    return render_template(
        "model.html",
        apply_url=url_for("apply_edit"),
        reset_url=url_for("reset_edit"),
        rate_url=url_for("rate"),
        continue_url=url_for("part2_intro"),
        current_score=rec.get("alignment_score"),
        has_override=bool(override),
        tree_json=stats["tree"],
        params_json=[f"{p}_diff" for p in FEATURES],
        labels_json=PARAM_DIRECTION_LABELS,
        node_explanations_json=stats.get("node_explanations") or [],
        summary_explanation_json=stats.get("summary_explanation") or "",
        past_scenarios_json=_scenarios_for_audit(decisions),
    )


@app.route("/results/rate", methods=["POST"])
def rate():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    score = payload.get("score")
    if not isinstance(score, int) or not (1 <= score <= 7):
        return jsonify({"error": "invalid score"}), 400
    users, rec = get_user_record()
    rec["alignment_score"] = score
    save_user_record(rec)
    return jsonify({"ok": True})


@app.route("/edit/apply", methods=["POST"])
def apply_edit():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    tree = payload.get("tree")
    if not tree:
        return jsonify({"error": "missing tree"}), 400
    save_fft_override(session["username"], tree)
    return jsonify({"ok": True, "redirect": url_for("results")})


@app.route("/edit/reset", methods=["POST"])
def reset_edit():
    guard = _require_user()
    if guard:
        return jsonify({"error": "not logged in"}), 401
    clear_fft_override(session["username"])
    return jsonify({"ok": True, "redirect": url_for("results")})


@app.route("/part2")
def part2_intro():
    guard = _require_user()
    if guard:
        return guard
    return redirect(url_for("questionnaire", part=2))


@app.route("/final")
def final():
    guard = _require_user()
    if guard:
        return guard
    users, rec = get_user_record()
    override = load_fft_override(session["username"])

    all_decisions = (
        [d for d in rec.get("part1", []) if d.get("choice") in ("A", "B")]
        + [d for d in rec.get("part2", []) if d.get("choice") in ("A", "B")]
    )
    if len(all_decisions) < 6:
        return render_template("not_ready.html", n_decisions=len(all_decisions))

    fft, nodes_df, stats, feat_names, error = train_fft_cached(all_decisions, FEATURES, override)
    if error or stats is None:
        return render_template("not_ready.html", n_decisions=len(all_decisions))

    svg = fft_svg_explained(
        tree=stats["tree"],
        palette=DEFAULT_FFT_PALETTE,
        node_explanations=stats.get("node_explanations"),
        summary_explanation=stats.get("summary_explanation"),
    )
    return render_template("final.html", svg=svg)


if __name__ == "__main__":
    app.run(debug=True)