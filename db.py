"""
SQL storage layer for the Preference Elicitation Portal.
SURA 2026 · IIT Delhi

Replaces the previous `users.json` flat-file store with a SQLite database
(`portal.db`). The schema is a direct relational translation of the JSON
document that used to live in users.json:

    users.json                              SQL
    ─────────────────────────────────────   ─────────────────────────────────
    <username>                              users.username           (PK)
    <username>.part1_index                  users.part1_index
    <username>.part2_index                  users.part2_index
    <username>.alignment_score              users.alignment_score
    <username>.fft_override                 users.fft_override       (JSON text)
    <username>.part1_scenarios[i]           scenarios(part=1, position=i)
    <username>.part2_scenarios[i]           scenarios(part=2, position=i)
    <username>.part1[i]                     decisions(part=1, seq=i)
    <username>.part2[i]                     decisions(part=2, seq=i)

`load_all_users()` reconstructs *exactly* the same nested dict the JSON file
used to yield, and `save_user_record()` / `save_all_users()` write it back, so
every caller in app.py keeps working against an unchanged data shape.

Only the stdlib `sqlite3` module is used — no new dependency.
"""

import json
import os
import sqlite3

APP_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.environ.get("DB_PATH") or os.path.join(APP_DIR, "portal.db")
LEGACY_JSON  = os.path.join(APP_DIR, "users.json")

# Set by init_db() — the parameter list the A_/B_ columns are built from.
_FEATURES = []


# ── connection ────────────────────────────────────────────────────────────────

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _cols(prefix):
    """Column name list for one side of a pair, e.g. A_age, A_dependents, ..."""
    return [f"{prefix}_{p}" for p in _FEATURES]


# ── schema ────────────────────────────────────────────────────────────────────

def init_db(features):
    """
    Create the schema if it does not exist, then import users.json once if the
    database is still empty (so existing participant data is preserved).

    `features` is the app's FEATURES list; the per-parameter A_/B_ columns are
    generated from it so the schema always matches the study's parameter set.
    """
    global _FEATURES
    _FEATURES = list(features)

    pair_cols = ",\n            ".join(
        f"{c} REAL" for c in (_cols("A") + _cols("B"))
    )

    with _connect() as conn:
        conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS users (
            username        TEXT PRIMARY KEY,
            part1_index     INTEGER NOT NULL DEFAULT 0,
            part2_index     INTEGER NOT NULL DEFAULT 0,
            alignment_score INTEGER,
            fft_override    TEXT
        );

        CREATE TABLE IF NOT EXISTS scenarios (
            username  TEXT    NOT NULL,
            part      INTEGER NOT NULL,
            position  INTEGER NOT NULL,
            {pair_cols},
            PRIMARY KEY (username, part, position),
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    NOT NULL,
            part      INTEGER NOT NULL,
            seq       INTEGER NOT NULL,
            scenario  INTEGER NOT NULL,
            choice    TEXT    NOT NULL,
            timestamp TEXT    NOT NULL,
            {pair_cols},
            FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_decisions_user_part
            ON decisions(username, part, seq);
        """)

    _migrate_legacy_json()


def _migrate_legacy_json():
    """One-time import of users.json, only when the users table is still empty."""
    with _connect() as conn:
        already = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    if already or not os.path.exists(LEGACY_JSON):
        return
    try:
        with open(LEGACY_JSON) as f:
            legacy = json.load(f)
    except Exception as e:
        print(f"[db] Could not read {LEGACY_JSON} for migration: {e}")
        return
    if not isinstance(legacy, dict) or not legacy:
        return
    save_all_users(legacy)
    print(f"[db] Migrated {len(legacy)} user record(s) from users.json into {DB_PATH}")


# ── read ──────────────────────────────────────────────────────────────────────

def _scenario_from_row(row):
    return {
        "A": {p: row[f"A_{p}"] for p in _FEATURES},
        "B": {p: row[f"B_{p}"] for p in _FEATURES},
    }


def _decision_from_row(row):
    """Rebuild the decision dict in the same key order the JSON store used."""
    d = {
        "username":  row["username"],
        "scenario":  row["scenario"],
        "choice":    row["choice"],
        "timestamp": row["timestamp"],
    }
    for p in _FEATURES:
        d[f"A_{p}"] = row[f"A_{p}"]
        d[f"B_{p}"] = row[f"B_{p}"]
    return d


def load_all_users():
    """Return the full store in the exact shape users.json used to hold."""
    out = {}
    with _connect() as conn:
        for u in conn.execute("SELECT * FROM users ORDER BY username"):
            name = u["username"]
            rec = {}

            for part in (1, 2):
                scen = [
                    _scenario_from_row(r)
                    for r in conn.execute(
                        "SELECT * FROM scenarios WHERE username=? AND part=? "
                        "ORDER BY position", (name, part))
                ]
                dec = [
                    _decision_from_row(r)
                    for r in conn.execute(
                        "SELECT * FROM decisions WHERE username=? AND part=? "
                        "ORDER BY seq", (name, part))
                ]
                if scen:
                    rec[f"part{part}_scenarios"] = scen
                    rec[f"part{part}"] = dec
                    rec[f"part{part}_index"] = u[f"part{part}_index"]

            if u["alignment_score"] is not None:
                rec["alignment_score"] = u["alignment_score"]
            if u["fft_override"]:
                rec["fft_override"] = json.loads(u["fft_override"])

            out[name] = rec
    return out


# ── write ─────────────────────────────────────────────────────────────────────

def save_user_record(username, record):
    """
    Persist one participant's record. The record dict is the source of truth —
    the user's scenarios and decisions are rewritten to match it, mirroring the
    whole-document overwrite the JSON store performed on every save.
    """
    with _connect() as conn:
        _write_user(conn, username, record)


def save_all_users(users):
    """Persist the whole store (kept so app.save_users() keeps its signature)."""
    with _connect() as conn:
        for username, record in users.items():
            _write_user(conn, username, record or {})


def _write_user(conn, username, record):
    override = record.get("fft_override")
    conn.execute(
        """
        INSERT INTO users (username, part1_index, part2_index,
                           alignment_score, fft_override)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET
            part1_index     = excluded.part1_index,
            part2_index     = excluded.part2_index,
            alignment_score = excluded.alignment_score,
            fft_override    = excluded.fft_override
        """,
        (
            username,
            int(record.get("part1_index", 0) or 0),
            int(record.get("part2_index", 0) or 0),
            record.get("alignment_score"),
            json.dumps(override) if override else None,
        ),
    )

    scen_cols = _cols("A") + _cols("B")
    for part in (1, 2):
        scenarios = record.get(f"part{part}_scenarios")
        if scenarios is not None:
            conn.execute("DELETE FROM scenarios WHERE username=? AND part=?",
                         (username, part))
            conn.executemany(
                f"INSERT INTO scenarios (username, part, position, "
                f"{', '.join(scen_cols)}) VALUES "
                f"(?, ?, ?, {', '.join('?' * len(scen_cols))})",
                [
                    (username, part, i)
                    + tuple(float(sc["A"][p]) for p in _FEATURES)
                    + tuple(float(sc["B"][p]) for p in _FEATURES)
                    for i, sc in enumerate(scenarios)
                ],
            )

        decisions = record.get(f"part{part}")
        if decisions is not None:
            conn.execute("DELETE FROM decisions WHERE username=? AND part=?",
                         (username, part))
            conn.executemany(
                f"INSERT INTO decisions (username, part, seq, scenario, choice, "
                f"timestamp, {', '.join(scen_cols)}) VALUES "
                f"(?, ?, ?, ?, ?, ?, {', '.join('?' * len(scen_cols))})",
                [
                    (username, part, i, int(d.get("scenario", i + 1)),
                     d.get("choice"), d.get("timestamp"))
                    + tuple(float(d.get(f"A_{p}", 0.0)) for p in _FEATURES)
                    + tuple(float(d.get(f"B_{p}", 0.0)) for p in _FEATURES)
                    for i, d in enumerate(decisions)
                ],
            )


# ── targeted helpers (fft_override) ───────────────────────────────────────────

def set_fft_override(username, tree_dict):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, fft_override) VALUES (?, ?) "
            "ON CONFLICT(username) DO UPDATE SET fft_override = excluded.fft_override",
            (username, json.dumps(tree_dict)),
        )


def get_fft_override(username):
    with _connect() as conn:
        row = conn.execute(
            "SELECT fft_override FROM users WHERE username=?", (username,)
        ).fetchone()
    return json.loads(row["fft_override"]) if row and row["fft_override"] else None


def delete_fft_override(username):
    with _connect() as conn:
        conn.execute("UPDATE users SET fft_override=NULL WHERE username=?", (username,))
