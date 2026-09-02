"""
Export every table from a SQLite database to CSV files.

Usage:
    python scripts/export_data.py [path/to/portal.db]

Defaults to ../portal.db relative to this script if no path is given.
Output CSVs land in ./export/ (created if absent).
The source database is opened read-only; nothing is written to it.
"""

import csv
import os
import sqlite3
import sys
from datetime import datetime


def export(db_path: str, out_dir: str = "export") -> None:
    if not os.path.exists(db_path):
        sys.exit(f"Error: {db_path!r} not found.")

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]

    if not tables:
        print("No tables found.")
        conn.close()
        return

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for table in tables:
        rows = conn.execute(f"SELECT * FROM [{table}]").fetchall()
        out_path = os.path.join(out_dir, f"{table}_{stamp}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if rows:
                writer.writerow(rows[0].keys())
                writer.writerows(rows)
            else:
                writer.writerow([])
        print(f"  {table:20s} → {out_path}  ({len(rows)} rows)")

    conn.close()
    print(f"\nDone. {len(tables)} table(s) exported from {db_path!r}")


if __name__ == "__main__":
    default_db = os.path.join(os.path.dirname(__file__), "..", "portal.db")
    db_path = sys.argv[1] if len(sys.argv) > 1 else default_db
    export(os.path.abspath(db_path))
