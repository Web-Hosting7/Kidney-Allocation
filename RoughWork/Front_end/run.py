# run.py — works on Mac, Linux, and Windows
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the venv Python executable
if sys.platform == "win32":
    python = os.path.join(SCRIPT_DIR, "venv", "Scripts", "python.exe")
else:
    python = os.path.join(SCRIPT_DIR, "venv", "bin", "python")

app = os.path.join(SCRIPT_DIR, "app.py")

# Check venv exists
if not os.path.exists(python):
    print("Virtual environment not found. Run this first:")
    print("  python3 -m venv venv")
    print("  venv/bin/pip install -r requirements.txt    # Mac/Linux")
    print("  venv\\Scripts\\pip install -r requirements.txt  # Windows")
    sys.exit(1)

# Launch streamlit using the venv's Python
subprocess.run([python, "-m", "streamlit", "run", app])