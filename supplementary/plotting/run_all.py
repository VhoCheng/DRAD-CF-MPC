"""Reproduce all four new experimental figures with the bundled CSV data."""
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parent
for name in ("generate_publication_plots.py", "generate_supplement_publication_plots.py"):
    subprocess.run([sys.executable, str(root / "scripts" / name)], check=True)
print("Done. PDF, PNG and data-audit JSON files are in figures/new/.")
