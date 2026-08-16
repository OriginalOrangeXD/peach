import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDA_BIN = ROOT / ".conda" / "bin"
if CONDA_BIN.is_dir():
    os.environ["PATH"] = str(CONDA_BIN) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CCX", str(CONDA_BIN / "ccx"))
