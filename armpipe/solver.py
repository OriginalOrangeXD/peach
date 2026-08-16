"""Run CalculiX (ccx) as a subprocess. No GUI."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class SolverError(RuntimeError):
    """ccx missing, non-zero exit, or missing result files."""


def find_ccx() -> Path:
    env = os.environ.get("CCX")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    which = shutil.which("ccx")
    if which:
        return Path(which)
    local = Path(__file__).resolve().parents[1] / ".conda" / "bin" / "ccx"
    if local.is_file():
        return local
    raise SolverError("ccx not on PATH; set CCX= or install calculix in .conda")


def run_ccx(inp_path: Path, workdir: Path | None = None) -> tuple[Path, Path]:
    """Run ``ccx job`` in ``workdir``. Returns ``(frd, dat)``."""
    inp_path = Path(inp_path).resolve()
    workdir = Path(workdir) if workdir is not None else inp_path.parent
    job = inp_path.stem
    if inp_path.parent != workdir.resolve():
        dest = workdir / inp_path.name
        if dest.resolve() != inp_path:
            dest.write_text(inp_path.read_text())
    ccx = find_ccx()
    proc = subprocess.run(
        [str(ccx), job],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    log = workdir / f"{job}.ccx.log"
    log.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
    frd = workdir / f"{job}.frd"
    dat = workdir / f"{job}.dat"
    if proc.returncode != 0 or not frd.is_file():
        tail = (proc.stdout or "")[-2000:]
        raise SolverError(
            f"stage=solver ccx exit={proc.returncode} job={job}\n{tail}"
        )
    return frd, dat
