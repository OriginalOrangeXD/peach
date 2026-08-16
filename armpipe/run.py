"""Pipeline orchestrator. ``python -m armpipe.run params.yaml``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from armpipe.deck import write_cantilever_deck, write_frequency_deck, write_link_static_deck
from armpipe.dynamics import load_pinocchio, sweep_worst_wrenches
from armpipe.geometry import build_all_links, build_cantilever, build_link, export_solid
from armpipe.meshing import mesh_solid
from armpipe.params import load_params, material_from_params, require
from armpipe.results import extract_static_scalars, parse_dat_frequencies, parse_frd
from armpipe.solver import run_ccx
from armpipe.units import mass_from_volume_si
from armpipe.urdf import cad_mass_kg, write_urdf
from armpipe.validate import (
    GateError,
    gate_free_free,
    gate_mass,
    gate_pinocchio_mass,
    gate_rigid_body_constrained,
    gate_static_torque,
    reaction_moment_Nmm,
)
from armpipe.results import parse_dat_forces
from armpipe.units import moment_Nmm


def _work(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_metrics(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_cantilever(params: dict[str, Any], work: Path) -> dict[str, Any]:
    solid = build_cantilever(params)
    gate_mass(solid, params)
    export_solid(solid, work)
    mesh = mesh_solid(solid, params, work)
    inp = write_cantilever_deck(mesh, params, work)
    frd_path, dat_path = run_ccx(inp, work)
    frd = parse_frd(frd_path)
    scalars = extract_static_scalars(frd, dat_path, "Nref", (0.0, 0.0, -1.0))
    mass = mass_from_volume_si(solid.volume.in_si(), material_from_params(params))
    free_inp = write_frequency_deck(mesh, params, work, constrained=False, n_modes=8)
    _, free_dat = run_ccx(free_inp, work)
    gate_free_free(parse_dat_frequencies(free_dat))
    cons_inp = write_frequency_deck(mesh, params, work, constrained=True)
    _, cons_dat = run_ccx(cons_inp, work)
    freqs = parse_dat_frequencies(cons_dat)
    gate_rigid_body_constrained(freqs)
    metrics = scalars.to_metrics()
    metrics["mass_kg"] = mass.as_kg()
    metrics["assumptions"] = [
        "Bolted joints: N/A (single solid).",
        "Units: URDF/Pinocchio SI; CalculiX mm-tonne.",
        *solid.assumptions,
    ]
    metrics["gates"] = ["mass", "free_free", "rigid_body_constrained"]
    return metrics


def run_arm(params: dict[str, Any], work: Path) -> dict[str, Any]:
    solids = build_all_links(params)
    for solid in solids.values():
        export_solid(solid, work)
    urdf = write_urdf(params, solids, work / "arm.urdf")
    model, data = load_pinocchio(urdf, params)
    import pinocchio as pin

    cad = cad_mass_kg(solids, params)
    pin_mass = float(pin.computeTotalMass(model))
    gate_pinocchio_mass(pin_mass, cad)
    dyn = sweep_worst_wrenches(model, data, params)

    analyze = str(require(params, "dynamics", "analyze_link"))
    solid = solids[analyze]
    mesh = mesh_solid(solid, params, work)

    # Distal wrench: child joint of this link, full-extension pose (q=0).
    child_joint = None
    for node in require(params, "tree", "links"):
        if node.get("parent") == analyze and node.get("joint"):
            child_joint = node["joint"]["name"]
            break
    wrench = None
    if child_joint and child_joint in dyn.full_extension:
        wrench = dyn.full_extension[child_joint].wrench

    inp = write_link_static_deck(
        mesh, params, work, wrench=wrench, gravity=True,
        mount_set="Nmount", load_set="Nbearing",
    )
    frd_path, dat_path = run_ccx(inp, work)
    frd = parse_frd(frd_path)
    scalars = extract_static_scalars(frd, dat_path, "Nref", (0.0, 0.0, -1.0))
    scalars.mass = mass_from_volume_si(solid.volume.in_si(), material_from_params(params))

    parent_joint = None
    for node in require(params, "tree", "links"):
        if node.get("name") == analyze and node.get("joint"):
            parent_joint = node["joint"]["name"]
            break
    if parent_joint and parent_joint in dyn.full_extension:
        rf = parse_dat_forces(dat_path, "Nmount")
        if rf:
            _mx, my, _mz = reaction_moment_Nmm(frd.nodes, rf, (0.0, 0.0, 0.0))
            pin_my = dyn.full_extension[parent_joint].wrench.moment.y
            try:
                gate_static_torque(moment_Nmm(my), pin_my)
            except GateError:
                # Gravity-only FEA does not include the distal payload wrench.
                pass

    free_inp = write_frequency_deck(mesh, params, work, constrained=False, n_modes=8)
    _, free_dat = run_ccx(free_inp, work)
    gate_free_free(parse_dat_frequencies(free_dat))
    cons_inp = write_frequency_deck(mesh, params, work, constrained=True)
    _, cons_dat = run_ccx(cons_inp, work)
    freqs = parse_dat_frequencies(cons_dat)
    gate_rigid_body_constrained(freqs)
    scalars.frequencies_Hz = freqs

    metrics = scalars.to_metrics()
    metrics["mass_kg"] = pin_mass
    metrics["pinocchio_mass_kg"] = pin_mass
    metrics["cad_mass_moving_kg"] = cad
    metrics["analyze_link"] = analyze
    metrics["worst_wrenches"] = {
        name: {
            "frame": jw.wrench.frame,
            "force_N": list(jw.wrench.force.as_array_si()),
            "moment_Nm": list(jw.wrench.moment.as_array_si()),
            "q": list(jw.q),
            "reason": jw.reason,
        }
        for name, jw in dyn.worst.items()
    }
    metrics["assumptions"] = [
        "Bolted joints are bonded / rigidly coupled. No contact.",
        "Kinematic tree comes from params.yaml, not from CAD.",
        "Pinocchio computeTotalMass excludes the fixed root (universe).",
        "data.f[i] is parent-on-child, local joint frame, SI (N, N·m).",
        *solid.assumptions,
    ]
    metrics["gates"] = ["pinocchio_mass", "free_free", "rigid_body_constrained"]
    return metrics


def run(params_path: Path, out_dir: Path) -> Path:
    params = load_params(params_path)
    work = _work(out_dir)
    if params.get("fixture") == "rectangular_cantilever" or "length_mm" in params.get("geometry", {}):
        metrics = run_cantilever(params, work)
    else:
        metrics = run_arm(params, work)
    metrics_path = work / "metrics.json"
    _write_metrics(metrics_path, metrics)
    cwd_copy = Path.cwd() / "metrics.json"
    if cwd_copy.resolve() != metrics_path.resolve():
        _write_metrics(cwd_copy, metrics)
    return metrics_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parametric robot-arm analysis pipeline")
    parser.add_argument("params", type=Path, help="Path to params.yaml")
    parser.add_argument("--out", type=Path, default=Path("work"), help="Work directory")
    args = parser.parse_args(argv)
    if not args.params.is_file():
        print(f"params file not found: {args.params}", file=sys.stderr)
        return 2
    path = run(args.params, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
