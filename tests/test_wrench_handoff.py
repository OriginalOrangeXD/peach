"""Gate 4: CalculiX reaction moment vs Pinocchio joint torque."""

from __future__ import annotations

from pathlib import Path

import pytest

from armpipe.deck import write_link_static_deck
from armpipe.dynamics import load_pinocchio, single_link_static_My
from armpipe.geometry import build_all_links, build_link
from armpipe.meshing import mesh_solid
from armpipe.params import load_params
from armpipe.results import parse_dat_forces, parse_frd
from armpipe.solver import run_ccx
from armpipe.units import moment_Nmm
from armpipe.urdf import write_urdf
from armpipe.validate import gate_static_torque, reaction_moment_Nmm

SINGLE = Path(__file__).resolve().parent / "fixtures" / "single_link.yaml"


def test_ccx_reaction_matches_pinocchio_mgr(tmp_path: Path):
    params = load_params(SINGLE)
    solids = build_all_links(params)
    urdf = write_urdf(params, solids, tmp_path / "one.urdf")
    model, data = load_pinocchio(urdf, params)
    my_pin, _ = single_link_static_My(model, data, "joint1")

    solid = build_link(params, "link1")
    mesh = mesh_solid(solid, params, tmp_path)
    inp = write_link_static_deck(mesh, params, tmp_path, wrench=None, gravity=True)
    frd_path, dat_path = run_ccx(inp, tmp_path)
    frd = parse_frd(frd_path)
    rf = parse_dat_forces(dat_path, "Nmount")
    mx, my, mz = reaction_moment_Nmm(frd.nodes, rf, (0.0, 0.0, 0.0))
    print(f"\nPinocchio My={my_pin:.6f} N·m  CCX My={my/1000:.6f} N·m  RF={sum(v[2] for v in rf.values()):.4f} N")
    gate_static_torque(moment_Nmm(my), moment_Nmm(my_pin * 1000.0), tolerance=0.02)
