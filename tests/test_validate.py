"""Validation gates, exercised on the cantilever fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from armpipe.deck import write_cantilever_deck, write_frequency_deck
from armpipe.geometry import build_cantilever
from armpipe.meshing import mesh_solid
from armpipe.params import load_params
from armpipe.results import extract_static_scalars, parse_dat_frequencies, parse_frd
from armpipe.solver import run_ccx
from armpipe.units import frequency_Hz, moment_Nmm, stress_MPa
from armpipe.validate import (
    GateError,
    cantilever_hand_torque_Nm,
    gate_free_free,
    gate_mass,
    gate_mesh_convergence,
    gate_rigid_body_constrained,
    gate_static_torque,
    reaction_moment_Nmm,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cantilever.yaml"


@pytest.fixture(scope="module")
def cantilever_mesh(tmp_path_factory):
    params = load_params(FIXTURE)
    work = tmp_path_factory.mktemp("cantilever_gates")
    solid = build_cantilever(params)
    mesh = mesh_solid(solid, params, work)
    return params, solid, mesh, work


def test_gate_mass_matches_box_from_params(cantilever_mesh):
    params, solid, _, _ = cantilever_mesh
    mass = gate_mass(solid, params)
    assert mass.as_kg() == pytest.approx(0.108, rel=0.01)


def test_gate_mass_raises_on_disagreement():
    params = load_params(FIXTURE)
    solid = build_cantilever(params)
    bad = dict(params)
    bad["geometry"] = dict(params["geometry"], volume_mm3_hand=1.0)
    with pytest.raises(GateError, match="validate.mass"):
        gate_mass(solid, bad)


def test_gate_static_torque_and_reaction_moment(cantilever_mesh):
    params, solid, mesh, work = cantilever_mesh
    inp = write_cantilever_deck(mesh, params, work)
    frd_path, dat_path = run_ccx(inp, work)
    frd = parse_frd(frd_path)
    scalars = extract_static_scalars(
        frd, dat_path, "Nref", (0.0, 0.0, -1.0)
    )
    from armpipe.results import parse_dat_forces

    rf = parse_dat_forces(dat_path, "Nfixed")
    mx, my, mz = reaction_moment_Nmm(frd.nodes, rf, (0.0, 0.0, 0.0))
    M_ccx = moment_Nmm(abs(my))
    M_ref = cantilever_hand_torque_Nm(params)
    gate_static_torque(M_ccx, M_ref, tolerance=0.02)
    assert M_ref.as_Nm() == pytest.approx(10.0)
    assert scalars.reaction_force_N[2] == pytest.approx(50.0, rel=0.01)


def test_gate_free_free_and_constrained_modes(cantilever_mesh):
    params, _, mesh, work = cantilever_mesh
    free_inp = write_frequency_deck(mesh, params, work, constrained=False, n_modes=8)
    _, free_dat = run_ccx(free_inp, work)
    free_f = parse_dat_frequencies(free_dat)
    print("\nfree-free Hz", [q.as_Hz() for q in free_f])
    gate_free_free(free_f)

    cons_inp = write_frequency_deck(mesh, params, work, constrained=True, n_modes=5)
    _, cons_dat = run_ccx(cons_inp, work)
    cons_f = parse_dat_frequencies(cons_dat)
    print("constrained Hz", [q.as_Hz() for q in cons_f])
    f1 = gate_rigid_body_constrained(cons_f, min_Hz=1.0)
    # Euler-Bernoulli cantilever f1 ≈ 206 Hz for this fixture.
    assert f1 == pytest.approx(206.4, rel=0.05)


def test_gate_mesh_convergence_synthetic():
    gate_mesh_convergence(stress_MPa(30.0), stress_MPa(30.9), tolerance=0.05)
    with pytest.raises(GateError, match="mesh_convergence"):
        gate_mesh_convergence(stress_MPa(30.0), stress_MPa(34.0), tolerance=0.05)


def test_gate_free_free_rejects_nonzero_rigid_modes():
    with pytest.raises(GateError, match="free_free"):
        gate_free_free([frequency_Hz(x) for x in (0.0, 0.0, 12.0, 0.0, 0.0, 0.0)])


def test_frequency_parser_uses_cycles_per_time(tmp_path: Path):
    dat = tmp_path / "modal.dat"
    dat.write_text(
        """
     E I G E N V A L U E   O U T P U T

 MODE NO    EIGENVALUE                       FREQUENCY
                                     REAL PART            IMAGINARY PART
                           (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)

      1   0.1681799E+07   0.1296842E+04   0.2063988E+03   0.0000000E+00
      2   0.6622557E+07   0.2573433E+04   0.4095746E+03   0.0000000E+00
"""
    )
    freqs = parse_dat_frequencies(dat)
    assert freqs[0].as_Hz() == pytest.approx(206.3988)
    assert freqs[1].as_Hz() == pytest.approx(409.5746)
