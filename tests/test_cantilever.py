"""Closed-form cantilever: geometry → mesh → deck → ccx → parse.

Textbook (Euler-Bernoulli, rectangular section, end load F, length L):

    I = b h³ / 12
    δ = F L³ / (3 E I)
    σ = M c / I = F L (h/2) / I

If the pipeline disagrees, the pipeline is wrong, not the textbook.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from armpipe.deck import write_cantilever_deck
from armpipe.geometry import build_cantilever, export_solid, validate_solid
from armpipe.meshing import mesh_solid
from armpipe.params import load_params, material_from_params, require
from armpipe.results import extract_static_scalars, parse_frd
from armpipe.solver import run_ccx
from armpipe.units import length_mm, stress_MPa

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cantilever.yaml"


def _beam_theory(params: dict) -> tuple[float, float]:
    """Return (δ_mm, σ_MPa) in mm-tonne numbers."""
    L = float(require(params, "geometry", "length_mm"))
    b = float(require(params, "geometry", "width_mm"))
    h = float(require(params, "geometry", "height_mm"))
    F = abs(float(require(params, "load", "force_N")[2]))
    E = material_from_params(params).in_mm_tonne().E.as_MPa()
    I = b * h**3 / 12.0
    delta_mm = F * L**3 / (3.0 * E * I)
    sigma_MPa = F * L * (h / 2.0) / I
    return delta_mm, sigma_MPa


def test_cantilever_closed_form(tmp_path: Path):
    params = load_params(FIXTURE)
    delta_th, sigma_th = _beam_theory(params)
    solid = build_cantilever(params)
    validate_solid(solid)
    export_solid(solid, tmp_path)
    assert (tmp_path / f"{solid.name}.step").stat().st_size > 0

    mesh = mesh_solid(solid, params, tmp_path)
    assert mesh.n_elements > 0
    assert "Nfixed" in mesh.node_sets and mesh.node_sets["Nfixed"]
    assert "Nload" in mesh.node_sets and mesh.node_sets["Nload"]

    inp = write_cantilever_deck(mesh, params, tmp_path)
    frd_path, dat_path = run_ccx(inp, tmp_path)
    frd = parse_frd(frd_path)
    h = float(require(params, "geometry", "height_mm"))

    scalars = extract_static_scalars(
        frd,
        dat_path,
        tip_nset="Nref",
        load_direction=(0.0, 0.0, -1.0),
        fiber_point_mm=(0.0, 0.0, h / 2.0),
    )

    delta = scalars.tip_deflection.as_mm()
    sigma = scalars.max_von_mises.as_MPa()
    print(
        f"\ncantilever: δ={delta:.4f} mm (theory {delta_th:.4f}), "
        f"σ_max={sigma:.3f} MPa (theory {sigma_th:.3f}), "
        f"elems={mesh.n_elements}, loc={scalars.max_von_mises_xyz_mm}, "
        f"RF={scalars.reaction_force_N}"
    )
    assert delta == pytest.approx(delta_th, rel=0.05), (
        f"tip deflection {delta:.4f} mm vs theory {delta_th:.4f} mm"
    )
    assert sigma == pytest.approx(sigma_th, rel=0.10), (
        f"max VM {sigma:.3f} MPa vs theory {sigma_th:.3f} MPa "
        f"at {scalars.max_von_mises_xyz_mm}"
    )
    assert scalars.reaction_force_N is not None
    fz = float(require(params, "load", "force_N")[2])
    assert scalars.reaction_force_N[2] == pytest.approx(-fz, rel=0.01)


def test_beam_theory_numbers_are_the_handbook_pair():
    params = load_params(FIXTURE)
    delta, sigma = _beam_theory(params)
    assert delta == pytest.approx(1.142857, rel=1e-6)
    assert sigma == pytest.approx(30.0, rel=1e-6)
    assert length_mm(delta).as_mm() == pytest.approx(1.142857, rel=1e-6)
    assert stress_MPa(sigma).as_MPa() == pytest.approx(30.0)
