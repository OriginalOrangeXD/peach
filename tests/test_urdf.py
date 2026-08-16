"""URDF from params + CAD mass properties, loaded by Pinocchio."""

from __future__ import annotations

from pathlib import Path

import pytest

from armpipe.geometry import build_all_links
from armpipe.params import load_params
from armpipe.urdf import assert_tree_matches_geometry, cad_mass_kg, write_urdf
from armpipe.validate import GateError, gate_pinocchio_mass

ARM = Path(__file__).resolve().parents[1] / "armpipe" / "params.yaml"
SINGLE = Path(__file__).resolve().parent / "fixtures" / "single_link.yaml"


def test_tree_origins_match_link_lengths():
    assert_tree_matches_geometry(load_params(ARM))


def test_mismatched_origin_raises():
    params = load_params(ARM)
    params["tree"]["links"][2]["joint"]["origin_xyz_m"] = [0.111, 0.0, 0.0]
    with pytest.raises(Exception, match="origin_xyz_m"):
        assert_tree_matches_geometry(params)


def test_pinocchio_loads_and_mass_within_2pct(tmp_path: Path):
    import pinocchio as pin

    from armpipe.dynamics import load_pinocchio

    params = load_params(ARM)
    solids = build_all_links(params)
    urdf = write_urdf(params, solids, tmp_path / "arm.urdf")
    model, _ = load_pinocchio(urdf, params)
    cad = cad_mass_kg(solids, params)
    pin_mass = float(pin.computeTotalMass(model))
    gate_pinocchio_mass(pin_mass, cad, tolerance=0.02)
    assert model.nq == 2


def test_single_link_pinocchio_mass(tmp_path: Path):
    import pinocchio as pin

    from armpipe.dynamics import load_pinocchio

    params = load_params(SINGLE)
    solids = build_all_links(params)
    urdf = write_urdf(params, solids, tmp_path / "one.urdf")
    model, _ = load_pinocchio(urdf, params)
    gate_pinocchio_mass(float(pin.computeTotalMass(model)), cad_mass_kg(solids, params))
