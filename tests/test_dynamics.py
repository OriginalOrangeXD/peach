"""Pinocchio RNEA frame convention: My = m*g*x_com for a horizontal link."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from armpipe.dynamics import load_pinocchio, single_link_static_My, sweep_worst_wrenches
from armpipe.geometry import build_all_links
from armpipe.params import load_params
from armpipe.urdf import write_urdf
from armpipe.validate import gate_rnea_mgr

SINGLE = Path(__file__).resolve().parent / "fixtures" / "single_link.yaml"
ARM = Path(__file__).resolve().parents[1] / "armpipe" / "params.yaml"


def test_single_link_static_matches_mgr(tmp_path: Path):
    params = load_params(SINGLE)
    solids = build_all_links(params)
    urdf = write_urdf(params, solids, tmp_path / "one.urdf")
    model, data = load_pinocchio(urdf, params)
    print("\njoints", list(model.names), "nq", model.nq)
    my, hand = single_link_static_My(model, data, "joint1")
    print(f"My_pin={my:.6f} N·m  m*g*r={hand:.6f} N·m")
    gate_rnea_mgr(my, hand, tolerance=0.01)
    # Bending pair: moment about +Y, force about -Z. Other components small.
    import pinocchio as pin

    pin.rnea(model, data, np.zeros(model.nq), np.zeros(model.nv), np.zeros(model.nv))
    jid = model.getJointId("joint1")
    f = data.f[jid]
    assert abs(f.angular[1]) > 10.0 * max(abs(f.angular[0]), abs(f.angular[2]), 1e-9)
    assert f.linear[2] > 0  # parent-on-child: joint holds the link up
    assert my < 0  # r × Fz_up = −Y for x_com > 0
    assert my * hand > 0


def test_arm_sweep_returns_local_frames(tmp_path: Path):
    params = load_params(ARM)
    solids = build_all_links(params)
    urdf = write_urdf(params, solids, tmp_path / "arm.urdf")
    model, data = load_pinocchio(urdf, params)
    result = sweep_worst_wrenches(model, data, params)
    assert "joint1" in result.worst
    w = result.worst["joint1"].wrench
    assert w.frame == "joint:joint1:local"
    assert w.system.name == "SI"
    # Full extension is the gravity-worst pose for this Y-axis arm.
    assert result.worst["joint1"].reason in {"full_extension_q0", "workspace_max_|M|"}
    assert _moment(w) > 0


def _moment(w) -> float:
    mx, my, mz = w.moment.as_array_si()
    return float(np.hypot(np.hypot(mx, my), mz))
