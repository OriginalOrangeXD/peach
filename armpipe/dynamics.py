"""Pinocchio inverse dynamics and worst-case joint wrenches.

Frame convention
----------------
After ``pin.rnea(model, data, q, v, a)``:

* ``data.f[i]`` is the spatial force **transmitted across joint i**,
  expressed in the **LOCAL frame of joint i** (Pinocchio's joint frame,
  which is the URDF joint origin: translation ``origin_xyz_m``, rotation
  ``origin_rpy``, in the parent link).
* ``data.f[i].linear`` is force (N). ``data.f[i].angular`` is moment (N·m).
* Index ``i`` matches ``model.names[i]``. ``i = 0`` is the universe and
  is ignored.
* This is **not** the world frame and **not** the COM frame.

The single-link static test locks this: a horizontal link along +X, joint
axis +Y, gravity −Z, must give ``My = m * g * x_com`` to 1 %, with
``|Mx|, |Mz|, |Fx|, |Fy|`` negligible next to the bending pair
``(My, Fz)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pinocchio as pin

from armpipe.params import require
from armpipe.units import (
    G_SI,
    Kind,
    UnitSystem,
    Wrench,
    force_N,
    moment_Nm,
    vec3,
)


class DynamicsError(ValueError):
    """Stage 4 failure."""


@dataclass
class JointWrench:
    joint: str
    body: str
    wrench: Wrench
    q: tuple[float, ...]
    reason: str


@dataclass
class DynamicsResult:
    model: Any
    data: Any
    total_mass_kg: float
    worst: dict[str, JointWrench] = field(default_factory=dict)
    full_extension: dict[str, JointWrench] = field(default_factory=dict)


def load_pinocchio(urdf_path: Path, params: dict[str, Any]) -> tuple[Any, Any]:
    """Load URDF in SI. Raises if Pinocchio cannot build the model."""
    try:
        model = pin.buildModelFromUrdf(str(urdf_path))
    except Exception as exc:
        raise DynamicsError(f"stage=dynamics Pinocchio failed to load {urdf_path}: {exc}") from exc
    gx, gy, gz = (float(v) for v in params.get("gravity_m_s2", (0.0, 0.0, -G_SI)))
    model.gravity.linear = np.array([gx, gy, gz])
    data = model.createData()
    return model, data


def _force_at(data, index: int, joint_name: str) -> Wrench:
    f = data.f[index]
    return Wrench(
        force=vec3(tuple(float(x) for x in f.linear), Kind.FORCE, UnitSystem.SI),
        moment=vec3(tuple(float(x) for x in f.angular), Kind.MOMENT, UnitSystem.SI),
        frame=f"joint:{joint_name}:local",
    )


def rnea_wrenches(model, data, q, v=None, a=None) -> dict[str, Wrench]:
    """Inverse dynamics. Returns per-joint wrenches in the local joint frame."""
    q = np.asarray(q, dtype=float)
    v = np.zeros(model.nv) if v is None else np.asarray(v, dtype=float)
    a = np.zeros(model.nv) if a is None else np.asarray(a, dtype=float)
    pin.rnea(model, data, q, v, a)
    out: dict[str, Wrench] = {}
    for i, name in enumerate(model.names):
        if i == 0:
            continue
        out[name] = _force_at(data, i, name)
    return out


def _moment_norm(w: Wrench) -> float:
    mx, my, mz = w.moment.as_array_si()
    return float(np.hypot(np.hypot(mx, my), mz))


def _grid(params: dict[str, Any], model) -> Iterable[np.ndarray]:
    n = int(require(params, "dynamics", "n_grid"))
    lowers = []
    uppers = []
    for node in require(params, "tree", "links"):
        joint = node.get("joint") or {}
        if joint.get("type") != "revolute":
            continue
        lowers.append(float(joint["limit_lower"]))
        uppers.append(float(joint["limit_upper"]))
    if len(lowers) != model.nq:
        raise DynamicsError(
            f"stage=dynamics revolute count {len(lowers)} != model.nq {model.nq}"
        )
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(lowers, uppers)]
    if not axes:
        yield np.zeros(0)
        return
    meshes = np.meshgrid(*axes, indexing="ij")
    for qs in zip(*(m.ravel() for m in meshes)):
        yield np.asarray(qs, dtype=float)
    yield np.zeros(model.nq)  # full extension (q=0) always included


def sweep_worst_wrenches(model, data, params: dict[str, Any]) -> DynamicsResult:
    """Coarse joint grid + q=0. Keep the pose with the largest |M| per joint."""
    worst: dict[str, JointWrench] = {}
    full: dict[str, JointWrench] = {}
    q0 = np.zeros(model.nq)
    for name, w in rnea_wrenches(model, data, q0).items():
        body = model.names[list(model.names).index(name)]
        full[name] = JointWrench(name, body, w, tuple(float(x) for x in q0), "full_extension_q0")
        worst[name] = full[name]
    for q in _grid(params, model):
        wrenches = rnea_wrenches(model, data, q)
        qt = tuple(float(x) for x in q)
        for name, w in wrenches.items():
            if _moment_norm(w) > _moment_norm(worst[name].wrench):
                worst[name] = JointWrench(name, name, w, qt, "workspace_max_|M|")
    return DynamicsResult(
        model=model,
        data=data,
        total_mass_kg=float(pin.computeTotalMass(model)),
        worst=worst,
        full_extension=full,
    )


def single_link_static_My(model, data, joint_name: str = "joint1") -> tuple[float, float]:
    """Return (My_pinocchio, m*g*x_com) in N·m for a horizontal 1-DOF link.

    ``x_com`` is the subtree COM X in the named joint's LOCAL frame at q=0.
    ``m`` is that subtree mass. Gravity is taken from ``model.gravity``.
    """
    if model.nq < 1:
        raise DynamicsError("stage=dynamics single-link test needs nq >= 1")
    jid = int(model.getJointId(joint_name))
    q = np.zeros(model.nq)
    pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))
    my = float(data.f[jid].angular[1])
    pin.centerOfMass(model, data, q)
    pin.computeSubtreeMasses(model, data)
    com_world = np.asarray(data.com[jid], dtype=float)
    com_local = np.asarray(data.oMi[jid].actInv(com_world), dtype=float)
    mass = float(data.mass[jid])
    # Parent-on-child convention (observed in Pinocchio 4.1): the joint
    # holds the subtree up, so Fz > 0 when g is −Z, and
    # M = r × F = (x, 0, 0) × (0, 0, +mg) = (0, −mg x, 0).
    g = abs(float(model.gravity.linear[2]))
    hand = -mass * g * float(com_local[0])
    return my, hand
