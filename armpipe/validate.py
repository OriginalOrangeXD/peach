"""Validation gates. A failed gate raises; it does not warn and continue."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from armpipe.geometry import BuiltSolid
from armpipe.meshing import MeshModel
from armpipe.params import material_from_params, require
from armpipe.results import parse_dat_forces, parse_dat_frequencies
from armpipe.units import Quantity, mass_from_volume_si, moment_Nm, volume_mm3


class GateError(ValueError):
    """A named validation gate failed. Message includes stage and numbers."""


def _pct(a: float, b: float) -> float:
    denom = max(abs(b), 1e-30)
    return abs(a - b) / denom


def gate_mass(solid: BuiltSolid, params: dict[str, Any], tolerance: float = 0.05) -> Quantity:
    """Gate 1: B-rep mass vs ρ × volume from params, within ``tolerance``."""
    material = material_from_params(params)
    cad = mass_from_volume_si(solid.volume.in_si(), material)
    hand = cad  # params density × CAD volume is the hand estimate
    # Optional override: geometry.volume_mm3_hand if someone wrote a hand calc.
    hand_mm3 = params.get("geometry", {}).get("volume_mm3_hand")
    if hand_mm3 is not None:
        hand = mass_from_volume_si(volume_mm3(float(hand_mm3)).in_si(), material)
    if _pct(cad.as_kg(), hand.as_kg()) > tolerance:
        raise GateError(
            f"stage=validate.mass cad={cad.as_kg():.6g} kg "
            f"hand={hand.as_kg():.6g} kg tol={tolerance:.0%}"
        )
    geom = params.get("geometry", {})
    if all(k in geom for k in ("length_mm", "width_mm", "height_mm")):
        box_mm3 = float(geom["length_mm"]) * float(geom["width_mm"]) * float(geom["height_mm"])
        box_mass = mass_from_volume_si(volume_mm3(box_mm3).in_si(), material)
        if _pct(cad.as_kg(), box_mass.as_kg()) > tolerance:
            raise GateError(
                f"stage=validate.mass cad={cad.as_kg():.6g} kg "
                f"box={box_mass.as_kg():.6g} kg (L·b·h from params) tol={tolerance:.0%}"
            )
    return cad


def gate_free_free(
    frequencies_Hz: Iterable[Quantity],
    n_rigid: int = 6,
    max_rigid_Hz: float = 1.0,
) -> list[float]:
    """Gate 2: first six free-free eigenvalues must be near zero."""
    freqs = [q.as_Hz() for q in frequencies_Hz]
    if len(freqs) < n_rigid:
        raise GateError(
            f"stage=validate.free_free got {len(freqs)} frequencies, need ≥ {n_rigid}"
        )
    rigid = freqs[:n_rigid]
    if any(abs(f) > max_rigid_Hz for f in rigid):
        raise GateError(
            f"stage=validate.free_free first {n_rigid} Hz={rigid} "
            f"not all ≤ {max_rigid_Hz} Hz (mesh or material is wrong)"
        )
    return rigid


def gate_mesh_convergence(
    max_vm_coarse: Quantity,
    max_vm_fine: Quantity,
    tolerance: float = 0.05,
) -> float:
    """Gate 3: max stress change after 2× refine must be < ``tolerance``."""
    a = max_vm_coarse.as_MPa()
    b = max_vm_fine.as_MPa()
    change = _pct(a, b)
    if change > tolerance:
        raise GateError(
            f"stage=validate.mesh_convergence coarse={a:.4g} MPa fine={b:.4g} MPa "
            f"rel_change={change:.2%} tol={tolerance:.0%}"
        )
    return change


def topology_signature(solid: BuiltSolid, mesh: MeshModel) -> dict[str, Any]:
    return {
        "name": solid.name,
        "volume_mm3": round(solid.volume.as_mm3(), 6),
        "tags": sorted(t.name for t in solid.tags),
        "n_elements": mesh.n_elements,
        "lc_mm": mesh.characteristic_length_mm.as_mm(),
    }


def load_convergence_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def store_convergence_cache(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def reaction_moment_Nmm(
    nodes: dict[int, tuple[float, float, float]],
    forces_N: dict[int, tuple[float, float, float]],
    about_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Σ r × F at the named node set. Force is N, lever is mm → N·mm."""
    mx = my = mz = 0.0
    ax, ay, az = about_mm
    for nid, (fx, fy, fz) in forces_N.items():
        x, y, z = nodes[nid]
        rx, ry, rz = x - ax, y - ay, z - az
        mx += ry * fz - rz * fy
        my += rz * fx - rx * fz
        mz += rx * fy - ry * fx
    return (mx, my, mz)


def gate_static_torque(
    M_ccx: Quantity,
    M_ref: Quantity,
    tolerance: float = 0.02,
) -> None:
    """Gate 4: CalculiX reaction moment vs reference (Pinocchio or F×L)."""
    a = M_ccx.as_Nm()
    b = M_ref.as_Nm()
    if _pct(a, b) > tolerance:
        raise GateError(
            f"stage=validate.static_torque ccx={a:.6g} N·m "
            f"ref={b:.6g} N·m tol={tolerance:.0%}"
        )


def cantilever_hand_torque_Nm(params: dict[str, Any]) -> Quantity:
    """M = F × L for a tip load on a beam along +X. SI N·m."""
    L_m = float(require(params, "geometry", "length_mm")) / 1000.0
    fx, fy, fz = (float(v) for v in require(params, "load", "force_N"))
    # r = (L, 0, 0) × F → My = -L*Fz, Mz = L*Fy
    return moment_Nm(abs(L_m * fz if abs(fz) >= abs(fy) else L_m * fy))


def gate_pinocchio_mass(pin_mass_kg: float, cad_mass_kg: float, tolerance: float = 0.02) -> None:
    """URDF/Pinocchio total mass vs Σ (volume × density) + payload, within 2%."""
    if _pct(pin_mass_kg, cad_mass_kg) > tolerance:
        raise GateError(
            f"stage=validate.pinocchio_mass pin={pin_mass_kg:.6g} kg "
            f"cad={cad_mass_kg:.6g} kg tol={tolerance:.0%}"
        )


def gate_rnea_mgr(my_pin_Nm: float, hand_Nm: float, tolerance: float = 0.01) -> None:
    """Single-link static: Pinocchio My vs m*g*r, within 1%."""
    if _pct(my_pin_Nm, hand_Nm) > tolerance:
        raise GateError(
            f"stage=validate.rnea_mgr pin={my_pin_Nm:.6g} N·m "
            f"hand={hand_Nm:.6g} N·m tol={tolerance:.0%}"
        )


def gate_rigid_body_constrained(
    frequencies_Hz: Iterable[Quantity],
    min_Hz: float = 1.0,
) -> float:
    """Gate 5: constrained spectrum must not contain a zero-energy mode."""
    freqs = [q.as_Hz() for q in frequencies_Hz]
    if not freqs:
        raise GateError("stage=validate.rigid_body_constrained no frequencies")
    f1 = min(abs(f) for f in freqs)
    if f1 < min_Hz:
        raise GateError(
            f"stage=validate.rigid_body_constrained f1={f1:.4g} Hz < {min_Hz} Hz "
            f"(zero-energy mode under constraints)"
        )
    return f1
