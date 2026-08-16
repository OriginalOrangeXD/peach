"""Emit URDF from params.yaml (tree) + build123d mass properties (SI).

The kinematic tree is never inferred from geometry. Joint origins, axes,
and limits come only from ``params['tree']``. Inertial mass, COM, and the
inertia tensor come from Stage 1 solids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

from armpipe.geometry import BuiltSolid
from armpipe.params import material_from_params, require
from armpipe.units import geometric_inertia_mm5_to_si, mass_from_volume_si


class UrdfError(ValueError):
    """Stage 2 failure."""


@dataclass
class InertialSI:
    mass_kg: float
    com_m: tuple[float, float, float]
    I_kg_m2: tuple[float, float, float, float, float, float]  # ixx ixy ixz iyy iyz izz


def inertial_from_solid(solid: BuiltSolid, params: dict[str, Any]) -> InertialSI:
    material = material_from_params(params)
    mass = mass_from_volume_si(solid.volume.in_si(), material)
    com = solid.com.in_si().as_array_si()
    I = geometric_inertia_mm5_to_si(solid.inertia_mm5, material.density)
    return InertialSI(
        mass_kg=mass.as_kg(),
        com_m=(float(com[0]), float(com[1]), float(com[2])),
        I_kg_m2=(
            float(I[0, 0]),
            float(I[0, 1]),
            float(I[0, 2]),
            float(I[1, 1]),
            float(I[1, 2]),
            float(I[2, 2]),
        ),
    )


def _xyz(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _add_inertial(link_el: Element, inertial: InertialSI) -> None:
    inn = SubElement(link_el, "inertial")
    SubElement(inn, "origin", xyz=_xyz(inertial.com_m), rpy="0 0 0")
    SubElement(inn, "mass", value=f"{inertial.mass_kg:.8g}")
    ixx, ixy, ixz, iyy, iyz, izz = inertial.I_kg_m2
    SubElement(
        inn,
        "inertia",
        ixx=f"{ixx:.8g}",
        ixy=f"{ixy:.8g}",
        ixz=f"{ixz:.8g}",
        iyy=f"{iyy:.8g}",
        iyz=f"{iyz:.8g}",
        izz=f"{izz:.8g}",
    )


def assert_tree_matches_geometry(params: dict[str, Any]) -> None:
    """Joint translations along a link must equal that link's length_mm / 1000."""
    geom = require(params, "geometry", "links")
    for node in require(params, "tree", "links"):
        parent = node.get("parent")
        joint = node.get("joint")
        if not parent or not joint:
            continue
        if parent not in geom:
            continue
        xyz = [float(v) for v in joint["origin_xyz_m"]]
        L_m = float(geom[parent]["length_mm"]) / 1000.0
        # Child joint sitting at the distal end of the parent: x ≈ L, yz ≈ 0.
        if abs(xyz[1]) < 1e-9 and abs(xyz[2]) < 1e-9 and xyz[0] > 1e-6:
            if abs(xyz[0] - L_m) > 1e-9:
                raise UrdfError(
                    f"stage=urdf joint {joint['name']} origin_xyz_m[0]={xyz[0]} "
                    f"!= geometry.links.{parent}.length_mm/1000={L_m}"
                )


def write_urdf(
    params: dict[str, Any],
    solids: dict[str, BuiltSolid],
    dest: Path,
) -> Path:
    """Write a SI URDF. ``solids`` keyed by link name; payload is not a solid."""
    assert_tree_matches_geometry(params)
    robot = Element("robot", name=str(params.get("name", "arm")))
    material = material_from_params(params)
    inertials: dict[str, InertialSI] = {}
    for name, solid in solids.items():
        inertials[name] = inertial_from_solid(solid, params)

    payload = params.get("payload") or {}
    payload_mass = float(payload.get("mass_kg", 0.0))
    if payload_mass > 0 and "payload" not in inertials:
        ixx, iyy, izz = (float(v) for v in payload.get("inertia_kg_m2", (1e-6, 1e-6, 1e-6)))
        inertials["payload"] = InertialSI(
            mass_kg=payload_mass,
            com_m=(0.0, 0.0, 0.0),
            I_kg_m2=(ixx, 0.0, 0.0, iyy, 0.0, izz),
        )

    for node in require(params, "tree", "links"):
        name = node["name"]
        link_el = SubElement(robot, "link", name=name)
        if name in inertials:
            _add_inertial(link_el, inertials[name])
        elif name == "payload" and payload_mass <= 0:
            _add_inertial(
                link_el,
                InertialSI(1e-9, (0.0, 0.0, 0.0), (1e-12, 0, 0, 1e-12, 0, 1e-12)),
            )
        else:
            raise UrdfError(f"stage=urdf no inertial data for link {name!r}")

    for node in require(params, "tree", "links"):
        joint = node.get("joint")
        parent = node.get("parent")
        if not joint or not parent:
            continue
        jtype = str(joint["type"])
        jel = SubElement(robot, "joint", name=str(joint["name"]), type=jtype)
        SubElement(jel, "parent", link=str(parent))
        SubElement(jel, "child", link=str(node["name"]))
        SubElement(
            jel,
            "origin",
            xyz=_xyz(joint["origin_xyz_m"]),
            rpy=_xyz(joint.get("origin_rpy", (0, 0, 0))),
        )
        SubElement(jel, "axis", xyz=_xyz(joint.get("axis", (0, 0, 1))))
        if jtype == "revolute":
            SubElement(
                jel,
                "limit",
                lower=str(joint["limit_lower"]),
                upper=str(joint["limit_upper"]),
                effort=str(joint.get("effort", 0)),
                velocity=str(joint.get("velocity", 0)),
            )

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    indent(robot)
    ElementTree(robot).write(dest, encoding="unicode", xml_declaration=True)
    return dest


def cad_mass_kg(
    solids: dict[str, BuiltSolid],
    params: dict[str, Any],
    *,
    include_root: bool = False,
) -> float:
    """Σ volume×density + payload.

    Pinocchio's ``computeTotalMass`` skips the universe/root inertia (the
    fixed base). Default comparison therefore excludes ``tree.root``.
    """
    material = material_from_params(params)
    root = str(require(params, "tree", "root"))
    total = 0.0
    for name, solid in solids.items():
        if not include_root and name == root:
            continue
        total += mass_from_volume_si(solid.volume.in_si(), material).as_kg()
    total += float((params.get("payload") or {}).get("mass_kg", 0.0))
    return total
