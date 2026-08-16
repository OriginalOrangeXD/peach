"""build123d part builders. One function per solid.

CAD native unit is millimetre (build123d ``Unit.MM``). Mass properties are
taken from the B-rep (volume, COM, inertia) — never from a box approximation.

Face names do not survive STEP export (verified: ``face.label`` is dropped).
Named boundary regions are therefore written to a ``tags.json`` sidecar that
``meshing.py`` matches to OCC surfaces by center and normal. Matching is
driven by these Stage-1 tags, not by meshing picking faces from coordinates
it invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from build123d import Align, Axis, Box, Cylinder, Part, Pos, Rotation, export_step

from armpipe.params import require
from armpipe.units import (
    Kind,
    Quantity,
    UnitSystem,
    Vec3,
    vec3,
    volume_mm3,
)


class GeometryError(ValueError):
    """Stage 1 validation failure."""


@dataclass(frozen=True)
class FaceTag:
    """A named CAD face that must become a mesh node set."""

    name: str
    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    area_mm2: float
    geom_type: str


@dataclass
class BuiltSolid:
    name: str
    part: Part
    tags: list[FaceTag]
    volume: Quantity
    com: Vec3
    inertia_mm5: np.ndarray
    assumptions: list[str] = field(default_factory=list)

    def tag(self, name: str) -> FaceTag:
        for t in self.tags:
            if t.name == name:
                return t
        raise GeometryError(f"solid {self.name!r} has no face tag {name!r}")


def _inertia_mm5(part: Part) -> np.ndarray:
    """Geometric inertia (density = 1) about the COM, in mm⁵.

    ``part.matrix_of_inertia`` is a property, not a method. Mass inertia for
    the URDF is ``I_geom * density`` after a unit conversion in ``units.py``.
    """
    raw = np.asarray(part.matrix_of_inertia, dtype=float)
    if raw.shape != (3, 3):
        raise GeometryError(f"inertia tensor shape {raw.shape}, expected (3, 3)")
    return raw


def _face_tag(face, name: str) -> FaceTag:
    c = face.center()
    n = face.normal_at()
    return FaceTag(
        name=name,
        center_mm=(float(c.X), float(c.Y), float(c.Z)),
        normal=(float(n.X), float(n.Y), float(n.Z)),
        area_mm2=float(face.area),
        geom_type=str(face.geom_type),
    )


def validate_solid(solid: BuiltSolid) -> None:
    """Gate: volume > 0 and inertia is symmetric positive-definite."""
    vol = solid.volume.as_mm3()
    if vol <= 0:
        raise GeometryError(f"stage=geometry volume={vol} mm³ is not > 0")
    I = np.asarray(solid.inertia_mm5, dtype=float)
    if not np.allclose(I, I.T, atol=1e-6 * max(1.0, np.max(np.abs(I)))):
        raise GeometryError(f"stage=geometry inertia is not symmetric:\n{I}")
    eigs = np.linalg.eigvalsh(0.5 * (I + I.T))
    if np.any(eigs <= 0):
        raise GeometryError(f"stage=geometry inertia eigs not PD: {eigs}")


def export_solid(solid: BuiltSolid, dest_dir: Path) -> Path:
    """Write STEP (mm) and ``tags.json``. Raises if STEP export fails."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    step_path = dest_dir / f"{solid.name}.step"
    tags_path = dest_dir / f"{solid.name}.tags.json"
    ok = export_step(solid.part, str(step_path))
    if not ok or not step_path.is_file() or step_path.stat().st_size == 0:
        raise GeometryError(f"stage=geometry STEP export failed for {solid.name}")
    payload = {
        "name": solid.name,
        "volume_mm3": solid.volume.as_mm3(),
        "com_mm": solid.com.as_array_mm_tonne(),
        "inertia_mm5": solid.inertia_mm5.tolist(),
        "tags": [t.__dict__ for t in solid.tags],
        "assumptions": solid.assumptions,
    }
    tags_path.write_text(json.dumps(payload, indent=2))
    return step_path


def build_cantilever(params: dict[str, Any]) -> BuiltSolid:
    """Rectangular cantilever. Axis +X, width Y, height Z, origin at the
    fixed-face center. Dimensions come only from ``params['geometry']``.
    """
    L = float(require(params, "geometry", "length_mm"))
    b = float(require(params, "geometry", "width_mm"))
    h = float(require(params, "geometry", "height_mm"))
    if min(L, b, h) <= 0:
        raise GeometryError("cantilever length/width/height must be > 0")

    part = Box(L, b, h, align=(Align.MIN, Align.CENTER, Align.CENTER))
    if not part.is_valid:
        raise GeometryError("cantilever solid is not valid")

    x_faces = part.faces().filter_by(Axis.X).sort_by(Axis.X)
    if len(x_faces) < 2:
        raise GeometryError(f"expected 2 X-normal faces, got {len(x_faces)}")
    tags = [
        _face_tag(x_faces[0], "Nfixed"),
        _face_tag(x_faces[-1], "Nload"),
    ]

    com = part.center()
    solid = BuiltSolid(
        name=str(params.get("name", "cantilever")),
        part=part,
        tags=tags,
        volume=volume_mm3(float(part.volume)),
        com=vec3((com.X, com.Y, com.Z), Kind.LENGTH, UnitSystem.MM_TONNE),
        inertia_mm5=_inertia_mm5(part),
        assumptions=[
            "CAD unit is millimetre.",
            "Origin is the fixed-face center; beam axis is +X; width is Y; height is Z.",
            "Face names are carried in tags.json because STEP drops face.label.",
            "matrix_of_inertia is the geometric (density=1) tensor about the COM, in mm^5.",
        ],
    )
    validate_solid(solid)
    return solid


def _link_block(params: dict[str, Any], link_name: str) -> dict[str, Any]:
    return require(params, "geometry", "links", link_name)


def _y_bore(x_mm: float, radius_mm: float, width_mm: float):
    """Cylinder along +Y through the link width, centred on X."""
    return Pos(x_mm, 0, 0) * Rotation(Axis.X, 90) * Cylinder(
        radius_mm, width_mm + 4.0, align=(Align.CENTER, Align.CENTER, Align.CENTER)
    )


def _tag_plane_min_x(part: Part, name: str) -> FaceTag:
    faces = part.faces().filter_by(Axis.X).sort_by(Axis.X)
    if not faces:
        raise GeometryError(f"no X-normal faces to tag as {name}")
    return _face_tag(faces[0], name)


def _tag_distal_bore(part: Part, name: str, radius_mm: float) -> FaceTag:
    """Tag the cylindrical bore with this radius that sits furthest along +X."""
    hits = []
    for face in part.faces():
        if not str(face.geom_type).endswith("CYLINDER"):
            continue
        if abs(float(face.radius) - radius_mm) > 0.15:
            continue
        hits.append(face)
    if not hits:
        raise GeometryError(f"no cylindrical face with r≈{radius_mm} mm for {name}")
    hits.sort(key=lambda f: float(f.center().X))
    return _face_tag(hits[-1], name)


def build_link(params: dict[str, Any], link_name: str | None = None) -> BuiltSolid:
    """Hollow rectangular link with proximal flange and distal Y-bore.

    Frame (mm): origin at the proximal face center, +X toward the distal
    joint, +Y = joint axis, +Z up when q=0. Same frame the URDF uses.
    """
    if link_name is None:
        link_name = str(params.get("_link_name") or require(params, "dynamics", "analyze_link"))
    g = _link_block(params, link_name)
    L = float(g["length_mm"])
    W = float(g["width_mm"])
    H = float(g["height_mm"])
    t = float(g["wall_mm"])
    bore = float(g["bore_d_mm"])
    flange = float(g["flange_thick_mm"])
    hollow = bool(g.get("hollow", True))
    if min(L, W, H, flange) <= 0:
        raise GeometryError(f"{link_name} dimensions must be > 0")
    if hollow and min(t, bore) <= 0:
        raise GeometryError(f"{link_name} hollow link needs wall_mm and bore_d_mm > 0")
    if hollow and 2 * t >= min(W, H):
        raise GeometryError(f"{link_name} wall_mm leaves no inner cavity")

    outer = Box(L, W, H, align=(Align.MIN, Align.CENTER, Align.CENTER))
    part = outer
    if hollow:
        n_faces0 = len(outer.faces())
        inner_L = L - 2.0 * flange
        if inner_L > 1.0:
            inner = Pos(flange, 0, 0) * Box(
                inner_L, W - 2.0 * t, H - 2.0 * t,
                align=(Align.MIN, Align.CENTER, Align.CENTER),
            )
            part = outer - inner
            if not part or len(part.faces()) == n_faces0:
                raise GeometryError(f"{link_name} inner pocket boolean was a no-op")
        r = bore / 2.0
        before = len(part.faces())
        part = part - _y_bore(L - flange / 2.0, r, W)
        if not part or len(part.faces()) == before:
            raise GeometryError(f"{link_name} distal bore boolean was a no-op")
    if not part.is_valid:
        raise GeometryError(f"{link_name} solid is not valid")

    tags = [_tag_plane_min_x(part, "Nmount")]
    if hollow:
        tags.append(_tag_distal_bore(part, "Nbearing", bore / 2.0))
    else:
        x_faces = part.faces().filter_by(Axis.X).sort_by(Axis.X)
        tags.append(_face_tag(x_faces[-1], "Nbearing"))
    com = part.center()
    solid = BuiltSolid(
        name=link_name,
        part=part,
        tags=tags,
        volume=volume_mm3(float(part.volume)),
        com=vec3((com.X, com.Y, com.Z), Kind.LENGTH, UnitSystem.MM_TONNE),
        inertia_mm5=_inertia_mm5(part),
        assumptions=[
            "CAD unit is millimetre.",
            "Link frame: origin at proximal face center, +X distal, +Y joint axis.",
            "Hollow box; distal load is a Y-bore; proximal BC is the x=0 flange.",
            "Kinematic length is params tree origin, not a measurement of this solid.",
        ],
    )
    validate_solid(solid)
    return solid


def build_base(params: dict[str, Any]) -> BuiltSolid:
    """Base block. Proximal face is world-fixed; top/distal bore is joint 1."""
    return build_link(params, "base")


def build_link1(params: dict[str, Any]) -> BuiltSolid:
    return build_link(params, "link1")


def build_link2(params: dict[str, Any]) -> BuiltSolid:
    return build_link(params, "link2")


def build_all_links(params: dict[str, Any]) -> dict[str, BuiltSolid]:
    """Build every CAD link named under geometry.links. Tree comes from params."""
    names = list(require(params, "geometry", "links").keys())
    return {name: build_link(params, name) for name in names}
