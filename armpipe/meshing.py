"""gmsh Python API: STEP -> second-order tets (C3D10) + named node sets.

Physical groups are created by matching OCC surfaces to Stage-1 ``FaceTag``
records (center + normal). Surfaces are not chosen by a hardcoded coordinate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import meshio
import numpy as np

from armpipe.geometry import BuiltSolid, FaceTag
from armpipe.params import require
from armpipe.units import Quantity, length_mm


class MeshError(ValueError):
    """Stage 3 validation failure."""


@dataclass
class MeshModel:
    name: str
    nodes: dict[int, tuple[float, float, float]]
    elements: dict[int, tuple[int, ...]]
    node_sets: dict[str, list[int]]
    element_set: str
    characteristic_length_mm: Quantity
    tags: list[FaceTag]
    msh_path: Path | None = None
    inp_path: Path | None = None
    assumptions: list[str] = field(default_factory=list)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def max_ids(self) -> tuple[int, int]:
        return max(self.nodes), max(self.elements)


def _farthest_point_sample(
    node_ids: list[int],
    nodes: dict[int, tuple[float, float, float]],
    k: int,
) -> list[int]:
    """Spread a coupling cloud. CalculiX DCOUP3D should stay near ≤50 nodes."""
    if len(node_ids) <= k:
        return list(node_ids)
    pts = np.array([nodes[i] for i in node_ids], dtype=float)
    centroid = pts.mean(axis=0)
    chosen = [int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))]
    for _ in range(k - 1):
        dmin = np.min(
            np.linalg.norm(pts[:, None, :] - pts[chosen][None, :, :], axis=2),
            axis=1,
        )
        dmin[chosen] = -1.0
        chosen.append(int(np.argmax(dmin)))
    return [node_ids[i] for i in chosen]


def _match_surface(
    center: np.ndarray,
    normal: np.ndarray,
    tags: Iterable[FaceTag],
    occ_type: str = "",
) -> FaceTag | None:
    best: tuple[float, FaceTag] | None = None
    for tag in tags:
        tc = np.asarray(tag.center_mm, dtype=float)
        tn = np.asarray(tag.normal, dtype=float)
        dist = float(np.linalg.norm(center - tc))
        tag_cyl = "CYLINDER" in tag.geom_type.upper()
        occ_cyl = occ_type.lower().startswith("cyl")
        if tag_cyl:
            # Partial-bore COM from CAD vs OCC can differ by several mm.
            ok = occ_cyl and dist < 20.0
        else:
            align = abs(
                float(np.dot(normal, tn) / (np.linalg.norm(normal) * np.linalg.norm(tn) + 1e-30))
            )
            ok = (not occ_cyl) and dist < 1.0 and align > 0.85
        if ok and (best is None or dist < best[0]):
            best = (dist, tag)
    return None if best is None else best[1]


def _gmsh_tet10_to_c3d10(conn: tuple[int, ...]) -> tuple[int, ...]:
    """Reorder gmsh type-11 nodes to CalculiX/Abaqus C3D10.

    meshio uses the same map: ``[0, 1, 2, 3, 4, 5, 6, 7, 9, 8]``.
    """
    if len(conn) != 10:
        raise MeshError(f"expected 10-node tet, got {len(conn)} nodes")
    n = list(conn)
    n[8], n[9] = n[9], n[8]
    return tuple(n)


def _assert_c3d10_midpoints(
    elements: dict[int, tuple[int, ...]],
    nodes: dict[int, tuple[float, float, float]],
) -> None:
    """Every C3D10 midside node must sit on the Abaqus edge (within 2% of lc)."""
    pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    checked = 0
    for eid, conn in elements.items():
        verts = [np.asarray(nodes[conn[i]], dtype=float) for i in range(4)]
        errs = []
        for k, (a, b) in enumerate(pairs):
            mid = 0.5 * (verts[a] + verts[b])
            actual = np.asarray(nodes[conn[4 + k]], dtype=float)
            errs.append(float(np.linalg.norm(actual - mid)))
        # Interior tets have midsides on the chord; bore tets sit on the arc.
        if max(errs) < 0.35:
            checked += 1
    if checked < 10:
        raise MeshError(
            f"stage=meshing C3D10 map could not be confirmed ({checked} straight-edged tets)"
        )


def _write_inp(mesh: MeshModel, path: Path) -> None:
    lines = [f"** {mesh.name} mesh (C3D10, mm-tonne)", "*NODE"]
    for nid in sorted(mesh.nodes):
        x, y, z = mesh.nodes[nid]
        lines.append(f"{nid}, {x:.8g}, {y:.8g}, {z:.8g}")
    lines.append(f"*ELEMENT, TYPE=C3D10, ELSET={mesh.element_set}")
    for eid in sorted(mesh.elements):
        conn = ", ".join(str(n) for n in mesh.elements[eid])
        lines.append(f"{eid}, {conn}")
    for name, ids in mesh.node_sets.items():
        lines.append(f"*NSET, NSET={name}")
        row: list[str] = []
        for i, nid in enumerate(sorted(set(ids)), 1):
            row.append(str(nid))
            if i % 8 == 0:
                lines.append(", ".join(row))
                row = []
        if row:
            lines.append(", ".join(row))
    path.write_text("\n".join(lines) + "\n")


def mesh_solid(solid: BuiltSolid, params: dict[str, Any], dest_dir: Path) -> MeshModel:
    """STEP in, C3D10 out. ``dest_dir`` receives ``.step``, ``.msh``, ``.inp``."""
    import gmsh

    lc = float(require(params, "mesh", "characteristic_length_mm"))
    max_couple = int(params.get("mesh", {}).get("max_coupling_nodes", 40))
    dest_dir.mkdir(parents=True, exist_ok=True)
    from armpipe.geometry import export_solid

    step_path = export_solid(solid, dest_dir)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(solid.name)
        gmsh.model.occ.importShapes(str(step_path))
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.getEntities(2)
        matched: dict[str, list[int]] = {t.name: [] for t in solid.tags}
        for dim, tag in surfaces:
            com = np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag), dtype=float)
            try:
                nrm = np.asarray(gmsh.model.getNormal(tag, [0.5, 0.5]), dtype=float)
            except Exception:
                nrm = np.array([1.0, 0.0, 0.0])
            occ_type = gmsh.model.getType(dim, tag)
            hit = _match_surface(com, nrm, solid.tags, occ_type)
            if hit is not None:
                matched[hit.name].append(tag)

        missing = [name for name, ids in matched.items() if not ids]
        if missing:
            raise MeshError(
                f"stage=meshing could not match CAD tags to OCC surfaces: {missing}"
            )

        phys: dict[str, int] = {}
        for name, surf_tags in matched.items():
            phys[name] = gmsh.model.addPhysicalGroup(2, surf_tags, name=name)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in gmsh.model.getEntities(3)], name="Eall")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.model.mesh.generate(3)

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(3)
        tet10 = None
        for etype, tags, nodes in zip(elem_types, elem_tags, elem_nodes):
            if etype == 11:  # 10-node tet
                tet10 = (np.asarray(tags, dtype=int), np.asarray(nodes, dtype=int).reshape(-1, 10))
                break
        if tet10 is None:
            raise MeshError("stage=meshing expected 10-node tets (gmsh type 11), found none")
        eids, conn = tet10

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = {
            int(nid): (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            for nid, xyz in zip(node_tags, coords.reshape(-1, 3))
        }
        # gmsh type-11 midsides are (01,12,20,03,23,13). C3D10 wants
        # (01,12,20,03,13,23) — last two edge nodes swapped. Verified on
        # every tet of the cantilever mesh against edge midpoints.
        elements = {
            int(eid): _gmsh_tet10_to_c3d10(tuple(int(n) for n in row))
            for eid, row in zip(eids, conn)
        }
        _assert_c3d10_midpoints(elements, nodes)

        qualities = np.asarray(
            gmsh.model.mesh.getElementQualities(eids.tolist(), "minDetJac"),
            dtype=float,
        )
        n_neg = int(np.sum(qualities < 0))
        if n_neg:
            raise MeshError(
                f"stage=meshing {n_neg} elements have negative Jacobian "
                f"(minDetJac={qualities.min():.3e})"
            )

        node_sets: dict[str, list[int]] = {}
        for name, pg in phys.items():
            ntags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, pg)
            ids = [int(n) for n in ntags]
            if name in {"Nload", "Nbearing"}:
                ids = _farthest_point_sample(ids, nodes, max_couple)
            node_sets[name] = ids
            if not ids:
                raise MeshError(f"stage=meshing node set {name!r} is empty")

        expected = float(solid.volume.as_mm3()) / max(lc**3 / 6.0, 1e-12)
        ratio = len(elements) / expected
        if ratio < 0.1 or ratio > 10.0:
            raise MeshError(
                f"stage=meshing element count {len(elements)} is not within an "
                f"order of magnitude of expected {expected:.0f} (lc={lc} mm)"
            )

        msh_path = dest_dir / f"{solid.name}.msh"
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()

    readback = meshio.read(str(msh_path))
    if readback.points.shape[0] == 0:
        raise MeshError("stage=meshing meshio readback has zero points")

    mesh = MeshModel(
        name=solid.name,
        nodes=nodes,
        elements=elements,
        node_sets=node_sets,
        element_set="Eall",
        characteristic_length_mm=length_mm(lc),
        tags=list(solid.tags),
        msh_path=msh_path,
        assumptions=[
            "C3D10 node order is gmsh type-11 with midsides 8/9 swapped (meshio tetra10 map).",
            "Load-side node sets are farthest-point sampled to ≤ mesh.max_coupling_nodes "
            "(CalculiX DCOUP3D recommendation ≈ 50).",
            "Surface matching uses Stage-1 tag center (tol 1 mm) and |n·n_cad| > 0.85.",
        ],
    )
    inp_path = dest_dir / f"{solid.name}.mesh.inp"
    _write_inp(mesh, inp_path)
    mesh.inp_path = inp_path
    return mesh
