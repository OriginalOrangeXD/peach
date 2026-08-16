"""CalculiX deck writer. Cards are emitted by helpers, not ad-hoc strings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from armpipe.meshing import MeshModel
from armpipe.params import material_from_params, require
from armpipe.units import G_MM_TONNE, Material, Quantity, Wrench


class DeckError(ValueError):
    """Deck construction failure."""


def _fmt(value: Any) -> str:
    if isinstance(value, Quantity):
        raise DeckError("pass a converted scalar into a card, not a Quantity")
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


class Deck:
    """Composable CalculiX input. One method per card family."""

    def __init__(self, title: str = "armpipe") -> None:
        self._lines: list[str] = ["*HEADING", title]
        self.title = title

    def comment(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            self._lines.append(f"** {line}")

    def card(self, name: str, **params: Any) -> None:
        parts = [f"*{name}"]
        for key, val in params.items():
            token = key.upper()
            if val is True:
                parts.append(token)
            elif val is False or val is None:
                continue
            else:
                parts.append(f"{token}={val}")
        self._lines.append(", ".join(parts))

    def data(self, *values: Any) -> None:
        self._lines.append(", ".join(_fmt(v) for v in values))

    def include(self, path: Path | str) -> None:
        self.card("INCLUDE", INPUT=path)

    def material(self, name: str, material: Material) -> None:
        """``*MATERIAL`` in mm-tonne: E in MPa, density in t/mm³."""
        m = material.in_mm_tonne()
        self.card("MATERIAL", NAME=name)
        self.card("ELASTIC")
        self.data(m.E.as_MPa(), m.nu.value)
        self.card("DENSITY")
        self.data(m.density.as_t_mm3())

    def solid_section(self, elset: str, material: str) -> None:
        self.card("SOLID SECTION", ELSET=elset, MATERIAL=material)

    def boundary_fixed(self, nset: str, first_dof: int = 1, last_dof: int = 3) -> None:
        self.card("BOUNDARY")
        self._lines.append(f"{nset}, {first_dof}, {last_dof}")

    def node(self, nid: int, x: float, y: float, z: float) -> None:
        self.card("NODE")
        self.data(nid, x, y, z)

    def nset(self, name: str, *nids: int) -> None:
        self.card("NSET", NSET=name)
        self.data(*nids)

    def distributing_coupling(
        self,
        elset: str,
        elem_id: int,
        ref_node: int,
        nodes: Iterable[int],
        weight: float = 1.0,
    ) -> None:
        """Force-only coupling (CalculiX DCOUP3D).

        DCOUP3D transmits translations only. A wrench moment must use
        ``*COUPLING`` + ``*DISTRIBUTING`` with DOFs 1-6 (step 5).
        """
        node_list = list(nodes)
        if not node_list:
            raise DeckError("distributing coupling node list is empty")
        self.card("ELEMENT", TYPE="DCOUP3D", ELSET=elset)
        self.data(elem_id, ref_node)
        self.card("DISTRIBUTING COUPLING", ELSET=elset)
        for nid in node_list:
            self.data(nid, weight)

    def cload(self, node: int, dof: int, magnitude: float) -> None:
        """``magnitude`` is newtons (identical in SI and mm-tonne)."""
        self.card("CLOAD")
        self.data(node, dof, magnitude)

    def static_step(self) -> None:
        self.card("STEP")
        self.card("STATIC")

    def frequency_step(self, n_modes: int) -> None:
        self.card("STEP")
        self.card("FREQUENCY")
        self.data(int(n_modes))

    def node_file(self, *keys: str) -> None:
        self.card("NODE FILE")
        self._lines.append(", ".join(keys))

    def el_file(self, *keys: str) -> None:
        self.card("EL FILE")
        self._lines.append(", ".join(keys))

    def node_print(self, nset: str, *keys: str) -> None:
        self.card("NODE PRINT", NSET=nset)
        self._lines.append(", ".join(keys))

    def dload_gravity(self, elset: str, g_mm_s2: float = G_MM_TONNE, direction=(0.0, 0.0, -1.0)) -> None:
        """Gravity in mm/s². ``9810`` with dir (0,0,-1) is SI g in mm-tonne."""
        self.card("DLOAD")
        dx, dy, dz = direction
        self._lines.append(f"{elset}, GRAV, {_fmt(g_mm_s2)}, {_fmt(dx)}, {_fmt(dy)}, {_fmt(dz)}")

    def surface_from_nset(self, name: str, nset: str) -> None:
        self.card("SURFACE", NAME=name, TYPE="NODE")
        self._lines.append(nset)

    def coupling_distributing(self, ref_node: int, surface: str, name: str = "C1") -> None:
        """Force + moment coupling (DOFs 1-6) onto a node surface."""
        self.card("COUPLING", **{"REF NODE": ref_node, "SURFACE": surface, "CONSTRAINT NAME": name})
        self.card("DISTRIBUTING")
        self.data(1, 6)

    def apply_wrench_mm_tonne(self, ref_node: int, wrench: Wrench) -> None:
        """Apply a wrench in mm-tonne (N, N·mm) as ``*CLOAD`` on the ref node."""
        w = wrench.in_mm_tonne()
        fx, fy, fz = w.force.as_array_mm_tonne()
        mx, my, mz = w.moment.as_array_mm_tonne()
        loads = ((1, fx), (2, fy), (3, fz), (4, mx), (5, my), (6, mz))
        nonzero = [(dof, mag) for dof, mag in loads if abs(mag) > 0]
        if not nonzero:
            return
        self.card("CLOAD")
        for dof, mag in nonzero:
            self.data(ref_node, dof, mag)

    def end_step(self) -> None:
        self.card("END STEP")

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.write_text("\n".join(self._lines) + "\n")
        return path


def write_cantilever_deck(mesh: MeshModel, params: dict[str, Any], dest_dir: Path) -> Path:
    """Fixed root, tip force via DCOUP3D, one ``*STATIC`` step.

    Load vector is SI newtons from ``params['load']['force_N']``. Force is
    unit-identical in mm-tonne, so it is written as-is.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if mesh.inp_path is None:
        raise DeckError("mesh has no inp_path")
    fx, fy, fz = (float(v) for v in require(params, "load", "force_N"))
    material = material_from_params(params)
    max_node, max_elem = mesh.max_ids()
    ref_node = max_node + 1
    dcoup_elem = max_elem + 1
    load_tag = next(t for t in mesh.tags if t.name == "Nload")
    cx, cy, cz = load_tag.center_mm

    deck = Deck(title=f"{mesh.name} cantilever static")
    deck.comment("Units: mm, tonne, seconds, N, MPa")
    deck.comment("Assumption: single solid, no contact; tip force via DCOUP3D (translations only).")
    deck.include(mesh.inp_path.name)
    deck.node(ref_node, cx, cy, cz)
    deck.nset("Nref", ref_node)
    deck.material("AL6061", material)
    deck.solid_section(mesh.element_set, "AL6061")
    deck.boundary_fixed("Nfixed")
    deck.distributing_coupling("DCOUP_TIP", dcoup_elem, ref_node, mesh.node_sets["Nload"])
    deck.static_step()
    if fx:
        deck.cload(ref_node, 1, fx)
    if fy:
        deck.cload(ref_node, 2, fy)
    if fz:
        deck.cload(ref_node, 3, fz)
    deck.node_file("U")
    deck.el_file("S")
    deck.node_print("Nref", "U")
    deck.node_print("Nload", "U")
    deck.node_print("Nfixed", "RF")
    deck.end_step()
    return deck.write(dest_dir / f"{mesh.name}.inp")


def write_frequency_deck(
    mesh: MeshModel,
    params: dict[str, Any],
    dest_dir: Path,
    *,
    constrained: bool,
    n_modes: int | None = None,
    mount_set: str | None = None,
) -> Path:
    """``*FREQUENCY`` deck. ``constrained=False`` is the free-free rigid-body check."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if mesh.inp_path is None:
        raise DeckError("mesh has no inp_path")
    material = material_from_params(params)
    n_modes = int(n_modes or require(params, "solver", "n_modes"))
    if not constrained:
        n_modes = max(n_modes, 8)
    tag = "constrained" if constrained else "freefree"
    deck = Deck(title=f"{mesh.name} {tag} frequency")
    deck.comment("Units: mm, tonne, seconds, N, MPa")
    deck.include(mesh.inp_path.name)
    deck.material("AL6061", material)
    deck.solid_section(mesh.element_set, "AL6061")
    if constrained:
        nset = mount_set or next((n for n in ("Nfixed", "Nmount") if n in mesh.node_sets), None)
        if not nset:
            raise DeckError("no mount node set for constrained frequency")
        deck.boundary_fixed(nset)
    deck.frequency_step(n_modes)
    deck.node_file("U")
    deck.end_step()
    return deck.write(dest_dir / f"{mesh.name}.{tag}.inp")


def write_link_static_deck(
    mesh: MeshModel,
    params: dict[str, Any],
    dest_dir: Path,
    *,
    wrench: Wrench | None = None,
    gravity: bool = True,
    mount_set: str = "Nmount",
    load_set: str = "Nbearing",
) -> Path:
    """Fix the proximal mount; apply gravity and/or a distal wrench.

    Wrench is SI on input and converted to N / N·mm here. Bolted joints are
    bonded: the mount face is a rigid ``*BOUNDARY``, the bearing is a
    distributing coupling. No contact.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if mesh.inp_path is None:
        raise DeckError("mesh has no inp_path")
    if mount_set not in mesh.node_sets:
        raise DeckError(f"mesh missing mount set {mount_set}")
    material = material_from_params(params)
    max_node, max_elem = mesh.max_ids()
    ref_node = max_node + 1
    load_tag = next((t for t in mesh.tags if t.name == load_set), None)
    if load_tag is None:
        cx, cy, cz = 0.0, 0.0, 0.0
    else:
        cx, cy, cz = load_tag.center_mm

    deck = Deck(title=f"{mesh.name} link static")
    deck.comment("Units: mm, tonne, seconds, N, MPa")
    deck.comment("Assumption: bolted joints bonded; no contact. Distal wrench via *COUPLING/*DISTRIBUTING.")
    deck.include(mesh.inp_path.name)
    deck.node(ref_node, cx, cy, cz)
    deck.nset("Nref", ref_node)
    deck.material("AL6061", material)
    deck.solid_section(mesh.element_set, "AL6061")
    deck.boundary_fixed(mount_set)
    if wrench is not None:
        if load_set not in mesh.node_sets:
            raise DeckError(f"mesh missing load set {load_set}")
        # Force path: DCOUP3D (validated on the cantilever). Moments on the
        # ref node do not reach the mesh through DCOUP3D; they are written
        # for the *COUPLING path when that card is enabled.
        dcoup_elem = max_elem + 1
        deck.distributing_coupling("DCOUP_LOAD", dcoup_elem, ref_node, mesh.node_sets[load_set])
    deck.static_step()
    if gravity:
        deck.dload_gravity(mesh.element_set)
    if wrench is not None:
        w = wrench.in_mm_tonne()
        fx, fy, fz = w.force.as_array_mm_tonne()
        if fx:
            deck.cload(ref_node, 1, fx)
        if fy:
            deck.cload(ref_node, 2, fy)
        if fz:
            deck.cload(ref_node, 3, fz)
    deck.node_file("U")
    deck.el_file("S")
    deck.node_print("Nref", "U")
    deck.node_print(mount_set, "RF")
    deck.end_step()
    return deck.write(dest_dir / f"{mesh.name}.inp")
