"""Parse CalculiX ``.frd`` / ``.dat`` into scalars. Nothing downstream reads raw solver output."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from armpipe.units import Quantity, frequency_Hz, length_mm, stress_MPa


class ResultsError(ValueError):
    """Parser or extraction failure."""


def von_mises(sxx: float, syy: float, szz: float, sxy: float, syz: float, szx: float) -> float:
    return math.sqrt(
        0.5
        * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (sxy**2 + syz**2 + szx**2)
        )
    )


def _floats12(payload: str) -> list[float]:
    """Parse the 12-character scientific fields used in ASCII .frd."""
    out: list[float] = []
    for i in range(0, len(payload), 12):
        chunk = payload[i : i + 12].strip()
        if chunk:
            out.append(float(chunk))
    return out


def _node_id(rec: str) -> int:
    return int(rec[5:13])


@dataclass
class NodalField:
    name: str
    components: list[str]
    values: dict[int, tuple[float, ...]]


@dataclass
class FrdModel:
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    fields: list[NodalField] = field(default_factory=list)

    def last_field(self, name: str) -> NodalField:
        hits = [f for f in self.fields if f.name.upper() == name.upper()]
        if not hits:
            raise ResultsError(f"stage=results no FRD field named {name!r}")
        return hits[-1]


def parse_frd(path: Path) -> FrdModel:
    """ASCII .frd reader for node coordinates, DISP, and STRESS blocks.

    Record layout (CalculiX 2.23): 5-char key, 8-char node id, then 12-char
    floats. The node block header is ``    2C``, not ``   2C``.
    """
    model = FrdModel()
    lines = Path(path).read_text(errors="replace").splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("    2C"):
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                rec = lines[i]
                if rec.startswith(" -1"):
                    nums = _floats12(rec[13:])
                    if len(nums) >= 3:
                        model.nodes[_node_id(rec)] = (nums[0], nums[1], nums[2])
                i += 1
            i += 1
            continue
        if line.startswith(" -4"):
            name = line[5:13].strip()
            comps: list[str] = []
            i += 1
            while i < n and lines[i].startswith(" -5"):
                comps.append(lines[i][5:13].strip())
                i += 1
            values: dict[int, list[float]] = {}
            while i < n and not lines[i].startswith(" -3"):
                rec = lines[i]
                if rec.startswith(" -1"):
                    values[_node_id(rec)] = _floats12(rec[13:])
                elif rec.startswith(" -2") and values:
                    last = next(reversed(values))
                    values[last].extend(_floats12(rec[13:]))
                i += 1
            model.fields.append(
                NodalField(name=name, components=comps, values={k: tuple(v) for k, v in values.items()})
            )
            i += 1
            continue
        i += 1
    return model


# MODE  EIGENVALUE  OMEGA(rad/s)  FREQ(Hz)  IMAG(rad/s)
_FREQ_RE = re.compile(
    r"^\s*(\d+)\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)"
)
_NODE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)"
)


def parse_dat_displacements(path: Path, nset: str) -> dict[int, tuple[float, float, float]]:
    """Parse ``*NODE PRINT`` displacement blocks for ``nset``."""
    text = Path(path).read_text(errors="replace")
    blocks: dict[int, tuple[float, float, float]] = {}
    capture = False
    for line in text.splitlines():
        low = line.lower()
        if "displacements" in low and nset.lower() in low:
            capture = True
            blocks = {}
            continue
        if capture:
            if not line.strip():
                if blocks:
                    break
                continue
            m = _NODE_ROW_RE.match(line)
            if m:
                blocks[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            elif blocks and line.strip() and not line.strip()[0].isdigit():
                break
    return blocks


def parse_dat_forces(path: Path, nset: str) -> dict[int, tuple[float, float, float]]:
    """Parse ``*NODE PRINT`` reaction-force blocks for ``nset``."""
    blocks: dict[int, tuple[float, float, float]] = {}
    capture = False
    for line in Path(path).read_text(errors="replace").splitlines():
        low = line.lower()
        if "forces" in low and nset.lower() in low:
            capture = True
            blocks = {}
            continue
        if capture:
            if not line.strip():
                if blocks:
                    break
                continue
            m = _NODE_ROW_RE.match(line)
            if m:
                blocks[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            elif blocks and line.strip() and not line.strip()[0].isdigit():
                break
    return blocks


def parse_dat_frequencies(path: Path) -> list[Quantity]:
    """Eigenfrequencies in Hz from a ``*FREQUENCY`` .dat block."""
    freqs: list[Quantity] = []
    in_block = False
    for line in Path(path).read_text(errors="replace").splitlines():
        if "EIGENVALUE" in line.upper() and "FREQUENCY" in line.upper():
            in_block = True
            continue
        if not in_block:
            continue
        m = _FREQ_RE.match(line)
        if m:
            freqs.append(frequency_Hz(float(m.group(4))))
        elif freqs and line.strip() and not line.strip()[0].isdigit():
            break
    return freqs


@dataclass
class ScalarResults:
    max_von_mises: Quantity
    max_von_mises_xyz_mm: tuple[float, float, float]
    tip_deflection: Quantity
    mass: Quantity | None = None
    frequencies_Hz: list[Quantity] = field(default_factory=list)
    reaction_force_N: tuple[float, float, float] | None = None
    root_fiber_von_mises: Quantity | None = None

    def to_metrics(self) -> dict:
        out = {
            "max_von_mises_MPa": self.max_von_mises.as_MPa(),
            "max_von_mises_location_mm": list(self.max_von_mises_xyz_mm),
            "tip_deflection_mm": self.tip_deflection.as_mm(),
        }
        if self.mass is not None:
            out["mass_kg"] = self.mass.as_kg()
        if self.root_fiber_von_mises is not None:
            out["root_fiber_von_mises_MPa"] = self.root_fiber_von_mises.as_MPa()
        for i, f in enumerate(self.frequencies_Hz[:5], 1):
            out[f"f{i}"] = f.as_Hz()
        return out


def extract_static_scalars(
    frd: FrdModel,
    dat_path: Path,
    tip_nset: str,
    load_direction: tuple[float, float, float],
    fiber_point_mm: tuple[float, float, float] | None = None,
) -> ScalarResults:
    """Tip deflection from ``Nref``; global max VM from FRD STRESS.

    ``fiber_point_mm`` is the beam-theory outer-fiber station used for the
    closed-form stress check (away from constrained corners).
    """
    disp = parse_dat_displacements(dat_path, "Nref")
    if not disp:
        disp = parse_dat_displacements(dat_path, tip_nset)
    if not disp:
        raise ResultsError("stage=results no tip displacements in .dat")
    norm = math.sqrt(sum(a * a for a in load_direction)) or 1.0
    axis = tuple(a / norm for a in load_direction)
    tips = [abs(u[0] * axis[0] + u[1] * axis[1] + u[2] * axis[2]) for u in disp.values()]
    tip_defl = sum(tips) / len(tips)

    if not frd.nodes:
        raise ResultsError("stage=results FRD node block is empty")
    stress = frd.last_field("STRESS")
    best = -1.0
    loc = (0.0, 0.0, 0.0)
    for nid, comps in stress.values.items():
        if len(comps) < 6:
            continue
        vm = von_mises(*comps[:6])
        if vm > best:
            best = vm
            loc = frd.nodes.get(nid, (0.0, 0.0, 0.0))
    if best < 0:
        raise ResultsError("stage=results no STRESS values in .frd")

    fiber_vm = None
    if fiber_point_mm is not None:
        target = fiber_point_mm
        nearest_id = min(
            frd.nodes,
            key=lambda nid: (frd.nodes[nid][0] - target[0]) ** 2
            + (frd.nodes[nid][1] - target[1]) ** 2
            + (frd.nodes[nid][2] - target[2]) ** 2,
        )
        comps = stress.values.get(nearest_id)
        if comps and len(comps) >= 6:
            fiber_vm = stress_MPa(von_mises(*comps[:6]))

    rf = parse_dat_forces(dat_path, "Nfixed")
    reaction = None
    if rf:
        reaction = (
            sum(v[0] for v in rf.values()),
            sum(v[1] for v in rf.values()),
            sum(v[2] for v in rf.values()),
        )

    return ScalarResults(
        max_von_mises=stress_MPa(best),
        max_von_mises_xyz_mm=loc,
        tip_deflection=length_mm(tip_defl),
        root_fiber_von_mises=fiber_vm,
        reaction_force_N=reaction,
    )
