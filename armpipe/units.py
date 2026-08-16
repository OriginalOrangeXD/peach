"""SI <-> CalculiX mm-tonne-second conversions.

Two unit systems exist in this pipeline. Mixing them is the most likely
failure mode. Every conversion lives here. Stages pass ``Quantity`` values,
not bare floats.

SI (URDF / Pinocchio)
    m, kg, s, N, N·m, Pa, kg/m³, kg·m²

mm-tonne (gmsh / CalculiX)
    mm, tonne, s, N, N·mm, MPa, t/mm³, t·mm²

Force is newtons in both systems: 1 t·mm/s² = 1 N.
Time is seconds in both. Frequency is hertz in both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class UnitSystem(Enum):
    SI = "SI"
    MM_TONNE = "MM_TONNE"


class Kind(Enum):
    LENGTH = "length"
    MASS = "mass"
    FORCE = "force"
    MOMENT = "moment"
    STRESS = "stress"
    DENSITY = "density"
    YOUNGS = "youngs"
    MASS_INERTIA = "mass_inertia"
    AREA_INERTIA = "area_inertia"
    VOLUME = "volume"
    ACCEL = "accel"
    FREQUENCY = "frequency"
    TIME = "time"
    ANGLE = "angle"
    POISSON = "poisson"


# Multiply a quantity in MM_TONNE by these factors to obtain SI.
# Force, time, frequency, angle, Poisson are identical in both systems.
_MM_TONNE_TO_SI: dict[Kind, float] = {
    Kind.LENGTH: 1.0e-3,          # mm -> m
    Kind.MASS: 1.0e3,             # tonne -> kg
    Kind.FORCE: 1.0,              # N -> N
    Kind.MOMENT: 1.0e-3,          # N·mm -> N·m
    Kind.STRESS: 1.0e6,           # MPa -> Pa
    Kind.DENSITY: 1.0e12,         # t/mm³ -> kg/m³
    Kind.YOUNGS: 1.0e6,           # MPa -> Pa
    Kind.MASS_INERTIA: 1.0e-3,    # t·mm² -> kg·m²
    Kind.AREA_INERTIA: 1.0e-12,   # mm⁴ -> m⁴
    Kind.VOLUME: 1.0e-9,          # mm³ -> m³
    Kind.ACCEL: 1.0e-3,           # mm/s² -> m/s²
    Kind.FREQUENCY: 1.0,          # Hz -> Hz
    Kind.TIME: 1.0,               # s -> s
    Kind.ANGLE: 1.0,              # rad -> rad
    Kind.POISSON: 1.0,
}

_SI_UNIT = {
    Kind.LENGTH: "m",
    Kind.MASS: "kg",
    Kind.FORCE: "N",
    Kind.MOMENT: "N*m",
    Kind.STRESS: "Pa",
    Kind.DENSITY: "kg/m^3",
    Kind.YOUNGS: "Pa",
    Kind.MASS_INERTIA: "kg*m^2",
    Kind.AREA_INERTIA: "m^4",
    Kind.VOLUME: "m^3",
    Kind.ACCEL: "m/s^2",
    Kind.FREQUENCY: "Hz",
    Kind.TIME: "s",
    Kind.ANGLE: "rad",
    Kind.POISSON: "1",
}

_MM_TONNE_UNIT = {
    Kind.LENGTH: "mm",
    Kind.MASS: "t",
    Kind.FORCE: "N",
    Kind.MOMENT: "N*mm",
    Kind.STRESS: "MPa",
    Kind.DENSITY: "t/mm^3",
    Kind.YOUNGS: "MPa",
    Kind.MASS_INERTIA: "t*mm^2",
    Kind.AREA_INERTIA: "mm^4",
    Kind.VOLUME: "mm^3",
    Kind.ACCEL: "mm/s^2",
    Kind.FREQUENCY: "Hz",
    Kind.TIME: "s",
    Kind.ANGLE: "rad",
    Kind.POISSON: "1",
}

G_SI = 9.81  # m/s²
G_MM_TONNE = 9810.0  # mm/s²

# 6061-T6 handbook values. Density and E are the canonical pair used to
# prove the two unit systems agree (2700 kg/m³ == 2.70e-9 t/mm³).
AL6061_T6_DENSITY_KG_M3 = 2700.0
AL6061_T6_E_PA = 70.0e9
AL6061_T6_NU = 0.33
AL6061_T6_DENSITY_T_MM3 = 2.70e-9
AL6061_T6_E_MPA = 70000.0


class UnitError(ValueError):
    """Wrong unit system, kind, or conversion request."""


@dataclass(frozen=True)
class Quantity:
    """A scalar tagged with physical kind and unit system.

    Stages must pass ``Quantity`` (or the compound types below) across
    boundaries. Use ``.in_si()`` / ``.in_mm_tonne()`` to cross systems.
    """

    value: float
    kind: Kind
    system: UnitSystem

    @property
    def unit(self) -> str:
        table = _SI_UNIT if self.system is UnitSystem.SI else _MM_TONNE_UNIT
        return table[self.kind]

    def _as(self, system: UnitSystem) -> Quantity:
        if self.system is system:
            return self
        if system is UnitSystem.SI:
            return Quantity(self.value * _MM_TONNE_TO_SI[self.kind], self.kind, UnitSystem.SI)
        return Quantity(self.value / _MM_TONNE_TO_SI[self.kind], self.kind, UnitSystem.MM_TONNE)

    def in_si(self) -> Quantity:
        """Return this quantity in the SI system."""
        return self._as(UnitSystem.SI)

    def in_mm_tonne(self) -> Quantity:
        """Return this quantity in the CalculiX mm-tonne system."""
        return self._as(UnitSystem.MM_TONNE)

    def require(self, kind: Kind, system: UnitSystem) -> float:
        if self.kind is not kind:
            raise UnitError(f"expected kind {kind.value}, got {self.kind.value}")
        if self.system is not system:
            raise UnitError(
                f"expected {system.value} {kind.value}, got {self.system.value} "
                f"({self.value} {self.unit})"
            )
        return self.value

    def as_m(self) -> float:
        return self.in_si().require(Kind.LENGTH, UnitSystem.SI)

    def as_mm(self) -> float:
        return self.in_mm_tonne().require(Kind.LENGTH, UnitSystem.MM_TONNE)

    def as_m3(self) -> float:
        return self.in_si().require(Kind.VOLUME, UnitSystem.SI)

    def as_mm3(self) -> float:
        return self.in_mm_tonne().require(Kind.VOLUME, UnitSystem.MM_TONNE)

    def as_kg(self) -> float:
        return self.in_si().require(Kind.MASS, UnitSystem.SI)

    def as_tonne(self) -> float:
        return self.in_mm_tonne().require(Kind.MASS, UnitSystem.MM_TONNE)

    def as_N(self) -> float:
        return self.in_si().require(Kind.FORCE, UnitSystem.SI)

    def as_Nm(self) -> float:
        return self.in_si().require(Kind.MOMENT, UnitSystem.SI)

    def as_Nmm(self) -> float:
        return self.in_mm_tonne().require(Kind.MOMENT, UnitSystem.MM_TONNE)

    def as_Pa(self) -> float:
        if self.kind not in (Kind.STRESS, Kind.YOUNGS):
            raise UnitError(f"as_Pa expects stress or youngs, got {self.kind.value}")
        return self.in_si().value

    def as_MPa(self) -> float:
        if self.kind not in (Kind.STRESS, Kind.YOUNGS):
            raise UnitError(f"as_MPa expects stress or youngs, got {self.kind.value}")
        return self.in_mm_tonne().value

    def as_kg_m3(self) -> float:
        return self.in_si().require(Kind.DENSITY, UnitSystem.SI)

    def as_t_mm3(self) -> float:
        return self.in_mm_tonne().require(Kind.DENSITY, UnitSystem.MM_TONNE)

    def as_Hz(self) -> float:
        return self.in_si().require(Kind.FREQUENCY, UnitSystem.SI)

    def __mul__(self, scale: float) -> Quantity:
        if not isinstance(scale, (int, float)):
            return NotImplemented
        return Quantity(self.value * float(scale), self.kind, self.system)

    __rmul__ = __mul__

    def __truediv__(self, scale: float) -> Quantity:
        if not isinstance(scale, (int, float)):
            return NotImplemented
        return Quantity(self.value / float(scale), self.kind, self.system)

    def __neg__(self) -> Quantity:
        return Quantity(-self.value, self.kind, self.system)

    def __repr__(self) -> str:
        return f"Quantity({self.value!r} {self.unit}, {self.system.value})"


def q_si(value: float, kind: Kind) -> Quantity:
    return Quantity(float(value), kind, UnitSystem.SI)


def q_mm_tonne(value: float, kind: Kind) -> Quantity:
    return Quantity(float(value), kind, UnitSystem.MM_TONNE)


def length_m(value: float) -> Quantity:
    return q_si(value, Kind.LENGTH)


def length_mm(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.LENGTH)


def mass_kg(value: float) -> Quantity:
    return q_si(value, Kind.MASS)


def mass_tonne(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.MASS)


def force_N(value: float) -> Quantity:
    return q_si(value, Kind.FORCE)


def moment_Nm(value: float) -> Quantity:
    return q_si(value, Kind.MOMENT)


def moment_Nmm(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.MOMENT)


def stress_Pa(value: float) -> Quantity:
    return q_si(value, Kind.STRESS)


def stress_MPa(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.STRESS)


def density_kg_m3(value: float) -> Quantity:
    return q_si(value, Kind.DENSITY)


def density_t_mm3(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.DENSITY)


def youngs_Pa(value: float) -> Quantity:
    return q_si(value, Kind.YOUNGS)


def youngs_MPa(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.YOUNGS)


def volume_m3(value: float) -> Quantity:
    return q_si(value, Kind.VOLUME)


def volume_mm3(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.VOLUME)


def mass_inertia_kg_m2(value: float) -> Quantity:
    return q_si(value, Kind.MASS_INERTIA)


def mass_inertia_t_mm2(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.MASS_INERTIA)


def area_inertia_m4(value: float) -> Quantity:
    return q_si(value, Kind.AREA_INERTIA)


def area_inertia_mm4(value: float) -> Quantity:
    return q_mm_tonne(value, Kind.AREA_INERTIA)


def accel_m_s2(value: float) -> Quantity:
    return q_si(value, Kind.ACCEL)


def frequency_Hz(value: float) -> Quantity:
    return q_si(value, Kind.FREQUENCY)


def poisson(value: float) -> Quantity:
    return q_si(value, Kind.POISSON)


# --- named crossing functions (required: unit system in the name) ---

def length_m_to_mm(q: Quantity) -> Quantity:
    """SI metres -> CalculiX millimetres."""
    q.require(Kind.LENGTH, UnitSystem.SI)
    return q.in_mm_tonne()


def length_mm_to_m(q: Quantity) -> Quantity:
    """CalculiX millimetres -> SI metres."""
    q.require(Kind.LENGTH, UnitSystem.MM_TONNE)
    return q.in_si()


def mass_kg_to_tonne(q: Quantity) -> Quantity:
    """SI kilograms -> CalculiX tonnes."""
    q.require(Kind.MASS, UnitSystem.SI)
    return q.in_mm_tonne()


def mass_tonne_to_kg(q: Quantity) -> Quantity:
    """CalculiX tonnes -> SI kilograms."""
    q.require(Kind.MASS, UnitSystem.MM_TONNE)
    return q.in_si()


def density_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI kg/m³ -> CalculiX t/mm³."""
    q.require(Kind.DENSITY, UnitSystem.SI)
    return q.in_mm_tonne()


def density_mm_tonne_to_si(q: Quantity) -> Quantity:
    """CalculiX t/mm³ -> SI kg/m³."""
    q.require(Kind.DENSITY, UnitSystem.MM_TONNE)
    return q.in_si()


def youngs_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI Pa -> CalculiX MPa."""
    q.require(Kind.YOUNGS, UnitSystem.SI)
    return q.in_mm_tonne()


def youngs_mm_tonne_to_si(q: Quantity) -> Quantity:
    """CalculiX MPa -> SI Pa."""
    q.require(Kind.YOUNGS, UnitSystem.MM_TONNE)
    return q.in_si()


def stress_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI Pa -> CalculiX MPa."""
    q.require(Kind.STRESS, UnitSystem.SI)
    return q.in_mm_tonne()


def stress_mm_tonne_to_si(q: Quantity) -> Quantity:
    """CalculiX MPa -> SI Pa."""
    q.require(Kind.STRESS, UnitSystem.MM_TONNE)
    return q.in_si()


def force_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI newtons -> CalculiX newtons (identity)."""
    q.require(Kind.FORCE, UnitSystem.SI)
    return q.in_mm_tonne()


def moment_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI N·m -> CalculiX N·mm."""
    q.require(Kind.MOMENT, UnitSystem.SI)
    return q.in_mm_tonne()


def moment_mm_tonne_to_si(q: Quantity) -> Quantity:
    """CalculiX N·mm -> SI N·m."""
    q.require(Kind.MOMENT, UnitSystem.MM_TONNE)
    return q.in_si()


def mass_inertia_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI kg·m² -> CalculiX t·mm²."""
    q.require(Kind.MASS_INERTIA, UnitSystem.SI)
    return q.in_mm_tonne()


def volume_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI m³ -> CalculiX mm³."""
    q.require(Kind.VOLUME, UnitSystem.SI)
    return q.in_mm_tonne()


def volume_mm_tonne_to_si(q: Quantity) -> Quantity:
    """CalculiX mm³ -> SI m³."""
    q.require(Kind.VOLUME, UnitSystem.MM_TONNE)
    return q.in_si()


def accel_si_to_mm_tonne(q: Quantity) -> Quantity:
    """SI m/s² -> CalculiX mm/s²."""
    q.require(Kind.ACCEL, UnitSystem.SI)
    return q.in_mm_tonne()


@dataclass(frozen=True)
class Vec3:
    """A 3-vector whose components share one kind and one unit system."""

    x: Quantity
    y: Quantity
    z: Quantity

    def __post_init__(self) -> None:
        kinds = {self.x.kind, self.y.kind, self.z.kind}
        systems = {self.x.system, self.y.system, self.z.system}
        if len(kinds) != 1 or len(systems) != 1:
            raise UnitError("Vec3 components must share kind and unit system")

    @property
    def kind(self) -> Kind:
        return self.x.kind

    @property
    def system(self) -> UnitSystem:
        return self.x.system

    def in_si(self) -> Vec3:
        return Vec3(self.x.in_si(), self.y.in_si(), self.z.in_si())

    def in_mm_tonne(self) -> Vec3:
        return Vec3(self.x.in_mm_tonne(), self.y.in_mm_tonne(), self.z.in_mm_tonne())

    def values(self) -> tuple[float, float, float]:
        return (self.x.value, self.y.value, self.z.value)

    def as_array_si(self) -> tuple[float, float, float]:
        s = self.in_si()
        return s.values()

    def as_array_mm_tonne(self) -> tuple[float, float, float]:
        s = self.in_mm_tonne()
        return s.values()


def vec3(values: Iterable[float], kind: Kind, system: UnitSystem) -> Vec3:
    x, y, z = (float(v) for v in values)
    return Vec3(
        Quantity(x, kind, system),
        Quantity(y, kind, system),
        Quantity(z, kind, system),
    )


@dataclass(frozen=True)
class Wrench:
    """Force + moment in one frame and one unit system.

    ``frame`` is a human-readable frame name (e.g. ``"joint_2_parent"``).
    The convention is documented by the producer (``dynamics.py``).
    """

    force: Vec3
    moment: Vec3
    frame: str

    def __post_init__(self) -> None:
        if self.force.kind is not Kind.FORCE:
            raise UnitError("Wrench.force must be Kind.FORCE")
        if self.moment.kind is not Kind.MOMENT:
            raise UnitError("Wrench.moment must be Kind.MOMENT")
        if self.force.system is not self.moment.system:
            raise UnitError("Wrench force and moment must share a unit system")

    @property
    def system(self) -> UnitSystem:
        return self.force.system

    def in_si(self) -> Wrench:
        """Return this wrench in SI (N, N·m)."""
        return Wrench(self.force.in_si(), self.moment.in_si(), self.frame)

    def in_mm_tonne(self) -> Wrench:
        """Return this wrench in mm-tonne (N, N·mm)."""
        return Wrench(self.force.in_mm_tonne(), self.moment.in_mm_tonne(), self.frame)


@dataclass(frozen=True)
class Material:
    """Linear-elastic isotropic material in one unit system."""

    name: str
    density: Quantity
    E: Quantity
    nu: Quantity

    def __post_init__(self) -> None:
        if self.density.kind is not Kind.DENSITY:
            raise UnitError("Material.density must be Kind.DENSITY")
        if self.E.kind is not Kind.YOUNGS:
            raise UnitError("Material.E must be Kind.YOUNGS")
        if self.nu.kind is not Kind.POISSON:
            raise UnitError("Material.nu must be Kind.POISSON")
        systems = {self.density.system, self.E.system}
        if len(systems) != 1:
            raise UnitError("Material density and E must share a unit system")

    @property
    def system(self) -> UnitSystem:
        return self.density.system

    def in_si(self) -> Material:
        """Return density in kg/m³ and E in Pa."""
        return Material(self.name, self.density.in_si(), self.E.in_si(), self.nu)

    def in_mm_tonne(self) -> Material:
        """Return density in t/mm³ and E in MPa."""
        return Material(self.name, self.density.in_mm_tonne(), self.E.in_mm_tonne(), self.nu)


def al6061_t6_si() -> Material:
    return Material(
        name="6061-T6",
        density=density_kg_m3(AL6061_T6_DENSITY_KG_M3),
        E=youngs_Pa(AL6061_T6_E_PA),
        nu=poisson(AL6061_T6_NU),
    )


def al6061_t6_mm_tonne() -> Material:
    return Material(
        name="6061-T6",
        density=density_t_mm3(AL6061_T6_DENSITY_T_MM3),
        E=youngs_MPa(AL6061_T6_E_MPA),
        nu=poisson(AL6061_T6_NU),
    )


def mass_from_volume_si(volume: Quantity, material: Material) -> Quantity:
    """mass_kg = volume_m³ × density_kg/m³. Inputs must be SI."""
    v = volume.in_si().require(Kind.VOLUME, UnitSystem.SI)
    rho = material.in_si().density.require(Kind.DENSITY, UnitSystem.SI)
    return mass_kg(v * rho)


def geometric_inertia_mm5_to_si(I_mm5, density: Quantity):
    """CAD geometric inertia (mm⁵, density=1) → mass inertia in kg·m².

    ``I_kg_m2 = I_mm5 * rho_kg_m3 * 1e-15``.
    """
    import numpy as np

    rho = density.in_si().require(Kind.DENSITY, UnitSystem.SI)
    return np.asarray(I_mm5, dtype=float) * rho * 1.0e-15
