"""Unit-system conversions. These numbers are the pipeline's contract."""

import pytest

from armpipe.units import (
    AL6061_T6_DENSITY_KG_M3,
    AL6061_T6_DENSITY_T_MM3,
    AL6061_T6_E_MPA,
    AL6061_T6_E_PA,
    G_MM_TONNE,
    G_SI,
    Kind,
    UnitError,
    UnitSystem,
    Wrench,
    accel_m_s2,
    accel_si_to_mm_tonne,
    al6061_t6_mm_tonne,
    al6061_t6_si,
    density_kg_m3,
    density_si_to_mm_tonne,
    force_N,
    force_si_to_mm_tonne,
    length_m,
    length_m_to_mm,
    length_mm,
    length_mm_to_m,
    mass_from_volume_si,
    mass_inertia_kg_m2,
    mass_inertia_si_to_mm_tonne,
    mass_kg,
    mass_kg_to_tonne,
    moment_Nm,
    moment_si_to_mm_tonne,
    stress_MPa,
    stress_Pa,
    stress_si_to_mm_tonne,
    vec3,
    volume_mm3,
    youngs_Pa,
    youngs_si_to_mm_tonne,
)


def test_length_m_to_mm():
    q = length_m_to_mm(length_m(1.0))
    assert q.system is UnitSystem.MM_TONNE
    assert q.as_mm() == pytest.approx(1000.0)


def test_length_mm_to_m():
    q = length_mm_to_m(length_mm(200.0))
    assert q.as_m() == pytest.approx(0.2)


def test_mass_kg_to_tonne():
    q = mass_kg_to_tonne(mass_kg(1.0))
    assert q.as_tonne() == pytest.approx(1.0e-3)


def test_density_6061_handbook_pair():
    """2700 kg/m³ must be exactly 2.70e-9 t/mm³."""
    converted = density_si_to_mm_tonne(density_kg_m3(AL6061_T6_DENSITY_KG_M3))
    assert converted.as_t_mm3() == pytest.approx(AL6061_T6_DENSITY_T_MM3, rel=0, abs=1e-21)
    assert al6061_t6_si().in_mm_tonne().density.as_t_mm3() == pytest.approx(
        al6061_t6_mm_tonne().density.as_t_mm3()
    )


def test_youngs_6061_handbook_pair():
    """70 GPa must be exactly 70000 MPa."""
    converted = youngs_si_to_mm_tonne(youngs_Pa(AL6061_T6_E_PA))
    assert converted.in_mm_tonne().value == pytest.approx(AL6061_T6_E_MPA)
    assert al6061_t6_si().in_mm_tonne().E.as_MPa() == pytest.approx(
        al6061_t6_mm_tonne().E.as_MPa()
    )


def test_force_is_identity_across_systems():
    q = force_si_to_mm_tonne(force_N(50.0))
    assert q.as_N() == pytest.approx(50.0)
    assert q.system is UnitSystem.MM_TONNE
    assert q.unit == "N"


def test_moment_nm_to_nmm():
    q = moment_si_to_mm_tonne(moment_Nm(1.0))
    assert q.as_Nmm() == pytest.approx(1000.0)


def test_stress_pa_to_mpa():
    q = stress_si_to_mm_tonne(stress_Pa(30.0e6))
    assert q.as_MPa() == pytest.approx(30.0)


def test_mass_inertia_kg_m2_to_t_mm2():
    """1 kg·m² = 0.001 t × 1e6 mm² = 1000 t·mm²."""
    q = mass_inertia_si_to_mm_tonne(mass_inertia_kg_m2(1.0))
    assert q.in_mm_tonne().value == pytest.approx(1000.0)


def test_gravity_si_to_mm_tonne():
    q = accel_si_to_mm_tonne(accel_m_s2(G_SI))
    assert q.in_mm_tonne().value == pytest.approx(G_MM_TONNE)


def test_round_trip_length():
    original = length_m(0.337)
    assert original.in_mm_tonne().in_si().as_m() == pytest.approx(0.337)


def test_crossing_function_rejects_wrong_kind():
    with pytest.raises(UnitError):
        length_m_to_mm(mass_kg(1.0))


def test_crossing_function_rejects_wrong_system():
    with pytest.raises(UnitError):
        length_m_to_mm(length_mm(10.0))


def test_require_rejects_bare_system_mismatch():
    with pytest.raises(UnitError, match="expected SI"):
        length_mm(10.0).require(Kind.LENGTH, UnitSystem.SI)


def test_mass_from_volume_matches_handbook_block():
    # 100 mm × 20 mm × 10 mm of 6061-T6
    volume = volume_mm3(100.0 * 20.0 * 10.0).in_si()
    mass = mass_from_volume_si(volume, al6061_t6_si())
    assert mass.as_kg() == pytest.approx(2.0e-5 * 2700.0)  # 20000 mm³ = 2e-5 m³ × 2700


def test_wrench_converts_force_and_moment_together():
    w = Wrench(
        force=vec3((1.0, 0.0, -2.0), Kind.FORCE, UnitSystem.SI),
        moment=vec3((0.0, 3.0, 0.0), Kind.MOMENT, UnitSystem.SI),
        frame="joint_1_parent",
    )
    ccx = w.in_mm_tonne()
    assert ccx.force.as_array_mm_tonne() == pytest.approx((1.0, 0.0, -2.0))
    assert ccx.moment.as_array_mm_tonne() == pytest.approx((0.0, 3000.0, 0.0))
    assert ccx.frame == "joint_1_parent"


def test_stress_mpa_round_trip():
    assert stress_MPa(276.0).in_si().in_mm_tonne().as_MPa() == pytest.approx(276.0)


def test_quantity_repr_names_the_unit():
    assert "mm" in repr(length_mm(200.0))
    assert "m" in repr(length_m(0.2))
