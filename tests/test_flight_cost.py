"""Tests for the drone-flight inspection cost estimate."""

import pytest

from model.flight_cost import flight_cost


BASE = {
    "material": "3",
    "length_m": 20.0,
    "crosses": "HWY ROUTE 9",
    "lat": 42.36,
    "lon": -71.06,  # Boston depot
}


def test_longer_bridge_costs_more():
    short_cost, short_minutes, _ = flight_cost(BASE)
    long_cost, long_minutes, _ = flight_cost({**BASE, "length_m": 120.0})

    assert long_cost > short_cost
    assert long_minutes > short_minutes


def test_water_crossing_costs_more():
    ordinary_cost, _, _ = flight_cost(BASE)
    water_cost, _, fragment = flight_cost({**BASE, "crosses": "CHARLES RIVER"})

    assert water_cost > ordinary_cost
    assert "water/rail access factor" in fragment


def test_bridge_far_from_all_depots_costs_more():
    near_cost, near_minutes, near_reason = flight_cost(BASE)
    far_cost, far_minutes, far_reason = flight_cost(
        {**BASE, "lat": 41.25, "lon": -70.10}
    )

    assert far_cost > near_cost
    assert far_minutes > near_minutes
    assert "from Boston" in near_reason
    assert " km from " in far_reason


def test_missing_coordinates_use_sixty_km_default():
    cost, minutes, fragment = flight_cost({**BASE, "lat": None, "lon": None})

    # (1500 + 8*20) + 2*60 = $1,780; duration includes round-trip travel.
    assert cost == pytest.approx(1780.0)
    assert minutes == pytest.approx(174.0)
    assert "60 km default travel" in fragment
    assert fragment.startswith("drone est.")


def test_missing_length_uses_material_estimate_and_labels_it():
    cost, minutes, fragment = flight_cost(
        {k: v for k, v in BASE.items() if k != "length_m"}
    )

    assert cost > 0
    assert minutes > 0
    assert "length estimated" in fragment
