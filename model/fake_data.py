"""Synthetic panel generator for Part B.

Produces a contract-shaped panel (same columns and types as
``data/processed/panel.parquet``) for the biennial years 2012..2022 so the
model, validation, and tests can run before Part A's real data lands.

This is fake data. It is never used in the real pipeline or any published
result — it exists only to exercise code paths. It deliberately bakes in a
plausible decline process (older + worse bridges drop faster) so the model has
real signal to learn, plus some rebuilds and NBI "N" (NaN) values to exercise
the not-applicable handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEARS = [2012, 2014, 2016, 2018, 2020, 2022]

MATERIALS = ["Concrete", "Prestressed concrete", "Steel", "Timber"]
COUNTIES = [
    "Suffolk", "Middlesex", "Worcester", "Essex", "Norfolk", "Bristol",
    "Plymouth", "Hampden", "Barnstable", "Berkshire",
]
CROSSES_POOL = [
    "WATER CHARLES RIVER", "WATER MERRIMACK RIVER", "RAILROAD MBTA",
    "HWY ROUTE 9", "CRANBERRY BOG", "WATER NEPONSET RIVER",
    "RAILROAD CSX TRACK", "LOCAL STREET", "WATER STREAM", "INTERSTATE 90",
]

# Per-material base annual hazard of dropping a grade; worse condition and
# older age amplify it. Timber and steel decline faster than concrete here.
MATERIAL_BASE_HAZARD = {
    "Concrete": 0.04,
    "Prestressed concrete": 0.05,
    "Steel": 0.08,
    "Timber": 0.12,
}


def _annual_drop_prob(material: str, age: float, condition: float) -> float:
    base = MATERIAL_BASE_HAZARD.get(material, 0.06)
    age_factor = 1.0 + max(0.0, (age - 40.0)) / 100.0
    cond_factor = 1.0 + max(0.0, (6.0 - condition)) * 0.15
    return float(np.clip(base * age_factor * cond_factor, 0.0, 0.6))


def make_panel(n_bridges: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Return a typed, contract-shaped panel for years 2012..2022."""
    rng = np.random.default_rng(seed)

    # --- static per-bridge attributes --------------------------------------
    ids = [f"FAKE{idx:07d}" for idx in range(n_bridges)]
    material = rng.choice(MATERIALS, size=n_bridges, p=[0.35, 0.25, 0.25, 0.15])
    county = rng.choice(COUNTIES, size=n_bridges)
    crosses = rng.choice(CROSSES_POOL, size=n_bridges)
    year_built = rng.integers(1930, 2011, size=n_bridges)
    lat = 42.0 + rng.normal(0, 0.4, size=n_bridges)
    lon = -71.5 + rng.normal(0, 0.6, size=n_bridges)  # negative: western hemi
    adt = (rng.lognormal(mean=8.0, sigma=1.3, size=n_bridges)).astype(int)
    underwater = rng.random(n_bridges) < 0.15
    fracture_critical = rng.random(n_bridges) < 0.10
    # Culvert bridges (~12%) have N/A deck/super/sub and a culvert rating.
    is_culvert = rng.random(n_bridges) < 0.12
    insp_freq_mo = rng.choice([24, 48], size=n_bridges, p=[0.85, 0.15])
    # insp_cost present for ~60% of bridges; missing for the rest (-> Part B
    # placeholder).
    has_cost = rng.random(n_bridges) < 0.6
    insp_cost_static = np.where(
        has_cost, rng.uniform(2500, 15000, size=n_bridges), np.nan
    )

    # starting condition grade in 2012
    start_cond = rng.integers(4, 9, size=n_bridges).astype(float)

    # recon year: ~15% have an original reconstruction already on record
    year_recon = np.where(
        rng.random(n_bridges) < 0.15,
        rng.integers(2000, 2012, size=n_bridges),
        np.nan,
    ).astype(float)

    # --- evolve condition and reconstructions year by year ------------------
    cond = start_cond.copy()
    recon = year_recon.copy()
    rows = []
    for yi, year in enumerate(YEARS):
        for b in range(n_bridges):
            if yi > 0:
                age = year - year_built[b]
                # two years between snapshots -> apply drop twice-ish
                p = _annual_drop_prob(material[b], age, cond[b])
                p2 = 1.0 - (1.0 - p) ** 2
                if rng.random() < p2 and cond[b] > 0:
                    cond[b] -= 1.0
                # ~3% chance of a rebuild between snapshots: bumps condition and
                # records a new recon year.
                if rng.random() < 0.03:
                    cond[b] = min(9.0, cond[b] + rng.integers(1, 4))
                    recon[b] = year

            c = cond[b]
            if is_culvert[b]:
                deck = sup = sub = np.nan
                culvert = c
            else:
                # deck/super/sub jitter around the latent condition; the model's
                # condition = min(...) so keep them near c.
                deck = min(9.0, max(0.0, c + rng.integers(0, 2)))
                sup = c
                sub = min(9.0, max(0.0, c + rng.integers(0, 2)))
                culvert = np.nan

            # inspection date: within ~freq months before the snapshot
            offset_mo = int(rng.integers(0, insp_freq_mo[b] + 1))
            insp_date = pd.Timestamp(year=year, month=6, day=1) - pd.DateOffset(
                months=offset_mo
            )

            rows.append(
                {
                    "id": ids[b],
                    "year": int(year),
                    "county": county[b],
                    "year_built": int(year_built[b]),
                    "year_recon": recon[b],
                    "material": material[b],
                    "deck": deck,
                    "super": sup,
                    "sub": sub,
                    "culvert": culvert,
                    "adt": int(adt[b]),
                    "insp_date": insp_date,
                    "insp_freq_mo": int(insp_freq_mo[b]),
                    "fracture_critical": bool(fracture_critical[b]),
                    "underwater": bool(underwater[b]),
                    "lat": float(lat[b]),
                    "lon": float(lon[b]),
                    "crosses": crosses[b],
                    "insp_cost": float(insp_cost_static[b])
                    if not np.isnan(insp_cost_static[b])
                    else np.nan,
                }
            )

    panel = pd.DataFrame(rows)

    # --- enforce contract dtypes -------------------------------------------
    panel["id"] = panel["id"].astype("string")
    panel["year"] = panel["year"].astype("int64")
    panel["county"] = panel["county"].astype("string")
    panel["year_built"] = panel["year_built"].astype("Int64")
    panel["year_recon"] = panel["year_recon"].astype("Int64")
    panel["material"] = panel["material"].astype("string")
    for c in ["deck", "super", "sub", "culvert"]:
        panel[c] = panel[c].astype("float64")
    panel["adt"] = panel["adt"].astype("Int64")
    panel["insp_date"] = pd.to_datetime(panel["insp_date"])
    panel["insp_freq_mo"] = panel["insp_freq_mo"].astype("Int64")
    panel["fracture_critical"] = panel["fracture_critical"].astype("bool")
    panel["underwater"] = panel["underwater"].astype("bool")
    panel["lat"] = panel["lat"].astype("float64")
    panel["lon"] = panel["lon"].astype("float64")
    panel["crosses"] = panel["crosses"].astype("string")
    panel["insp_cost"] = panel["insp_cost"].astype("float64")

    return panel


if __name__ == "__main__":
    p = make_panel()
    print(f"panel: {len(p)} rows, {p['id'].nunique()} bridges, years {sorted(p['year'].unique())}")
    print(p.dtypes)
    print(p.head())
