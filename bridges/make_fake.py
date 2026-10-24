# Generate a fake bridges table matching the output schema exactly.
# Push bridges/fake.parquet first so B and C are unblocked.
import os
import random
import pandas as pd

random.seed(7)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fake.parquet")

SCHEMA = [
    "structure_id", "year", "county", "material", "age", "adt",
    "deck", "super", "sub", "worst_rating", "insp_freq_months",
    "fracture_critical", "underwater", "year_recon", "rebuilt",
    "lat", "lon", "carries", "crosses", "improvement_cost",
    "insp_date",
]

COUNTIES = ["Middlesex", "Worcester", "Essex", "Suffolk", "Norfolk", "Hampden"]
MATERIALS = ["Steel", "Concrete", "Prestressed concrete", "Wood", "Masonry"]

def rating_or_missing():
    if random.random() < 0.05:
        return None
    return random.randint(2, 9)

def make_row(i):
    year = 2022
    deck = rating_or_missing()
    sup = rating_or_missing()
    sub = rating_or_missing()
    present = [r for r in (deck, sup, sub) if r is not None]
    worst = min(present) if present else None

    year_built = random.randint(1920, 2015)
    year_recon = random.choice([0, 0, 0, random.randint(1980, 2018)])
    rebuilt = year_recon != 0

    return {
        "structure_id": f"FAKE{i:06d}",
        "year": year,
        "county": random.choice(COUNTIES),
        "material": random.choice(MATERIALS),
        "age": year - year_built,
        "adt": random.randint(50, 120000),
        "deck": deck,
        "super": sup,
        "sub": sub,
        "worst_rating": worst,
        "insp_freq_months": random.choice([12, 24, 48]),
        "fracture_critical": random.random() < 0.15,
        "underwater": random.random() < 0.1,
        "year_recon": year_recon,
        "rebuilt": rebuilt,
        "lat": round(random.uniform(41.4, 42.8), 6),
        "lon": round(random.uniform(-73.4, -70.0), 6),
        "carries": f"Route {random.randint(1, 495)}",
        "crosses": random.choice(["River", "Stream", "Railroad", "Highway"]),
        "improvement_cost": random.choice([0, 0, random.randint(50, 5000) * 1000]),
        "insp_date": pd.Timestamp(year=2021, month=random.randint(1, 12), day=1),
    }

def main():
    rows = [make_row(i) for i in range(200)]
    df = pd.DataFrame(rows, columns=SCHEMA)
    df.to_parquet(OUT, index=False)
    print("wrote", OUT, df.shape)
    print(list(df.columns))

if __name__ == "__main__":
    main()
