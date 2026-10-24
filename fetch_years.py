import os
import urllib.request
import pandas as pd

YEARS = [2018, 2019, 2020, 2021, 2022]
os.makedirs("data", exist_ok=True)

# NBI item numbers, matched by column suffix so naming differences don't break us
WANT = {
    "id": "_008", "lat": "_016", "lon": "_017", "adt": "_029",
    "insp_date": "_090", "imp_cost": "_094",
    "deck": "_058", "super": "_059", "sub": "_060", "year_built": "_027",
}

def get(year):
    path = f"data/MA{year % 100}.txt"
    if not os.path.exists(path):
        url = f"https://www.fhwa.dot.gov/bridge/nbi/{year}/delimited/MA{year % 100}.txt"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(year, "FAILED:", e)
            return None
    return pd.read_csv(path, dtype=str, encoding="latin-1", low_memory=False)

def find_col(df, suffix):
    return next((c for c in df.columns if c.strip().endswith(suffix)), None)

dfs = {}
for y in YEARS:
    df = get(y)
    if df is None:
        continue
    dfs[y] = df
    print(f"\n{y}: {len(df)} rows, {len(df.columns)} cols")
    for name, suffix in WANT.items():
        col = find_col(df, suffix)
        fill = df[col].str.strip().ne("").mean() if col else 0
        print(f"  {name:10s} -> {col}  (non-empty {fill:.0%})")

if 2018 in dfs and 2022 in dfs:
    id18, id22 = find_col(dfs[2018], "_008"), find_col(dfs[2022], "_008")
    s18 = set(dfs[2018][id18].str.strip())
    s22 = set(dfs[2022][id22].str.strip())
    print(f"\nIDs in both 2018 and 2022: {len(s18 & s22)} "
          f"(2018 only {len(s18 - s22)}, 2022 only {len(s22 - s18)})")