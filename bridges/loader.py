# Person A loader. Cleans any NBI year into the fixed contract schema.
# Match columns by NBI item suffix so year renames do not break us.
# Read everything as strings, latin-1. Item positions verified against MA22.txt.
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data")

SCHEMA = [
    "structure_id", "year", "county", "material", "age", "adt",
    "deck", "super", "sub", "worst_rating", "insp_freq_months",
    "fracture_critical", "underwater", "year_recon", "rebuilt",
    "lat", "lon", "carries", "crosses", "improvement_cost",
]

# NBI item suffix we match (column ending in _SUFFIX). Verified in MA22.txt.
ITEMS = {
    "structure_id": "008",
    "county": "003",
    "crosses": "006A",
    "carries": "007",
    "lat": "016",
    "lon": "017",
    "year_built": "027",
    "adt": "029",
    "material": "043A",
    "deck": "058",
    "super": "059",
    "sub": "060",
    "insp_date": "090",
    "insp_freq_months": "091",
    "fracture_critical": "092A",
    "underwater": "092B",
    "improvement_cost": "094",
    "year_recon": "106",
}

# Massachusetts county FIPS codes to names. Verified against bridges/data.csv.
COUNTY_FIPS = {
    "001": "Barnstable", "003": "Berkshire", "005": "Bristol",
    "007": "Dukes", "009": "Essex", "011": "Franklin",
    "013": "Hampden", "015": "Hampshire", "017": "Middlesex",
    "019": "Nantucket", "021": "Norfolk", "023": "Plymouth",
    "025": "Suffolk", "027": "Worcester",
}

MA_LAT = (41.0, 43.0)
MA_LON = (-74.0, -69.5)

def raw_path(year):
    yy = str(year)[2:]
    return os.path.join(RAW_DIR, f"MA{yy}.txt")

def find_col(cols, suffix):
    suffix = suffix.upper()
    hits = [c for c in cols if c.upper().replace("-", "_").endswith("_" + suffix)]
    return hits[0] if hits else None

def is_blank(s):
    return s == "" or s.lower() in ("nan", "none")

def to_rating(val):
    s = str(val).strip().upper()
    if is_blank(s) or s == "N":
        return None
    try:
        n = int(float(s))
    except ValueError:
        return None
    return n if 0 <= n <= 9 else None

def to_int(val):
    s = str(val).strip()
    if is_blank(s):
        return None
    try:
        return int(float(s))
    except ValueError:
        return None

def dms_to_decimal(val, deg_width):
    # Packed degrees minutes seconds with two decimal seconds digits.
    # lat is 8 digits (DD MM SS ss), lon is 9 digits (DDD MM SS ss).
    s = "".join(ch for ch in str(val).strip() if ch.isdigit())
    if s == "" or set(s) == {"0"}:
        return None
    s = s.zfill(deg_width + 6)
    try:
        deg = int(s[0:deg_width])
        minutes = int(s[deg_width:deg_width + 2])
        seconds = int(s[deg_width + 2:deg_width + 4])
        frac = int(s[deg_width + 4:deg_width + 6]) / 100.0
        return deg + minutes / 60.0 + (seconds + frac) / 3600.0
    except ValueError:
        return None

def county_name(val):
    s = str(val).strip()
    if is_blank(s):
        return None
    return COUNTY_FIPS.get(s.zfill(3), s.zfill(3))

def load_year(year):
    df = pd.read_csv(
        raw_path(year), dtype=str, encoding="latin-1",
        sep=",", quotechar="'", low_memory=False,
    )
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    cols = list(df.columns)
    col, missing = {}, []
    for key, suffix in ITEMS.items():
        found = find_col(cols, suffix)
        col[key] = found
        if found is None:
            missing.append(f"{key} (item {suffix})")
    print(f"[{year}] MISSING: {', '.join(missing) if missing else 'none'}")

    def g(key):
        c = col.get(key)
        return df[c] if c is not None else pd.Series([None] * len(df))

    out = pd.DataFrame()
    out["structure_id"] = g("structure_id").astype(str).str.strip()
    out["year"] = year
    out["county"] = g("county").map(county_name)
    out["material"] = g("material")

    year_built = g("year_built").map(to_int)
    out["age"] = year_built.map(lambda yb: year - yb if yb else None)
    out["adt"] = g("adt").map(to_int)

    out["deck"] = g("deck").map(to_rating)
    out["super"] = g("super").map(to_rating)
    out["sub"] = g("sub").map(to_rating)
    out["worst_rating"] = out[["deck", "super", "sub"]].min(axis=1, skipna=True)

    out["insp_freq_months"] = g("insp_freq_months").map(to_int)

    fc = g("fracture_critical").astype(str).str.strip().str.upper()
    out["fracture_critical"] = fc.isin(["Y", "1"])
    uw = g("underwater").astype(str).str.strip().str.upper()
    out["underwater"] = uw.isin(["Y", "1"])

    out["year_recon"] = g("year_recon").map(to_int).fillna(0).astype(int)
    out["rebuilt"] = out["year_recon"] != 0  # refined in the panel

    lat = g("lat").map(lambda v: dms_to_decimal(v, 2))
    lon = g("lon").map(lambda v: dms_to_decimal(v, 3)).map(
        lambda x: -x if x is not None else None)
    out["lat"] = [v if (v is not None and MA_LAT[0] <= v <= MA_LAT[1]) else None for v in lat]
    out["lon"] = [v if (v is not None and MA_LON[0] <= v <= MA_LON[1]) else None for v in lon]

    out["carries"] = g("carries")
    out["crosses"] = g("crosses")
    out["improvement_cost"] = g("improvement_cost").map(to_int)

    out = out[SCHEMA]
    print(f"[{year}] rows={len(out)} worst_rating non-null={out['worst_rating'].notna().sum()} "
          f"lat non-null={out['lat'].notna().sum()}")
    return out

def build_panel(years=(2012, 2016, 2018, 2022)):
    frames = {y: load_year(y) for y in years}
    ordered = sorted(years)

    prev_ids, prev_y = None, None
    for y in ordered:
        ids = set(frames[y]["structure_id"])
        if prev_ids is not None:
            inter = ids & prev_ids
            base = len(prev_ids) or 1
            print(f"{prev_y}->{y}: match {len(inter)}/{base} "
                  f"({100 * len(inter) / base:.1f}%) added={len(ids - prev_ids)} "
                  f"dropped={len(prev_ids - ids)}")
        prev_ids, prev_y = ids, y

    panel = pd.concat([frames[y] for y in ordered], ignore_index=True)
    panel = panel.sort_values(["structure_id", "year"]).reset_index(drop=True)

    panel["rebuilt"] = False
    for sid, grp in panel.groupby("structure_id"):
        grp = grp.sort_values("year")
        prev_worst, prev_recon = None, None
        for idx, row in grp.iterrows():
            flag = False
            if (prev_worst is not None and pd.notna(row["worst_rating"])
                    and row["worst_rating"] > prev_worst):
                flag = True
            if prev_recon is not None and row["year_recon"] != prev_recon:
                flag = True
            if row["year_recon"] != 0 and prev_recon is None:
                flag = True
            panel.at[idx, "rebuilt"] = flag
            prev_worst = row["worst_rating"] if pd.notna(row["worst_rating"]) else prev_worst
            prev_recon = row["year_recon"]

    for y in ordered:
        n = int(panel[(panel["year"] == y) & (panel["rebuilt"])].shape[0])
        print(f"rebuilt in {y}: {n}")

    return panel

def main():
    os.makedirs(HERE, exist_ok=True)
    df22 = load_year(2022)
    df22.to_parquet(os.path.join(HERE, "bridges_2022.parquet"), index=False)
    print("wrote bridges/bridges_2022.parquet", df22.shape)

    panel = build_panel()
    panel.to_parquet(os.path.join(HERE, "panel.parquet"), index=False)
    print("wrote bridges/panel.parquet", panel.shape)

if __name__ == "__main__":
    main()
