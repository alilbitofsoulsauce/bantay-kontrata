"""Extract the local records and keep a provenance record for all three sources."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parent
PHILGEPS = (
    "https://huggingface.co/datasets/bettergovph/philgeps-data/resolve/"
    "b98de579579e8c7e79bac2a22ac7491549db6d0d/philgeps.parquet"
)
DPWH = "https://lovethephilippines.com/data/files/dpwh_projects.csv.gz"
BOUNDARIES = (
    "https://github.com/bendlikeabamboo/barangay-boundaries-repository/releases/"
    "download/v2026.4.13.0/barangay-boundaries-2023-10-24-hierarchical.zip"
)
BUYER = "CITY OF TAGUM, DAVAO DEL NORTE"


def extract_awards(path):
    with duckdb.connect() as connection:
        rows = connection.execute(
            """
            SELECT CAST(id AS VARCHAR) AS record_id,
                   CAST(award_date AS DATE) AS award_date,
                   reference_id, contract_no, award_title, notice_title,
                   awardee_name, area_of_delivery, business_category,
                   contract_amount, award_status
            FROM read_parquet(?)
            WHERE upper(trim(organization_name)) = ?
              AND award_date >= DATE '2017-01-01'
              AND award_date < DATE '2026-01-01'
            ORDER BY award_date, record_id
            """,
            [str(path), BUYER],
        ).fetchdf()
    for field in rows.select_dtypes(include=["object"]).columns:
        rows[field] = rows[field].astype("string").str.strip().replace("", pd.NA)
    eligible = rows[
        (rows.award_status.str.casefold() == "active") & (rows.contract_amount > 0)
    ].copy()
    # Same day and identical remaining published fields, including reference.
    # A shared reference alone never removes another award row.
    dedup = eligible.drop_duplicates(subset=eligible.columns.drop("record_id"))
    if dedup.record_id.isna().any() or dedup.record_id.duplicated().any():
        raise ValueError("Archive record IDs must be present and unique within the extract.")
    return dedup, {
        "input_rows": len(rows),
        "inactive_or_nonpositive_removed": len(rows) - len(eligible),
        "observational_duplicates_removed": len(eligible) - len(dedup),
        "rows_written": len(dedup),
    }


def extract_barangays(path):
    with ZipFile(path) as archive:
        collection = json.load(archive.open("barangays.geojson"))
    features = []
    for feature in collection["features"]:
        props = feature["properties"]
        if str(props.get("psgc_code", "")).startswith("1102319"):
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "name": props["psgc_name"],
                        "psgc": props["psgc_code"],
                    },
                    "geometry": feature["geometry"],
                }
            )
    if len(features) != 23:
        raise ValueError(f"Expected 23 Tagum barangays, found {len(features)}.")
    return {
        "type": "FeatureCollection",
        "features": sorted(features, key=lambda feature: feature["properties"]["name"]),
    }


def extract_projects(path, barangays):
    raw = pd.read_csv(path, low_memory=False, dtype={"location_psgc": "string"})
    province = raw.province.fillna("").str.contains(r"\bdavao del norte\b", case=False)
    tagum_title = raw.title.fillna("").str.contains(
        r"\b(?:tagum city|city of tagum)\b", case=False, regex=True
    )
    coords = raw[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
    lon_range = coords.lon.between(125.60, 126.00)
    lat_range = coords.lat.between(7.30, 7.70)
    selected = raw.loc[province & (tagum_title | (lon_range & lat_range))].copy()
    selected["lat"] = coords.loc[selected.index, "lat"]
    selected["lon"] = coords.loc[selected.index, "lon"]

    # A title match can be retained without trustworthy coordinates.
    polygons = [
        (feature["properties"]["name"], shape(feature["geometry"]))
        for feature in barangays["features"]
    ]
    barangay_names = {}
    for ix, row in selected.loc[selected.lon.notna() & selected.lat.notna()].iterrows():
        point = Point(row.lon, row.lat)
        matches = [name for name, polygon in polygons if polygon.covers(point)]
        if len(matches) == 1:
            barangay_names[ix] = matches[0]
    selected["barangay"] = pd.Series(barangay_names, dtype="string")
    selected["tagum_title"] = tagum_title.loc[selected.index]
    selected = selected.loc[selected.tagum_title | selected.barangay.notna()].copy()
    selected["location_basis"] = "Tagum title only"
    selected.loc[selected.barangay.notna(), "location_basis"] = "Point within a Tagum barangay"
    selected.loc[
        selected.barangay.notna() & selected.tagum_title, "location_basis"
    ] = "Tagum title and mapped point"

    cost = pd.to_numeric(selected.contract_cost_php, errors="coerce").where(lambda x: x > 0)
    budget = pd.to_numeric(selected.abc_php, errors="coerce").where(lambda x: x > 0)
    selected["contract_cost_php"] = cost
    selected["abc_php"] = budget
    selected["cost_abc_ratio"] = cost / budget

    started = pd.to_datetime(selected.start_date, errors="coerce")
    target = pd.to_datetime(selected.target_end_date, errors="coerce")
    actual = pd.to_datetime(selected.actual_end_date, errors="coerce")
    complete = selected.status.str.casefold().eq("completed")
    selected["variance_days"] = (actual - target).dt.days.where(
        complete & (target >= started) & (actual >= started)
    ).astype("Int64")

    fields = [
        "dpwh_contract_id", "title", "status", "type_of_work", "infra_year",
        "start_date", "target_end_date", "actual_end_date", "variance_days",
        "abc_php", "contract_cost_php", "cost_abc_ratio", "lat", "lon",
        "barangay", "location_basis", "source_url",
    ]
    projects = selected[fields].sort_values(["infra_year", "dpwh_contract_id"])
    if projects.dpwh_contract_id.isna().any() or projects.dpwh_contract_id.duplicated().any():
        raise ValueError("DPWH contract IDs must be present and unique within the extract.")
    return projects, {
        "input_rows": len(raw),
        "rows_written": len(projects),
        "mapped_rows": int(projects.barangay.notna().sum()),
        "valid_variance_rows": int(projects.variance_days.notna().sum()),
        "valid_ratio_rows": int(projects.cost_abc_ratio.notna().sum()),
    }


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--philgeps", type=Path, required=True)
    parser.add_argument("--dpwh", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    awards, award_counts = extract_awards(args.philgeps)
    barangays = extract_barangays(args.boundaries)
    projects, project_counts = extract_projects(args.dpwh, barangays)
    args.output.mkdir(exist_ok=True, parents=True)
    awards.to_csv(args.output / "tagum_awards.csv", index=False)
    projects.to_csv(args.output / "tagum_dpwh.csv", index=False)
    (args.output / "tagum_barangays.geojson").write_text(json.dumps(barangays), encoding="utf-8")
    provenance = {
        "extracted_utc": datetime.now(timezone.utc).date().isoformat(),
        "philgeps": {
            "source": PHILGEPS, "sha256": sha256(args.philgeps), "buyer": BUYER,
            **award_counts,
        },
        "dpwh": {
            "source": DPWH, "sha256": sha256(args.dpwh), **project_counts,
        },
        "boundaries": {
            "source": BOUNDARIES, "sha256": sha256(args.boundaries), "barangays": 23,
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
