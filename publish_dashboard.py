"""Rebuild the offline dashboard, ranked tables and validation report from the bundled sources."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from integration import OFFICIAL_EVIDENCE, build_integrated_data
from screen import ROOT, evaluate, local_diffi, run_screen, scenario_amount
from tender_screen import evaluate_tenders, run_tender_screen, tender_window_scenario

AWARD_EXPORT = [
    "record_id", "reference_id", "contract_no", "award_date", "award_title", "notice_title",
    "awardee_name", "category", "contract_amount", "peer_median_php", "peer_scope",
    "vendor_share_pct", "vendor_category_30d", "days_to_quarter_end", "category_365d",
    "anomaly_score", "review", "top_tree_shap", "extended_score", "extended_review",
    *OFFICIAL_EVIDENCE,
]
TENDER_EXPORT = [
    "bid_reference", "solicitation_number", "notice_title", "classification", "procurement_mode",
    "mode_group", "published_date", "closing_date", "bid_notice_status", "approved_budget_php",
    "notice_window_days", "budget_peer_median_php", "window_peer_median_days", "budget_peer_group",
    "window_peer_group", "prior_mode_share", "prior_class_share", "days_to_quarter_end",
    "official_notice_link", "observed_official_awards", "observed_official_award_total_php",
    "observed_award_total_to_abc", "anomaly_score", "review", "top_tree_shap",
    "extended_score", "extended_review",
]


def guarded_csv(frame, path):
    """Keep source text from becoming a formula when opened in a spreadsheet."""
    safe = frame.copy()
    for col in safe.select_dtypes(include=["object", "string"]):
        safe[col] = safe[col].astype("string").str.replace(
            r"^(?=\s*[=+\-@\t\r\n])", "'", regex=True,
        )
    safe.to_csv(path, index=False, float_format="%.12g", date_format="%Y-%m-%d")


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="s"))


def scored_awards(result, links):
    return result["eval"].merge(
        links[["record_id", *OFFICIAL_EVIDENCE]], on="record_id", how="left", validate="one_to_one",
    )


def analyze(data_dir=ROOT / "data"):
    data = build_integrated_data(data_dir)
    award = run_screen(data_dir)
    tender = run_tender_screen(data.tenders)
    award_rows = scored_awards(award, data.award_links)
    coverage = {
        **data.counts,
        "evaluation_awards_with_official_id": int(award_rows.official_award_reference.notna().sum()),
        "evaluation_awards_with_official_tender": int(award_rows.approved_budget_php.notna().sum()),
        "award_flags_with_official_id": int((award_rows.review & award_rows.official_award_reference.notna()).sum()),
        "award_flags_with_official_tender": int((award_rows.review & award_rows.approved_budget_php.notna()).sum()),
    }
    return {
        "data": data,
        "award": award,
        "tender": tender,
        "award_rows": award_rows,
        "metrics": {"coverage": coverage, "award": evaluate(award), "tender": evaluate_tenders(tender)},
    }


def model_view(result, metrics, rows, fields, date_field, row_key, scenario_fn):
    test = result["views"]["Evaluation"]
    diffi = local_diffi(result["forest"], test)
    values = records(rows[fields + ["tree_path_length"]])
    for i, record in enumerate(values):
        record[date_field] = record[date_field][:10]
        record["tree_shap"] = result["effects"][i].round(7).tolist()
        record["local_diffi"] = diffi[i].round(7).tolist()
        record["exiffi"] = result["exiffi"][i].round(7).tolist()
        if record["review"]:
            record["scenario"] = scenario_fn(rows.iloc[i], test[i], result)
    values.sort(key=lambda row: (-row["anomaly_score"], row[row_key]))
    for rank, row in enumerate(values, start=1):
        row["rank"] = rank

    dates = pd.to_datetime(rows[date_field])
    monthly = (
        rows.groupby(dates.dt.strftime("%Y-%m"))
        .agg(scored=("review", "size"), reviews=("review", "sum"))
        .reset_index(names="month").to_dict(orient="records")
    )
    periods = result["processed"].groupby("split")[date_field].agg(rows="size", start="min", end="max")
    periods["usable_rows"] = pd.Series({name: len(frame) for name, frame in result["groups"].items()})
    periods["usable_rows"] = periods["usable_rows"].fillna(0).astype(int)
    cutoff = result["cutoffs"]["Isolation Forest"]
    return {
        "records": values, "monthly": monthly, "periods": records(periods.reset_index()),
        "tree_base": result["tree_base"], "features": list(result["features"].values()),
        "cutoffs": result["cutoffs"], "metrics": metrics,
        "calibration_review_count": int((result["calibration_scores"]["Isolation Forest"] > cutoff).sum()),
    }


def dashboard_data(analysis, data_dir):
    stats = analysis["metrics"]
    award = model_view(
        analysis["award"], stats["award"], analysis["award_rows"], AWARD_EXPORT,
        "award_date", "record_id", scenario_amount,
    )
    tender = model_view(
        analysis["tender"], stats["tender"], analysis["tender"]["eval"], TENDER_EXPORT,
        "published_date", "bid_reference", tender_window_scenario,
    )
    projects = pd.read_csv(data_dir / "tagum_dpwh.csv")
    floods = analysis["data"].flood_links[[
        "dpwh_contract_id", "flood_map_project_id", "flood_map_component_id", "flood_map_municipality",
    ]]
    projects = projects.merge(floods, on="dpwh_contract_id", how="left", validate="one_to_one")
    boundaries = json.loads((data_dir / "tagum_barangays.geojson").read_text(encoding="utf-8"))
    return {
        "award": award, "tender": tender, "coverage": stats["coverage"], "projects": records(projects),
        "barangays": boundaries,
    }


def export_ranked(rows, fields, key, path):
    ranked = rows.sort_values(["anomaly_score", key], ascending=[False, True])[fields].copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    guarded_csv(ranked, path)


def publish(analysis, output, data_dir=ROOT / "data"):
    template = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
    workflow = (ROOT / "review_workflow.js").read_text(encoding="utf-8")
    data = dashboard_data(analysis, data_dir)
    snapshot_bytes = json.dumps(
        data, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")
    data["snapshot_id"] = sha256(snapshot_bytes).hexdigest()
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    output.mkdir(exist_ok=True, parents=True)
    dashboard = output / "Bantay_Kontrata_Dashboard.html"
    html = template.replace("__REVIEW_WORKFLOW__", workflow).replace(
        "__DASHBOARD_DATA__", payload.replace("<", r"\u003c"),
    )
    dashboard.write_text(html, encoding="utf-8")
    export_ranked(analysis["award_rows"], AWARD_EXPORT, "record_id", output / "tagum_2025_ranked_awards.csv")
    tender_rows = analysis["tender"]["eval"]
    tender_years = pd.to_datetime(tender_rows.published_date).dt.year
    first, last = tender_years.min(), tender_years.max()
    period = str(first) if first == last else f"{first}-{last}"
    export_ranked(
        tender_rows, TENDER_EXPORT, "bid_reference", output / f"tagum_{period}_ranked_tenders.csv",
    )
    integrated = analysis["data"]
    for file, frame in [
        ("tagum_award_register.csv", integrated.award_register),
        ("tagum_award_links.csv", integrated.award_links),
        ("tagum_possible_award_overlaps.csv", integrated.unresolved_links),
    ]:
        guarded_csv(frame, output / file)
    (output / "validation.json").write_text(
        json.dumps({**analysis["metrics"], "snapshot_id": data["snapshot_id"]}, indent=2, ensure_ascii=False)
        + "\n", encoding="utf-8",
    )
    return dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "review")
    args = parser.parse_args()
    analysis = analyze(args.data)
    print(publish(analysis, args.output, args.data))
    for label, model_name in [("awards", "award"), ("tenders", "tender")]:
        model = analysis[model_name]
        flags = analysis["metrics"][model_name]["n_flags"]
        print(f"{len(model['eval']):,} {label} scored. {flags:,} above the review cutoff.")


if __name__ == "__main__":
    main()
