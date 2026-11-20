"""Rebuild the offline dashboard, ranked tables, XAI layers, and validation report from bundled sources."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from integration import OFFICIAL_EVIDENCE, build_integrated_data
from screen import ROOT, evaluate, local_diffi, path_adjustment, run_screen, scenario_amount
from tender_screen import evaluate_tenders, run_tender_screen, tender_window_scenario
from xai_engine import (
    build_procurement_intelligence,
    compute_2d_decision_surface,
    compute_counterfactual_recourse,
    compute_imbalance_diffi,
    compute_xai_consensus,
    find_nearest_peers,
    run_dpwh_screen,
    statutory_procurement_checks,
)

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


def guarded_csv(frame: pd.DataFrame, path: Path) -> None:
    """Keep source text from becoming a formula when opened in a spreadsheet."""
    safe = frame.copy()
    for col in safe.select_dtypes(include=["object", "string"]):
        safe[col] = safe[col].astype("string").str.replace(
            r"^(?=\s*[=+\-@\t\r\n])", "'", regex=True,
        )
    safe.to_csv(path, index=False, float_format="%.12g", date_format="%Y-%m-%d")


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-normalized Python record dictionaries."""
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="s"))


def scored_awards(result: dict[str, Any], links: pd.DataFrame) -> pd.DataFrame:
    """Attach reconciled official PhilGEPS award and tender evidence to held-out awards."""
    return result["eval"].merge(
        links[["record_id", *OFFICIAL_EVIDENCE]], on="record_id", how="left", validate="one_to_one",
    )


def analyze(data_dir: Path = ROOT / "data") -> dict[str, Any]:
    """Run data reconciliation, award/tender/DPWH anomaly screens, and XAI evaluations."""
    data = build_integrated_data(data_dir)
    award = run_screen(data_dir)
    tender = run_tender_screen(data.tenders)
    dpwh_raw = pd.read_csv(data_dir / "tagum_dpwh.csv", dtype={"dpwh_contract_id": str})
    dpwh = run_dpwh_screen(dpwh_raw, data.flood_links)
    award_rows = scored_awards(award, data.award_links)
    coverage = {
        **data.counts,
        "evaluation_awards_with_official_id": int(award_rows.official_award_reference.notna().sum()),
        "evaluation_awards_with_official_tender": int(award_rows.approved_budget_php.notna().sum()),
        "award_flags_with_official_id": int((award_rows.review & award_rows.official_award_reference.notna()).sum()),
        "award_flags_with_official_tender": int((award_rows.review & award_rows.approved_budget_php.notna()).sum()),
    }
    dpwh_metrics = {
        "n_flags": int(dpwh["eval"]["review"].sum()),
        "n_extended_flags": int(dpwh["eval"]["extended_review"].sum()),
        "n_both": int((dpwh["eval"]["review"] & dpwh["eval"]["extended_review"]).sum()),
        "maximum_shap_error": float(dpwh["path_errors"].max()),
    }
    return {
        "data": data,
        "award": award,
        "tender": tender,
        "dpwh": dpwh,
        "award_rows": award_rows,
        "metrics": {
            "coverage": coverage,
            "award": evaluate(award),
            "tender": evaluate_tenders(tender),
            "dpwh": dpwh_metrics,
        },
    }


def _summarize_recourse_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute validity rate and mean sparsity across primary Isolation Forest flagged cases."""
    flagged_cfs = [
        row["counterfactual"]
        for row in values
        if row.get("review") and isinstance(row.get("counterfactual"), dict)
    ]
    if not flagged_cfs:
        return {"flagged_cases": 0, "valid_count": 0, "validity_rate": 0.0, "mean_sparsity": 0.0}
    valid_cfs = [cf for cf in flagged_cfs if cf.get("valid")]
    mean_sparsity = float(np.mean([cf["sparsity"] for cf in valid_cfs])) if valid_cfs else 0.0
    return {
        "flagged_cases": len(flagged_cfs),
        "valid_count": len(valid_cfs),
        "validity_rate": round(len(valid_cfs) / len(flagged_cfs), 4),
        "mean_sparsity": round(mean_sparsity, 3),
    }


def model_view(
    kind: str,
    result: dict[str, Any],
    metrics: dict[str, Any],
    rows: pd.DataFrame,
    fields: list[str],
    date_field: str,
    row_key: str,
    peer_group_col: str,
    scenario_fn: Any,
) -> dict[str, Any]:
    """Build enriched dashboard payload for a chronological anomaly screen (award or tender)."""
    test = result["views"]["Evaluation"]
    cutoff = float(result["cutoffs"]["Isolation Forest"])
    diffi = local_diffi(result["forest"], test)
    global_diffi, imb_diffi = compute_imbalance_diffi(
        result["forest"], test, outlier_mask=(rows["review"].to_numpy(dtype=bool))
    )
    feature_names = list(result["features"].values())
    feature_keys = list(result["features"].keys())
    values = records(rows[fields + ["tree_path_length"]])

    for i, record in enumerate(values):
        record[date_field] = str(record[date_field])[:10]
        record["feature_vector"] = [round(float(v), 5) for v in test[i]]
        record["tree_shap"] = result["effects"][i].round(7).tolist()
        record["local_diffi"] = diffi[i].round(7).tolist()
        record["imbalance_diffi"] = imb_diffi[i].round(7).tolist()
        record["exiffi"] = result["exiffi"][i].round(7).tolist()
        record["consensus"] = compute_xai_consensus(
            result["effects"][i],
            diffi[i],
            result["exiffi"][i],
            feature_names,
            bool(record["review"]),
            bool(record["extended_review"]),
        )
        record["statutory_flags"] = statutory_procurement_checks(kind, record)
        record["peer_ids"] = find_nearest_peers(
            rows,
            test,
            i,
            peer_group_col,
            str(rows.iloc[i][peer_group_col] or ""),
            row_key,
            exclude_id=str(record[row_key]),
            k=4,
        )
        if record["review"]:
            record["scenario"] = scenario_fn(rows.iloc[i], test[i], result)
        if record["review"] or record["extended_review"]:
            record["counterfactual"] = compute_counterfactual_recourse(kind, record, test[i], result)
        else:
            record["counterfactual"] = None

    values.sort(key=lambda row: (-row["anomaly_score"], row[row_key]))
    for rank, row in enumerate(values, start=1):
        row["rank"] = rank

    recourse_summary = _summarize_recourse_metrics(values)
    enriched_metrics = {**metrics, "counterfactual_recourse": recourse_summary}

    dates = pd.to_datetime(rows[date_field])
    monthly = (
        rows.groupby(dates.dt.strftime("%Y-%m"))
        .agg(scored=("review", "size"), reviews=("review", "sum"))
        .reset_index(names="month").to_dict(orient="records")
    )
    periods = result["processed"].groupby("split")[date_field].agg(rows="size", start="min", end="max")
    periods["usable_rows"] = pd.Series({name: len(frame) for name, frame in result["groups"].items()})
    periods["usable_rows"] = periods["usable_rows"].fillna(0).astype(int)
    path_c = float(path_adjustment(int(result["forest"].max_samples_)))
    surfaces = [
        compute_2d_decision_surface(result, 0, 1, grid_size=24),
        compute_2d_decision_surface(result, 0, 2, grid_size=24),
    ]

    return {
        "records": values,
        "monthly": monthly,
        "periods": records(periods.reset_index()),
        "tree_base": result["tree_base"],
        "path_c": round(path_c, 7),
        "features": feature_names,
        "feature_keys": feature_keys,
        "global_diffi": [round(float(v), 6) for v in global_diffi],
        "mean_abs_shap": [round(float(v), 6) for v in np.mean(np.abs(result["effects"]), axis=0)],
        "mean_local_diffi": [round(float(v), 6) for v in np.mean(diffi, axis=0)],
        "mean_exiffi": [round(float(v), 6) for v in np.mean(result["exiffi"], axis=0)],
        "surfaces": surfaces,
        "cutoffs": result["cutoffs"],
        "metrics": enriched_metrics,
        "calibration_review_count": int((result["calibration_scores"]["Isolation Forest"] > cutoff).sum()),
    }


def _dpwh_model_view(dpwh: dict[str, Any], dpwh_metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build enriched DPWH project list and DPWH model metadata for the dashboard."""
    x_eval = dpwh["views"]["Evaluation"]
    id_to_vec = {
        str(dpwh["eval"].iloc[i]["dpwh_contract_id"]): [round(float(v), 5) for v in x_eval[i]]
        for i in range(len(dpwh["eval"]))
    }
    project_records: list[dict[str, Any]] = []
    for rec in dpwh["records"]:
        item = dict(rec)
        cid = str(item["dpwh_contract_id"])
        item["feature_vector"] = id_to_vec.get(cid, [0.0] * 5)
        item.pop("waterfall", None)
        item.pop("recourse", None)
        item.pop("statutory_checks", None)
        project_records.append(item)

    recourse_summary = _summarize_recourse_metrics(project_records)
    cutoff = float(dpwh["cutoffs"]["Isolation Forest"])
    path_c = float(path_adjustment(int(dpwh["forest"].max_samples_)))
    dpwh_periods: list[dict[str, Any]] = []
    for split_name in ("Training", "Calibration", "Evaluation"):
        grp = dpwh["groups"][split_name]
        yrs = grp["infra_year"].dropna().astype(int)
        start_yr = f"{int(yrs.min())}-01-01" if len(yrs) else "2016-01-01"
        end_yr = f"{int(yrs.max())}-12-31" if len(yrs) else "2025-12-31"
        dpwh_periods.append({
            "split": split_name,
            "rows": int(len(grp)),
            "usable_rows": int(len(grp)),
            "start": start_yr,
            "end": end_yr,
        })
    dpwh_meta = {
        "records": project_records,
        "periods": dpwh_periods,
        "features": list(dpwh["features"].values()),
        "feature_keys": list(dpwh["features"].keys()),
        "tree_base": dpwh["tree_base"],
        "path_c": round(path_c, 7),
        "cutoffs": dpwh["cutoffs"],
        "global_diffi": dpwh["global_xai"]["global_diffi"],
        "mean_abs_shap": dpwh["global_xai"]["mean_abs_shap"],
        "mean_local_diffi": dpwh["global_xai"]["mean_local_diffi"],
        "mean_exiffi": dpwh["global_xai"]["mean_exiffi"],
        "surfaces": dpwh["surfaces"],
        "barangay_summary": dpwh["barangay_summary"],
        "calibration_review_count": int((dpwh["calibration_scores"]["Isolation Forest"] > cutoff).sum()),
        "metrics": {**dpwh_metrics, "counterfactual_recourse": recourse_summary},
    }
    return project_records, dpwh_meta


def dashboard_data(analysis: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    """Assemble the complete canonical JSON dataset embedded into Bantay_Kontrata_Dashboard.html."""
    stats = analysis["metrics"]
    award = model_view(
        "award",
        analysis["award"],
        stats["award"],
        analysis["award_rows"],
        AWARD_EXPORT,
        "award_date",
        "record_id",
        "category",
        scenario_amount,
    )
    tender = model_view(
        "tender",
        analysis["tender"],
        stats["tender"],
        analysis["tender"]["eval"],
        TENDER_EXPORT,
        "published_date",
        "bid_reference",
        "mode_group",
        tender_window_scenario,
    )
    project_records, dpwh_meta = _dpwh_model_view(analysis["dpwh"], stats["dpwh"])
    # Avoid duplicating 336 project rows twice in JSON: dpwh_meta exposes metadata; projects holds rows
    dpwh_summary = {k: v for k, v in dpwh_meta.items() if k != "records"}
    intelligence = build_procurement_intelligence(analysis["data"], analysis["award_rows"])
    boundaries = json.loads((data_dir / "tagum_barangays.geojson").read_text(encoding="utf-8"))
    stats["award"] = award["metrics"]
    stats["tender"] = tender["metrics"]
    stats["dpwh"] = dpwh_summary["metrics"]
    return {
        "award": award,
        "tender": tender,
        "dpwh": dpwh_summary,
        "intelligence": intelligence,
        "coverage": stats["coverage"],
        "projects": project_records,
        "barangays": boundaries,
    }


def export_ranked(rows: pd.DataFrame, fields: list[str], key: str, path: Path) -> None:
    """Write ranked evaluation records to CSV with spreadsheet formula protection."""
    ranked = rows.sort_values(["anomaly_score", key], ascending=[False, True])[fields].copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    guarded_csv(ranked, path)


def publish(analysis: dict[str, Any], output: Path, data_dir: Path = ROOT / "data") -> Path:
    """Render Bantay_Kontrata_Dashboard.html, ranked CSV tables, crosswalks, and validation.json."""
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


def main() -> None:
    """CLI entrypoint to rebuild all dashboard and review artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "review")
    args = parser.parse_args()
    analysis = analyze(args.data)
    print(publish(analysis, args.output, args.data))
    for label, model_name in [("awards", "award"), ("tenders", "tender"), ("DPWH projects", "dpwh")]:
        model = analysis[model_name]
        flags = analysis["metrics"][model_name]["n_flags"]
        print(f"{len(model['eval']):,} {label} scored. {flags:,} above the review cutoff.")


if __name__ == "__main__":
    main()
