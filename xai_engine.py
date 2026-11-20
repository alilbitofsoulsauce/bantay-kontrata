"""Four-Pillar Explainable AI (XAI) & Procurement Intelligence Engine for Bantay-Kontrata Tagum."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from integration import OFFICIAL_EVIDENCE, IntegratedData, build_integrated_data
from screen import (
    FEATURES,
    MIN_PEERS,
    MISSING_CATEGORY,
    ROOT,
    SEED,
    TREES,
    ExtendedForest,
    explain_paths,
    local_diffi as compute_local_diffi,
    path_adjustment,
    run_screen,
    score_cutoff,
)
from tender_screen import TENDER_FEATURES, comparable_mode, run_tender_screen

# --- Domain & Statutory Constants (RA 9184 & RA 12009 IRR Feb 2025) ---
DPWH_FEATURES: dict[str, str] = {
    "cost_abc_ratio_feat": "Contract cost / ABC ratio",
    "abc_log_ratio": "Budget vs. work-type peers",
    "variance_days_feat": "Schedule delay variance (days)",
    "planned_duration_days": "Planned contract duration (days)",
    "barangay_project_density": "Barangay project concentration",
}

RA12009_SVP_MIN_PHP = 1_600_000.0
RA12009_SVP_MAX_PHP = 2_000_000.0
RA9184_SVP_MIN_PHP = 850_000.0
RA9184_SVP_MAX_PHP = 1_000_000.0
PUBLIC_BID_MIN_WINDOW_DAYS = 7.0
ALT_MODE_MIN_WINDOW_DAYS = 3.0
TIGHT_ABC_MIN_RATIO = 0.995
TIGHT_ABC_MAX_RATIO = 1.0005
VENDOR_BURST_30D_MIN = 3
VENDOR_CONCENTRATION_MIN_PCT = 5.0
QUARTER_END_RUSH_MAX_DAYS = 5
DPWH_DELAY_MIN_DAYS = 30.0

DEFAULT_OUTLIER_QUANTILE = 0.95
DPWH_MIN_WORK_TYPE_PEERS = 10
DPWH_DEFAULT_RATIO_FALLBACK = 0.996
DPWH_DEFAULT_DURATION_DAYS = 120.0
DPWH_MIN_CALIBRATION_ROWS = 30

SPLIT_CLUSTER_WINDOW_DAYS = 30
SPLIT_CLUSTER_MIN_AWARDS = 3
SPLIT_CLUSTER_MIN_TOTAL_PHP = 500_000.0
SPLIT_CLUSTER_SVP_LOW_PHP = 800_000.0
SPLIT_CLUSTER_SVP_HIGH_PHP = 2_500_000.0
MAX_TOP_VENDORS = 60
MAX_SPLIT_CLUSTERS = 40
MAX_OVERLAP_LEADS = 120


def _opt_float(value: Any) -> float | None:
    """Safely parse an optional finite float from heterogeneous DataFrame/JSON values."""
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any) -> str:
    """Return normalized stripped text, converting missing markers to an empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split())


def _normalize_key(value: Any) -> str:
    """Normalize vendor/category text for case-insensitive grouping."""
    return _clean_text(value).casefold()


def _sanitize_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize a DataFrame into JSON-safe Python dicts with no NaN or Infinity."""
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="s"))


def _robust_scales(matrix: np.ndarray) -> np.ndarray:
    """Compute per-feature IQR scales with standard deviation and unit fallbacks."""
    iqr = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    std = np.std(matrix, axis=0)
    return np.where(iqr > 1e-6, iqr, np.where(std > 1e-6, std, 1.0))


def _tree_node_imbalance(tree: Any) -> np.ndarray:
    """Compute Carletti et al. (2023) Induced Imbalance Coefficients lambda(k) in [0.5, 1.0]."""
    lambdas = np.full(tree.node_count, 0.5, dtype=float)
    left = tree.children_left
    right = tree.children_right
    internal_idx = np.flatnonzero(left != right)
    if len(internal_idx) == 0:
        return lambdas

    counts = tree.n_node_samples.astype(float)
    nk = np.maximum(counts[internal_idx], 1.0)
    nl = counts[left[internal_idx]]
    nr = counts[right[internal_idx]]

    raw_imbalance = np.abs(nl - nr) / nk
    min_imbalance = np.where((nk.astype(int) % 2) == 1, 1.0 / nk, 0.0)
    max_imbalance = np.maximum(nk - 2.0, 0.0) / nk
    span = max_imbalance - min_imbalance
    scaled = np.divide(raw_imbalance - min_imbalance, span, out=np.zeros_like(span), where=span > 0)
    lambdas[internal_idx] = np.clip(0.5 + 0.5 * scaled, 0.5, 1.0)
    return lambdas


def compute_imbalance_diffi(
    forest: IsolationForest,
    x: np.ndarray,
    outlier_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Global-DIFFI and Local Imbalance-Weighted DIFFI (Carletti et al., 2023).

    @param forest: Fitted scikit-learn IsolationForest estimator.
    @param x: 2D array of shape (n_samples, n_features) to evaluate.
    @param outlier_mask: Optional boolean mask identifying predicted outliers.
    @returns: Tuple of (global_diffi [n_features], local_imbalance_diffi [n_samples, n_features]).
    """
    raw = np.asarray(x, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1) if raw.size > 0 else raw.reshape(0, getattr(forest, "n_features_in_", 0))
    n_samples = len(raw)
    n_features = int(getattr(forest, "n_features_in_", raw.shape[1] if raw.ndim == 2 else 0))

    if n_samples == 0:
        return np.zeros(n_features, dtype=float), np.zeros((0, n_features), dtype=float)

    importance = np.zeros((n_samples, n_features), dtype=float)
    visits = np.zeros((n_samples, n_features), dtype=float)

    for estimator, feat_cols in zip(forest.estimators_, forest.estimators_features_):
        cols = np.asarray(feat_cols, dtype=int)
        tree = estimator.tree_
        lambdas = _tree_node_imbalance(tree)
        route = estimator.decision_path(raw[:, cols]).tocoo()
        is_internal = tree.feature[route.col] >= 0
        visited_rows = route.row[is_internal]
        visited_nodes = route.col[is_internal]
        if len(visited_rows) == 0:
            continue
        visited_features = cols[tree.feature[visited_nodes]]
        depths = np.bincount(visited_rows, minlength=n_samples)
        weights = lambdas[visited_nodes] / np.maximum(depths[visited_rows], 1)
        np.add.at(importance, (visited_rows, visited_features), weights)
        np.add.at(visits, (visited_rows, visited_features), 1.0)

    local_imb = np.divide(importance, visits, out=np.zeros_like(importance), where=visits > 0)

    if outlier_mask is None:
        scores = -forest.score_samples(raw)
        threshold = float(np.quantile(scores, DEFAULT_OUTLIER_QUANTILE))
        mask = scores > threshold
    else:
        mask = np.asarray(outlier_mask, dtype=bool).ravel()
        if len(mask) != n_samples:
            raise ValueError("outlier_mask length must match the number of rows in x.")

    inlier_mask = ~mask
    i_out = importance[mask].sum(axis=0) if np.any(mask) else np.zeros(n_features, dtype=float)
    v_out = visits[mask].sum(axis=0) if np.any(mask) else np.zeros(n_features, dtype=float)
    i_in = importance[inlier_mask].sum(axis=0) if np.any(inlier_mask) else np.zeros(n_features, dtype=float)
    v_in = visits[inlier_mask].sum(axis=0) if np.any(inlier_mask) else np.zeros(n_features, dtype=float)

    cfi_out = np.divide(i_out, v_out, out=np.zeros(n_features, dtype=float), where=v_out > 0)
    cfi_in = np.divide(i_in, v_in, out=np.zeros(n_features, dtype=float), where=v_in > 0)
    global_diffi = np.divide(cfi_out, cfi_in, out=np.zeros(n_features, dtype=float), where=cfi_in > 0)
    return global_diffi, local_imb


def build_shap_waterfall(
    shap_values: np.ndarray | list[float],
    base_path: float,
    sample_size: int,
    feature_names: list[str],
    cutoff: float,
    input_contexts: list[str] | None = None,
) -> dict[str, Any]:
    """Map exact TreeSHAP path-length attributions to Isolation Forest score steps.

    @param shap_values: 1D array/list of TreeSHAP path-length contributions per feature.
    @param base_path: Expected baseline isolation path length E[h(x)].
    @param sample_size: Subsample size used by the Isolation Forest (max_samples_).
    @param feature_names: Ordered human-readable feature labels.
    @param cutoff: Calibrated Isolation Forest anomaly score cutoff.
    @param input_contexts: Optional formatted input descriptions per feature index.
    @returns: Structured waterfall dictionary from baseline path/score to final path/score.
    """
    shap_arr = np.asarray(shap_values, dtype=float).ravel()
    raw_c = float(path_adjustment(max(int(sample_size), 2)))
    path_c = raw_c if raw_c > 0 else 1.0

    curr_path = float(base_path)
    base_score = float(2.0 ** (-curr_path / path_c))
    curr_score = base_score
    order = np.argsort(shap_arr, kind="mergesort")

    steps: list[dict[str, Any]] = []
    for raw_idx in order:
        idx = int(raw_idx)
        val = float(shap_arr[idx])
        p_before = curr_path
        p_after = p_before + val
        s_before = curr_score
        s_after = float(2.0 ** (-p_after / path_c))
        steps.append({
            "feature_index": idx,
            "feature": str(feature_names[idx]) if idx < len(feature_names) else f"Feature {idx}",
            "context": str(input_contexts[idx]) if input_contexts and idx < len(input_contexts) else "",
            "shap_splits": round(val, 6),
            "path_before": round(p_before, 6),
            "path_after": round(p_after, 6),
            "score_before": round(s_before, 6),
            "score_after": round(s_after, 6),
            "score_delta": round(s_after - s_before, 6),
        })
        curr_path = p_after
        curr_score = s_after

    cutoff_val = float(cutoff)
    return {
        "base_path": round(float(base_path), 6),
        "final_path": round(curr_path, 6),
        "base_score": round(base_score, 6),
        "final_score": round(curr_score, 6),
        "cutoff": round(cutoff_val, 6),
        "above_cutoff": bool(curr_score > cutoff_val),
        "path_c": round(path_c, 6),
        "steps": steps,
    }


def _resolve_majority_driver(
    ts_idx: int,
    ld_idx: int,
    ex_idx: int,
    prefer_exiffi: bool,
) -> tuple[int, int]:
    """Determine the consensus top feature index and method agreement count (1..3)."""
    if ts_idx == ld_idx == ex_idx:
        return ts_idx, 3
    if ts_idx == ld_idx or ts_idx == ex_idx:
        return ts_idx, 2
    if ld_idx == ex_idx:
        return ld_idx, 2
    return (ex_idx if prefer_exiffi else ts_idx), 1


def compute_xai_consensus(
    tree_shap: np.ndarray | list[float],
    local_diffi: np.ndarray | list[float],
    exiffi: np.ndarray | list[float],
    feature_names: list[str],
    if_flag: bool,
    eif_flag: bool,
) -> dict[str, Any]:
    """Evaluate primary risk-driver agreement across TreeSHAP, Local-DIFFI, and ExIFFI.

    @param tree_shap: 1D TreeSHAP split attributions (negative = shorter path = higher risk).
    @param local_diffi: 1D Local-DIFFI feature importance values.
    @param exiffi: 1D ExIFFI oblique feature importance values.
    @param feature_names: Human-readable feature names aligned with attribution vectors.
    @param if_flag: Whether axis-aligned Isolation Forest flagged the record.
    @param eif_flag: Whether oblique Extended Isolation Forest flagged the record.
    @returns: Consensus tier, label, agreement count, per-method tops, and audit summary.
    """
    ts_idx = int(np.argmin(np.asarray(tree_shap, dtype=float)))
    ld_idx = int(np.argmax(np.asarray(local_diffi, dtype=float)))
    ex_idx = int(np.argmax(np.asarray(exiffi, dtype=float)))

    is_if = bool(if_flag)
    is_eif = bool(eif_flag)
    top_idx, agreement_count = _resolve_majority_driver(
        ts_idx, ld_idx, ex_idx, prefer_exiffi=(is_eif and not is_if)
    )
    top_feature = str(feature_names[top_idx])
    method_tops = {
        "tree_shap": str(feature_names[ts_idx]),
        "local_diffi": str(feature_names[ld_idx]),
        "exiffi": str(feature_names[ex_idx]),
    }

    if not is_if and not is_eif:
        tier = "normal"
        label = "Within normal peer range"
        summary = (
            f"Both Isolation Forest and Extended Isolation Forest remain at or below cutoff; "
            f"{top_feature} shows the largest relative variation ({agreement_count}/3 XAI methods)."
        )
    elif is_eif and not is_if:
        tier = "oblique_only"
        label = "EIF oblique interaction flag"
        summary = (
            f"Extended Isolation Forest flagged an oblique multivariate combination led by "
            f"{method_tops['exiffi']} while axis-aligned Isolation Forest stayed within cutoff."
        )
    elif agreement_count == 3:
        tier = "high_consensus"
        label = "3/3 XAI methods agree"
        summary = (
            f"All three XAI paradigms (TreeSHAP, Local-DIFFI, and ExIFFI) converge on "
            f"{top_feature} as the primary isolation driver."
        )
    elif agreement_count == 2:
        tier = "moderate_consensus"
        label = "2/3 XAI methods agree"
        summary = (
            f"Two of three XAI methods identify {top_feature} as the primary risk driver; "
            f"verify alongside secondary split contributors."
        )
    else:
        tier = "oblique_interaction"
        label = "Multi-feature / oblique interaction"
        summary = (
            f"TreeSHAP ({method_tops['tree_shap']}), Local-DIFFI ({method_tops['local_diffi']}), "
            f"and ExIFFI ({method_tops['exiffi']}) highlight distinct features, indicating a compound interaction."
        )

    return {
        "top_feature": top_feature,
        "top_index": int(top_idx),
        "agreement_count": int(agreement_count),
        "tier": tier,
        "label": label,
        "both_forests_flag": bool(is_if and is_eif),
        "method_tops": method_tops,
        "summary": summary,
    }


def _check_svp_ceiling(row: dict[str, Any]) -> dict[str, str] | None:
    """Check RA 12009 (PHP 2M) and RA 9184 (PHP 1M) Small Value Procurement ceiling proximity."""
    mode_text = " ".join([
        _clean_text(row.get("procurement_mode")),
        _clean_text(row.get("official_procurement_mode")),
        _clean_text(row.get("mode_group")),
    ]).casefold()
    if "small value" not in mode_text:
        return None

    candidates = [
        _opt_float(row.get("approved_budget_php")),
        _opt_float(row.get("contract_amount")),
        _opt_float(row.get("award_export_abc_php")),
    ]
    for amount in (val for val in candidates if val is not None and val > 0):
        if RA12009_SVP_MIN_PHP <= amount <= RA12009_SVP_MAX_PHP:
            return {
                "code": "ra12009_svp_ceiling",
                "title": "SVP Ceiling Proximity (RA 12009 ₱2M Threshold)",
                "severity": "review",
                "detail": (
                    f"Small Value Procurement amount/ABC (₱{amount:,.2f}) sits within 80%–100% of "
                    f"the RA 12009 ₱2,000,000 SVP ceiling for 1st-class cities."
                ),
                "document_to_check": (
                    "Approved Annual Procurement Plan (APP), Purchase Request (PR), and BAC Resolution "
                    "justifying Small Value Procurement instead of Competitive Bidding."
                ),
            }
        if RA9184_SVP_MIN_PHP <= amount <= RA9184_SVP_MAX_PHP:
            return {
                "code": "ra12009_svp_ceiling",
                "title": "SVP Ceiling Proximity (RA 9184 ₱1M Threshold)",
                "severity": "review",
                "detail": (
                    f"Small Value Procurement amount/ABC (₱{amount:,.2f}) clusters just below "
                    f"the ₱1,000,000 RA 9184 Annex H SVP threshold."
                ),
                "document_to_check": (
                    "Purchase Request (PR) lot consolidation log, APP schedule, and BAC Resolution "
                    "confirming no splitting of requisitions."
                ),
            }
    return None


def _check_solicitation_window(row: dict[str, Any]) -> dict[str, str] | None:
    """Check minimum PhilGEPS posting window under RA 9184 Sec. 21 / RA 12009 IRR."""
    window = _opt_float(row.get("notice_window_days"))
    if window is None:
        return None
    mode_text = (
        _clean_text(row.get("procurement_mode")) or _clean_text(row.get("official_procurement_mode"))
    )
    is_public_bidding = "public bidding" in mode_text.casefold()
    if is_public_bidding and window < PUBLIC_BID_MIN_WINDOW_DAYS:
        return {
            "code": "compressed_bid_window",
            "title": "Compressed Public Bidding Window (<7 Days)",
            "severity": "review",
            "detail": (
                f"Solicitation window is {int(round(window))} calendar day(s), below the standard "
                f"7-calendar-day PhilGEPS posting minimum for Public Bidding."
            ),
            "document_to_check": (
                "PhilGEPS Bid Notice Abstract, Invitation to Bid posting certification, and BAC minutes."
            ),
        }
    if not is_public_bidding and window < ALT_MODE_MIN_WINDOW_DAYS:
        return {
            "code": "compressed_bid_window",
            "title": "Compressed Alternative Mode Posting (<3 Days)",
            "severity": "review",
            "detail": (
                f"Solicitation window is {int(round(window))} calendar day(s), shorter than the "
                f"standard 3-calendar-day RFQ posting window."
            ),
            "document_to_check": (
                "Request for Quotation (RFQ) transmittal log, PhilGEPS posting proof, or emergency justification."
            ),
        }
    return None


def _check_tight_abc_margin(kind: str, row: dict[str, Any]) -> dict[str, str] | None:
    """Detect contract awards or infrastructure costs clustering within 99.5%–100.05% of ABC."""
    ratio: float | None = None
    if kind == "dpwh":
        cost = _opt_float(row.get("contract_cost_php"))
        abc = _opt_float(row.get("abc_php"))
        raw_ratio = _opt_float(row.get("cost_abc_ratio"))
        if cost is not None and abc is not None and cost > 0 and abc > 0:
            ratio = cost / abc
        elif raw_ratio is not None and raw_ratio > 0 and row.get("contract_cost_php") is not None:
            ratio = raw_ratio
    elif kind == "tender":
        obs_total = _opt_float(row.get("observed_official_award_total_php"))
        abc = _opt_float(row.get("approved_budget_php"))
        if obs_total is not None and abc is not None and obs_total > 0 and abc > 0:
            ratio = obs_total / abc
    else:
        amount = _opt_float(row.get("contract_amount"))
        abc = _opt_float(row.get("approved_budget_php")) or _opt_float(row.get("award_export_abc_php"))
        if amount is not None and abc is not None and amount > 0 and abc > 0:
            ratio = amount / abc

    if ratio is not None and TIGHT_ABC_MIN_RATIO <= ratio <= TIGHT_ABC_MAX_RATIO:
        return {
            "code": "tight_abc_margin",
            "title": "Tight Award-to-ABC Margin (99.5%–100.0%)",
            "severity": "info",
            "detail": (
                f"Awarded cost is {ratio * 100:.2f}% of the Approved Budget for the Contract (ABC), "
                f"leaving minimal competitive price discount."
            ),
            "document_to_check": (
                "Abstract of Bids as Calculated, Detailed Unit Price Analysis (DUPA), and number of competing bidders."
            ),
        }
    return None


def statutory_procurement_checks(kind: str, row: dict[str, Any]) -> list[dict[str, str]]:
    """Run deterministic statutory audit checks grounded in RA 9184 and RA 12009 (NGPA).

    @param kind: Screen domain ('award', 'tender', or 'dpwh').
    @param row: Record dictionary containing domain fields.
    @returns: List of triggered check dictionaries with code, title, severity, detail, document_to_check.
    """
    checks: list[dict[str, str]] = []

    svp_check = _check_svp_ceiling(row)
    if svp_check is not None:
        checks.append(svp_check)

    window_check = _check_solicitation_window(row)
    if window_check is not None:
        checks.append(window_check)

    margin_check = _check_tight_abc_margin(kind, row)
    if margin_check is not None:
        checks.append(margin_check)

    if kind == "award":
        burst = _opt_float(row.get("vendor_category_30d"))
        if burst is not None and burst >= VENDOR_BURST_30D_MIN:
            checks.append({
                "code": "vendor_30d_burst",
                "title": "30-Day Same-Category Vendor Burst (Split-Contract Check)",
                "severity": "review",
                "detail": (
                    f"Awardee received {int(round(burst))} prior award(s) in the same business category "
                    f"within the preceding 30 calendar days (RA 9184 Sec. 54.1 / RA 12009 anti-splitting rule)."
                ),
                "document_to_check": (
                    "Purchase Requests (PRs) across the 30-day window, APP itemization, and Abstract of Quotations."
                ),
            })

        share = _opt_float(row.get("vendor_share_pct"))
        if share is not None and share >= VENDOR_CONCENTRATION_MIN_PCT:
            checks.append({
                "code": "vendor_concentration",
                "title": "High Prior Vendor Award Concentration",
                "severity": "info",
                "detail": (
                    f"Awardee accounted for {share:.1f}% of citywide awards over the prior 365 days."
                ),
                "document_to_check": (
                    "PhilGEPS Document Request List (DRL), invited supplier registry, and post-qualification report."
                ),
            })

    q_days = _opt_float(row.get("days_to_quarter_end"))
    if q_days is not None and 0 <= q_days <= QUARTER_END_RUSH_MAX_DAYS:
        checks.append({
            "code": "quarter_end_rush",
            "title": "Quarter-End Fiscal Window (≤5 Days)",
            "severity": "info",
            "detail": (
                f"Recorded {int(round(q_days))} calendar day(s) before quarter close; verify delivery "
                f"readiness and allotment timing."
            ),
            "document_to_check": (
                "Obligation Request and Status (ORS), Notice to Proceed (NTP), and Inspection & Acceptance Report (IAR)."
            ),
        })

    if kind == "dpwh":
        variance = _opt_float(row.get("variance_days"))
        if variance is not None and variance >= DPWH_DELAY_MIN_DAYS:
            checks.append({
                "code": "dpwh_schedule_delay",
                "title": "Significant Infrastructure Schedule Overrun (≥30 Days)",
                "severity": "review",
                "detail": (
                    f"Actual completion exceeded target completion by {int(round(variance))} calendar days."
                ),
                "document_to_check": (
                    "DPWH Statement of Work Accomplished (SWA), Time Extension / Suspension Orders, "
                    "and Liquidated Damages computation."
                ),
            })

    return checks


def _mutable_spec(kind: str, row: dict[str, Any], x_vec: np.ndarray) -> list[dict[str, Any]]:
    """Build mutable feature descriptors and bidirectional domain converters for recourse search."""
    if kind == "award":
        peer_med = _opt_float(row.get("peer_median_php")) or 100_000.0
        peer_med = max(peer_med, 1.0)
        return [
            {
                "idx": 0,
                "key": "contract_amount",
                "feature": FEATURES["amount_log_ratio"],
                "to_domain": lambda v, pm=peer_med: round(max(1.0, float(pm * (10.0 ** v))), 2),
                "to_feat": lambda dom, pm=peer_med: float(np.log10(max(dom, 1.0) / pm)),
                "fmt": lambda dom: f"₱{dom:,.2f}",
                "orig_dom": _opt_float(row.get("contract_amount")) or round(peer_med * (10.0 ** float(x_vec[0])), 2),
            },
            {
                "idx": 1,
                "key": "vendor_share_pct",
                "feature": FEATURES["vendor_share_pct"],
                "to_domain": lambda v: round(max(0.0, float(v)), 4),
                "to_feat": lambda dom: float(max(0.0, dom)),
                "fmt": lambda dom: f"{dom:.2f}%",
                "orig_dom": round(max(0.0, float(x_vec[1])), 4),
            },
            {
                "idx": 2,
                "key": "vendor_category_30d",
                "feature": FEATURES["vendor_category_30d"],
                "to_domain": lambda v: float(max(0, int(round(v)))),
                "to_feat": lambda dom: float(max(0, int(round(dom)))),
                "fmt": lambda dom: f"{int(round(dom))} prior awards",
                "orig_dom": float(max(0, int(round(x_vec[2])))),
            },
        ]
    if kind == "tender":
        bud_med = max(_opt_float(row.get("budget_peer_median_php")) or 250_000.0, 1.0)
        win_med = max(_opt_float(row.get("window_peer_median_days")) or 7.0, 1.0)
        return [
            {
                "idx": 0,
                "key": "approved_budget_php",
                "feature": TENDER_FEATURES["budget_log_ratio"],
                "to_domain": lambda v, bm=bud_med: round(max(1.0, float(bm * (10.0 ** v))), 2),
                "to_feat": lambda dom, bm=bud_med: float(np.log10(max(dom, 1.0) / bm)),
                "fmt": lambda dom: f"₱{dom:,.2f}",
                "orig_dom": _opt_float(row.get("approved_budget_php")) or round(bud_med * (10.0 ** float(x_vec[0])), 2),
            },
            {
                "idx": 1,
                "key": "notice_window_days",
                "feature": TENDER_FEATURES["window_log_ratio"],
                "to_domain": lambda v, wm=win_med: float(max(1, int(round(wm * (10.0 ** v))))),
                "to_feat": lambda dom, wm=win_med: float(np.log10(max(1.0, round(dom)) / wm)),
                "fmt": lambda dom: f"{int(round(dom))} days",
                "orig_dom": _opt_float(row.get("notice_window_days")) or float(max(1, int(round(win_med * (10.0 ** float(x_vec[1])))))),
            },
            {
                "idx": 2,
                "key": "prior_mode_share",
                "feature": TENDER_FEATURES["prior_mode_share"],
                "to_domain": lambda v: round(float(np.clip(v, 0.0, 100.0)), 4),
                "to_feat": lambda dom: float(np.clip(dom, 0.0, 100.0)),
                "fmt": lambda dom: f"{dom:.1f}%",
                "orig_dom": round(float(x_vec[2]), 4),
            },
        ]
    if kind == "dpwh":
        return [
            {
                "idx": 0,
                "key": "cost_abc_ratio",
                "feature": DPWH_FEATURES["cost_abc_ratio_feat"],
                "to_domain": lambda v: round(max(0.05, float(v)), 4),
                "to_feat": lambda dom: float(max(0.05, dom)),
                "fmt": lambda dom: f"{dom:.4f} ({dom * 100:.2f}% ABC)",
                "orig_dom": round(float(x_vec[0]), 4),
            },
            {
                "idx": 2,
                "key": "variance_days",
                "feature": DPWH_FEATURES["variance_days_feat"],
                "to_domain": lambda v: float(int(round(v))),
                "to_feat": lambda dom: float(int(round(dom))),
                "fmt": lambda dom: f"{int(round(dom)):+d} days",
                "orig_dom": float(int(round(x_vec[2]))),
            },
            {
                "idx": 3,
                "key": "planned_duration_days",
                "feature": DPWH_FEATURES["planned_duration_days"],
                "to_domain": lambda v: float(max(1, int(round(v)))),
                "to_feat": lambda dom: float(max(1, int(round(dom)))),
                "fmt": lambda dom: f"{int(round(dom))} days",
                "orig_dom": float(max(1, int(round(x_vec[3])))),
            },
        ]
    raise ValueError(f"Unsupported counterfactual kind: {kind!r}")


def _snapped_quantile_grid(train_col: np.ndarray, spec: dict[str, Any], n_points: int) -> np.ndarray:
    """Generate round-trip domain-snapped empirical quantile candidates for one mutable feature."""
    raw_q = np.quantile(train_col, np.linspace(0.05, 0.95, n_points))
    extras = np.array([np.median(train_col)])
    merged = np.concatenate([raw_q, extras])
    snapped = [spec["to_feat"](spec["to_domain"](float(val))) for val in merged]
    return np.unique(np.asarray(snapped, dtype=float))


def _format_recourse_changes(
    specs: list[dict[str, Any]],
    x_orig: np.ndarray,
    x_best: np.ndarray,
) -> list[dict[str, Any]]:
    """Build structured domain-unit change entries for altered mutable features."""
    changes: list[dict[str, Any]] = []
    for spec in specs:
        idx = spec["idx"]
        if np.isclose(x_best[idx], x_orig[idx], atol=1e-7):
            continue
        from_dom = float(spec["orig_dom"])
        to_dom = float(spec["to_domain"](float(x_best[idx])))
        if np.isclose(from_dom, to_dom, atol=1e-6):
            continue
        changes.append({
            "feature": str(spec["feature"]),
            "key": str(spec["key"]),
            "from_display": str(spec["fmt"](from_dom)),
            "to_display": str(spec["fmt"](to_dom)),
            "from_val": round(from_dom, 4),
            "to_val": round(to_dom, 4),
        })
    return changes


def compute_counterfactual_recourse(
    kind: str,
    row: dict[str, Any],
    x_vec: np.ndarray,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Find minimal sparse mutable procurement adjustments that bring IF score <= cutoff.

    @param kind: Screen domain ('award', 'tender', or 'dpwh').
    @param row: Record metadata dictionary used for domain unit conversions.
    @param x_vec: 1D feature vector of shape (5,).
    @param result: Fitted screen dictionary containing 'forest', 'cutoffs', and 'views'.
    @returns: Counterfactual recourse dictionary with validity, sparsity, changes, and narrative.
    """
    x_orig = np.asarray(x_vec, dtype=float).ravel().copy()
    forest: IsolationForest = result["forest"]
    cutoff = float(result["cutoffs"]["Isolation Forest"])
    orig_score = float(-forest.score_samples(x_orig[None, :])[0])

    if orig_score <= cutoff:
        return {
            "valid": True,
            "sparsity": 0,
            "original_score": round(orig_score, 6),
            "recourse_score": round(orig_score, 6),
            "cutoff": round(cutoff, 6),
            "changes": [],
            "narrative": "Record is already at or below the calibrated review cutoff; no adjustment is required.",
        }

    train = result["views"]["Training"]
    scales = _robust_scales(train)
    specs = _mutable_spec(kind, row, x_orig)

    # Step A: 1-feature quantile search (25 quantiles per mutable lever)
    candidates_1d: list[np.ndarray] = []
    for spec in specs:
        idx = spec["idx"]
        for val in _snapped_quantile_grid(train[:, idx], spec, 25):
            if not np.isclose(val, x_orig[idx], atol=1e-7):
                trial = x_orig.copy()
                trial[idx] = val
                candidates_1d.append(trial)

    best_vec: np.ndarray | None = None
    best_score = orig_score
    is_valid = False

    if candidates_1d:
        mat_1d = np.vstack(candidates_1d)
        scores_1d = -forest.score_samples(mat_1d)
        passing = np.flatnonzero(scores_1d <= cutoff)
        if len(passing) > 0:
            dists = np.sum(np.abs(mat_1d[passing] - x_orig[None, :]) / scales[None, :], axis=1)
            win_pos = int(passing[int(np.argmin(dists))])
            best_vec = mat_1d[win_pos]
            best_score = float(scores_1d[win_pos])
            is_valid = True
        else:
            min_pos = int(np.argmin(scores_1d))
            best_vec = mat_1d[min_pos]
            best_score = float(scores_1d[min_pos])

    # Step B: 2-feature Cartesian quantile search (13 x 13 per pair) + Step C: 3-feature (7 x 7 x 7)
    if not is_valid:
        grids_2d = [_snapped_quantile_grid(train[:, s["idx"]], s, 13) for s in specs]
        candidates_multi: list[np.ndarray] = []
        for a in range(len(specs)):
            for b in range(a + 1, len(specs)):
                ia, ib = specs[a]["idx"], specs[b]["idx"]
                ga, gb = np.meshgrid(grids_2d[a], grids_2d[b], indexing="xy")
                block = np.repeat(x_orig[None, :], ga.size, axis=0)
                block[:, ia] = ga.ravel()
                block[:, ib] = gb.ravel()
                candidates_multi.append(block)

        mat_2d = np.vstack(candidates_multi)
        scores_2d = -forest.score_samples(mat_2d)
        passing_2d = np.flatnonzero(scores_2d <= cutoff)
        if len(passing_2d) > 0:
            sub = mat_2d[passing_2d]
            sparsity_arr = np.sum(~np.isclose(sub, x_orig[None, :], atol=1e-7), axis=1)
            dists = np.sum(np.abs(sub - x_orig[None, :]) / scales[None, :], axis=1)
            order = np.lexsort((dists, sparsity_arr))
            chosen_idx = int(passing_2d[int(order[0])])
            best_vec = mat_2d[chosen_idx]
            best_score = float(scores_2d[chosen_idx])
            is_valid = True
        else:
            min_2d = int(np.argmin(scores_2d))
            if float(scores_2d[min_2d]) < best_score:
                best_vec = mat_2d[min_2d]
                best_score = float(scores_2d[min_2d])

            grids_3d = [_snapped_quantile_grid(train[:, s["idx"]], s, 7) for s in specs]
            g0, g1, g2 = np.meshgrid(grids_3d[0], grids_3d[1], grids_3d[2], indexing="ij")
            mat_3d = np.repeat(x_orig[None, :], g0.size, axis=0)
            mat_3d[:, specs[0]["idx"]] = g0.ravel()
            mat_3d[:, specs[1]["idx"]] = g1.ravel()
            mat_3d[:, specs[2]["idx"]] = g2.ravel()
            scores_3d = -forest.score_samples(mat_3d)
            passing_3d = np.flatnonzero(scores_3d <= cutoff)
            if len(passing_3d) > 0:
                sub3 = mat_3d[passing_3d]
                sp3 = np.sum(~np.isclose(sub3, x_orig[None, :], atol=1e-7), axis=1)
                d3 = np.sum(np.abs(sub3 - x_orig[None, :]) / scales[None, :], axis=1)
                order3 = np.lexsort((d3, sp3))
                c3 = int(passing_3d[int(order3[0])])
                best_vec = mat_3d[c3]
                best_score = float(scores_3d[c3])
                is_valid = True
            else:
                min_3d = int(np.argmin(scores_3d))
                if float(scores_3d[min_3d]) < best_score:
                    best_vec = mat_3d[min_3d]
                    best_score = float(scores_3d[min_3d])

    final_vec = best_vec if best_vec is not None else x_orig
    changes = _format_recourse_changes(specs, x_orig, final_vec)
    sparsity = len(changes)
    change_phrases = "; ".join(f"{c['feature']} from {c['from_display']} to {c['to_display']}" for c in changes)

    if is_valid and changes:
        narrative = (
            f"Adjusting {sparsity} mutable lever(s) ({change_phrases}) lowers the Isolation Forest "
            f"score from {orig_score:.4f} to {best_score:.4f} (at or below the {cutoff:.4f} cutoff)."
        )
    else:
        narrative = (
            f"Locked contextual features keep this record above the {cutoff:.4f} cutoff even after "
            f"optimal mutable adjustments (score {orig_score:.4f} -> {best_score:.4f})."
        )

    return {
        "valid": bool(is_valid),
        "sparsity": int(sparsity),
        "original_score": round(orig_score, 6),
        "recourse_score": round(best_score, 6),
        "cutoff": round(cutoff, 6),
        "changes": changes,
        "narrative": narrative,
    }


def compute_2d_decision_surface(
    result: dict[str, Any],
    fx: int,
    fy: int,
    grid_size: int = 24,
) -> dict[str, Any]:
    """Compute orthogonal IF vs oblique EIF + ExIFFI 2D anomaly contours on (fx, fy).

    @param result: Fitted screen dictionary with views, forest, extended, scaler, cutoffs, features.
    @param fx: Horizontal feature index (0..n_features-1).
    @param fy: Vertical feature index (0..n_features-1), distinct from fx.
    @param grid_size: Mesh resolution per axis (default 24).
    @returns: Compact 4-decimal contour dictionary for interactive canvas rendering.
    """
    feature_names = list(result["features"].values())
    n_features = len(feature_names)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if not (0 <= fx < n_features and 0 <= fy < n_features) or fx == fy:
        raise ValueError(f"fx ({fx}) and fy ({fy}) must be distinct valid feature indices in 0..{n_features - 1}.")

    combined = np.vstack([result["views"]["Training"], result["views"]["Evaluation"]])
    medians = np.median(result["views"]["Training"], axis=0)

    def _axis_grid(col_idx: int) -> np.ndarray:
        low, high = np.percentile(combined[:, col_idx], [2.0, 98.0])
        span = float(high - low)
        if span <= 1e-9:
            return np.linspace(float(low) - 1.0, float(high) + 1.0, grid_size)
        pad = 0.05 * span
        return np.linspace(float(low) - pad, float(high) + pad, grid_size)

    x_grid = _axis_grid(fx)
    y_grid = _axis_grid(fy)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")

    mesh = np.repeat(medians[None, :], grid_size * grid_size, axis=0)
    mesh[:, fx] = xx.ravel()
    mesh[:, fy] = yy.ravel()

    if_flat = -result["forest"].score_samples(mesh)
    eif_flat, exiffi_mesh = result["extended"].score(result["scaler"].transform(mesh), explain=True)

    totals = exiffi_mesh.sum(axis=1)
    pair_sums = exiffi_mesh[:, fx] + exiffi_mesh[:, fy]
    shares = np.divide(pair_sums, totals, out=np.zeros_like(totals), where=totals > 0)
    pair_share = float(np.mean(shares))

    return {
        "fx": int(fx),
        "fy": int(fy),
        "fx_name": str(feature_names[fx]),
        "fy_name": str(feature_names[fy]),
        "x_grid": [round(float(v), 4) for v in x_grid],
        "y_grid": [round(float(v), 4) for v in y_grid],
        "if_scores": [[round(float(v), 4) for v in row] for row in if_flat.reshape(grid_size, grid_size)],
        "eif_scores": [[round(float(v), 4) for v in row] for row in eif_flat.reshape(grid_size, grid_size)],
        "if_cutoff": round(float(result["cutoffs"]["Isolation Forest"]), 6),
        "eif_cutoff": round(float(result["cutoffs"]["Extended Isolation Forest"]), 6),
        "exiffi_pair_share": round(pair_share, 4),
    }


def find_nearest_peers(
    rows_df: pd.DataFrame,
    x_matrix: np.ndarray,
    target_idx: int | None,
    group_col: str,
    group_val: str,
    id_col: str,
    exclude_id: str | None = None,
    k: int = 4,
    target_vec: np.ndarray | None = None,
) -> list[str]:
    """Return up to k nearest peer IDs using IQR-scaled distance, preferring same-group peers.

    @param rows_df: Candidate DataFrame aligned row-for-row with x_matrix.
    @param x_matrix: 2D feature matrix of shape (n_rows, n_features).
    @param target_idx: Optional index in x_matrix when querying an existing row.
    @param group_col: Column name for peer cohort matching (e.g. category, mode_group, type_of_work).
    @param group_val: Target group value to prioritize.
    @param id_col: Unique identifier column name.
    @param exclude_id: Optional record ID to exclude from returned peers.
    @param k: Maximum number of peer IDs to return.
    @param target_vec: Explicit 1D feature vector when querying a custom/live record.
    @returns: Ordered list of up to k peer identifier strings.
    """
    if k <= 0 or len(rows_df) == 0 or len(x_matrix) == 0:
        return []
    if target_vec is not None:
        vec = np.asarray(target_vec, dtype=float).ravel()
    elif target_idx is not None:
        vec = np.asarray(x_matrix[int(target_idx)], dtype=float).ravel()
    else:
        raise ValueError("Either target_idx or target_vec must be provided.")

    scales = _robust_scales(x_matrix)
    dists = np.linalg.norm((x_matrix - vec[None, :]) / scales[None, :], axis=1)

    ids = rows_df[id_col].astype(str).to_numpy()
    skip_id = str(exclude_id) if exclude_id is not None else (
        str(ids[int(target_idx)]) if target_idx is not None else None
    )
    eligible = np.ones(len(rows_df), dtype=bool)
    if skip_id is not None:
        eligible &= ids != skip_id
    if target_idx is not None and 0 <= int(target_idx) < len(rows_df):
        eligible[int(target_idx)] = False

    chosen: list[str] = []
    if group_col in rows_df.columns and _clean_text(group_val):
        groups = rows_df[group_col].fillna("").astype(str).to_numpy()
        same_idx = np.flatnonzero(eligible & (groups == str(group_val)))
        same_sorted = same_idx[np.argsort(dists[same_idx], kind="mergesort")]
        for idx in same_sorted[:k]:
            chosen.append(str(ids[int(idx)]))

    if len(chosen) < k:
        seen = set(chosen)
        rem_idx = np.flatnonzero(eligible)
        rem_sorted = rem_idx[np.argsort(dists[rem_idx], kind="mergesort")]
        for idx in rem_sorted:
            rec_id = str(ids[int(idx)])
            if rec_id not in seen:
                chosen.append(rec_id)
                seen.add(rec_id)
                if len(chosen) >= k:
                    break

    return chosen


def compute_global_xai_summary(
    forest: IsolationForest,
    x_eval: np.ndarray,
    effects: np.ndarray,
    loc_diffi: np.ndarray,
    exiffi: np.ndarray,
    feature_names: list[str],
    cutoff: float,
) -> dict[str, Any]:
    """Aggregate global feature importance across Global-DIFFI, |TreeSHAP|, Local-DIFFI, and ExIFFI."""
    scores = -forest.score_samples(x_eval)
    glob_diffi, _ = compute_imbalance_diffi(forest, x_eval, outlier_mask=(scores > cutoff))
    return {
        "features": list(feature_names),
        "global_diffi": [round(float(v), 6) for v in glob_diffi],
        "mean_abs_shap": [round(float(v), 6) for v in np.mean(np.abs(effects), axis=0)],
        "mean_local_diffi": [round(float(v), 6) for v in np.mean(loc_diffi, axis=0)],
        "mean_exiffi": [round(float(v), 6) for v in np.mean(exiffi, axis=0)],
    }


def _engineer_dpwh_features(
    projects_df: pd.DataFrame,
    flood_links_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge flood-control links and engineer 5 finite infrastructure features for all DPWH rows."""
    frame = projects_df.copy()
    frame["dpwh_contract_id"] = frame["dpwh_contract_id"].astype(str)

    flood_cols = ["flood_map_project_id", "flood_map_component_id", "flood_map_municipality"]
    if flood_links_df is not None and "flood_map_project_id" not in frame.columns:
        avail = ["dpwh_contract_id"] + [c for c in flood_cols if c in flood_links_df.columns]
        sub_flood = flood_links_df[avail].copy()
        sub_flood["dpwh_contract_id"] = sub_flood["dpwh_contract_id"].astype(str)
        frame = frame.merge(sub_flood, on="dpwh_contract_id", how="left", validate="one_to_one")
    for col in flood_cols:
        if col not in frame.columns:
            frame[col] = None
        else:
            frame[col] = frame[col].replace("", None)

    for num_col in ("infra_year", "abc_php", "contract_cost_php", "cost_abc_ratio", "variance_days", "lat", "lon"):
        if num_col in frame.columns:
            frame[num_col] = pd.to_numeric(frame[num_col], errors="coerce")

    raw_ratio = frame["cost_abc_ratio"].copy()
    valid_calc = raw_ratio.isna() & frame["contract_cost_php"].gt(0) & frame["abc_php"].gt(0)
    raw_ratio.loc[valid_calc] = frame.loc[valid_calc, "contract_cost_php"] / frame.loc[valid_calc, "abc_php"]

    overall_ratio_med = float(raw_ratio.dropna().median()) if raw_ratio.notna().any() else DPWH_DEFAULT_RATIO_FALLBACK
    work_ratio_med = frame.assign(_r=raw_ratio).groupby("type_of_work")["_r"].transform("median")
    frame["cost_abc_ratio_feat"] = raw_ratio.fillna(work_ratio_med).fillna(overall_ratio_med)

    pos_abc = frame["abc_php"].where(frame["abc_php"] > 0)
    city_abc_med = float(pos_abc.dropna().median()) if pos_abc.notna().any() else 1_000_000.0
    work_abc_counts = pos_abc.groupby(frame["type_of_work"]).transform("count")
    work_abc_med = pos_abc.groupby(frame["type_of_work"]).transform("median")
    peer_abc = np.where(work_abc_counts >= DPWH_MIN_WORK_TYPE_PEERS, work_abc_med, city_abc_med)
    frame["abc_peer_median_php"] = pd.Series(peer_abc, index=frame.index).fillna(city_abc_med)
    frame["abc_log_ratio"] = np.where(
        frame["abc_php"] > 0,
        np.log10(np.maximum(frame["abc_php"], 1.0) / frame["abc_peer_median_php"]),
        0.0,
    )

    frame["variance_days_feat"] = frame["variance_days"].fillna(0.0)

    start_dt = pd.to_datetime(frame["start_date"], errors="coerce")
    target_dt = pd.to_datetime(frame["target_end_date"], errors="coerce")
    raw_dur = (target_dt - start_dt).dt.days.astype(float)
    raw_dur = raw_dur.where(raw_dur.isna(), np.maximum(raw_dur, 1.0))
    overall_dur_med = float(raw_dur.dropna().median()) if raw_dur.notna().any() else DPWH_DEFAULT_DURATION_DAYS
    work_dur_med = frame.assign(_d=raw_dur).groupby("type_of_work")["_d"].transform("median")
    frame["planned_duration_days"] = np.maximum(raw_dur.fillna(work_dur_med).fillna(overall_dur_med), 1.0)

    brgy_clean = frame["barangay"].fillna("").astype(str).str.strip()
    brgy_counts = brgy_clean[brgy_clean != ""].value_counts()
    frame["barangay_project_density"] = brgy_clean.map(brgy_counts).fillna(0.0).astype(float)
    return frame


def _build_barangay_summary(scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate DPWH project counts, XAI flags, flood links, delays, and costs per barangay."""
    mapped = scored_df.loc[scored_df["barangay"].fillna("").astype(str).str.strip() != ""].copy()
    mapped["barangay"] = mapped["barangay"].astype(str).str.strip()
    summaries: list[dict[str, Any]] = []
    for brgy, grp in mapped.groupby("barangay", sort=True):
        flood_cnt = int(grp["flood_map_project_id"].notna().sum())
        mean_var = float(grp["variance_days"].dropna().mean()) if grp["variance_days"].notna().any() else 0.0
        cost_vals = grp["contract_cost_php"].fillna(grp["abc_php"]).fillna(0.0)
        summaries.append({
            "barangay": str(brgy),
            "project_count": int(len(grp)),
            "flagged_count": int(grp["review"].sum()),
            "flood_linked_count": flood_cnt,
            "mean_variance_days": round(mean_var, 2),
            "total_cost_php": round(float(cost_vals.sum()), 2),
            "max_anomaly_score": round(float(grp["anomaly_score"].max()), 6),
        })
    return summaries


def run_dpwh_screen(
    projects_df: pd.DataFrame,
    flood_links_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit IF + EIF on historical Tagum DPWH infrastructure and score/explain all 336 projects.

    @param projects_df: Tagum DPWH contract DataFrame (336 rows).
    @param flood_links_df: Optional Sumbong sa Pangulo flood-control match DataFrame.
    @returns: Full DPWH screen dictionary with records, XAI vectors, surfaces, and barangay_summary.
    """
    frame = _engineer_dpwh_features(projects_df, flood_links_df)
    feat_cols = list(DPWH_FEATURES.keys())
    feat_names = list(DPWH_FEATURES.values())

    years = frame["infra_year"].fillna(2020).astype(int)
    if int((years == 2023).sum()) >= DPWH_MIN_CALIBRATION_ROWS:
        train_mask = years <= 2022
        cal_mask = years == 2023
    elif int(((years >= 2022) & (years <= 2023)).sum()) >= DPWH_MIN_CALIBRATION_ROWS and int((years <= 2021).sum()) >= 30:
        train_mask = years <= 2021
        cal_mask = (years >= 2022) & (years <= 2023)
    else:
        order_idx = np.argsort(years.to_numpy(dtype=int), kind="mergesort")
        n_total = len(frame)
        n_tr = max(2, int(0.6 * n_total))
        n_ca = max(2, int(0.2 * n_total))
        train_mask = pd.Series(False, index=frame.index)
        cal_mask = pd.Series(False, index=frame.index)
        train_mask.iloc[order_idx[:n_tr]] = True
        cal_mask.iloc[order_idx[n_tr:n_tr + n_ca]] = True

    frame["split"] = np.select([train_mask, cal_mask], ["Training", "Calibration"], default="Evaluation")
    x_all = frame[feat_cols].to_numpy(dtype=float)
    x_train = x_all[train_mask.to_numpy()]
    x_cal = x_all[cal_mask.to_numpy()]

    forest = IsolationForest(
        n_estimators=TREES,
        max_samples=min(256, len(x_train)),
        max_features=1.0,
        contamination="auto",
        random_state=SEED,
        n_jobs=1,
    ).fit(x_train)
    scaler = RobustScaler().fit(x_train)
    extended = ExtendedForest(scaler.transform(x_train), seed=SEED, trees=TREES)

    calibration_scores = {
        "Isolation Forest": -forest.score_samples(x_cal),
        "Extended Isolation Forest": extended.score(scaler.transform(x_cal))[0],
        "Cost/ABC ratio baseline": np.abs(x_cal[:, 0] - np.median(x_train[:, 0])),
    }
    cutoffs = {name: score_cutoff(vals) for name, vals in calibration_scores.items()}

    scores_if = -forest.score_samples(x_all)
    scores_eif, exiffi = extended.score(scaler.transform(x_all), explain=True)
    effects, tree_base, path_errors = explain_paths(forest, x_all)
    loc_diffi = compute_local_diffi(forest, x_all)
    glob_diffi, imb_diffi = compute_imbalance_diffi(
        forest, x_all, outlier_mask=(scores_if > cutoffs["Isolation Forest"])
    )

    scored = frame.copy().reset_index(drop=True)
    scored["anomaly_score"] = scores_if
    scored["extended_score"] = scores_eif
    scored["review"] = scores_if > cutoffs["Isolation Forest"]
    scored["extended_review"] = scores_eif > cutoffs["Extended Isolation Forest"]
    scored["tree_path_length"] = tree_base + effects.sum(axis=1)
    scored["top_tree_shap"] = np.array(feat_names)[effects.argmin(axis=1)]

    groups = {
        "Training": scored.loc[train_mask.to_numpy()].reset_index(drop=True),
        "Calibration": scored.loc[cal_mask.to_numpy()].reset_index(drop=True),
        "Evaluation": scored,
    }
    views = {"Training": x_train, "Calibration": x_cal, "Evaluation": x_all}

    core_result: dict[str, Any] = {
        "processed": scored,
        "groups": groups,
        "views": views,
        "forest": forest,
        "scaler": scaler,
        "extended": extended,
        "cutoffs": cutoffs,
        "calibration_scores": calibration_scores,
        "eval": scored,
        "effects": effects,
        "exiffi": exiffi,
        "local_diffi": loc_diffi,
        "imbalance_diffi": imb_diffi,
        "global_diffi": glob_diffi,
        "tree_base": tree_base,
        "path_errors": path_errors,
        "baseline_name": "Cost/ABC ratio baseline",
        "features": dict(DPWH_FEATURES),
    }

    raw_records = _sanitize_json_records(scored)
    sample_sz = int(forest.max_samples_)
    if_cutoff = float(cutoffs["Isolation Forest"])
    for i, rec in enumerate(raw_records):
        contexts = [
            f"{scored.iloc[i]['cost_abc_ratio_feat']:.4f}",
            f"₱{float(scored.iloc[i]['abc_php']):,.0f} vs ₱{float(scored.iloc[i]['abc_peer_median_php']):,.0f}",
            f"{scored.iloc[i]['variance_days_feat']:+.0f}d",
            f"{scored.iloc[i]['planned_duration_days']:.0f}d",
            f"{int(scored.iloc[i]['barangay_project_density'])} projects",
        ]
        rec["tree_shap"] = effects[i].round(7).tolist()
        rec["local_diffi"] = loc_diffi[i].round(7).tolist()
        rec["imbalance_diffi"] = imb_diffi[i].round(7).tolist()
        rec["exiffi"] = exiffi[i].round(7).tolist()
        rec["waterfall"] = build_shap_waterfall(
            effects[i], tree_base, sample_sz, feat_names, if_cutoff, input_contexts=contexts
        )
        rec["consensus"] = compute_xai_consensus(
            effects[i], loc_diffi[i], exiffi[i], feat_names, bool(rec["review"]), bool(rec["extended_review"])
        )
        checks_list = statutory_procurement_checks("dpwh", rec)
        rec["statutory_checks"] = checks_list
        rec["statutory_flags"] = checks_list
        rec["peer_ids"] = find_nearest_peers(
            scored, x_all, i, "type_of_work", str(rec.get("type_of_work") or ""),
            "dpwh_contract_id", exclude_id=str(rec["dpwh_contract_id"]), k=4,
        )
        rec_cf = (
            compute_counterfactual_recourse("dpwh", rec, x_all[i], core_result)
            if rec["review"]
            else None
        )
        rec["recourse"] = rec_cf
        rec["counterfactual"] = rec_cf

    raw_records.sort(key=lambda r: (-r["anomaly_score"], r["dpwh_contract_id"]))
    for rank, rec in enumerate(raw_records, start=1):
        rec["rank"] = rank

    core_result["records"] = raw_records
    core_result["global_xai"] = compute_global_xai_summary(
        forest, x_all, effects, loc_diffi, exiffi, feat_names, if_cutoff
    )
    core_result["surfaces"] = [
        compute_2d_decision_surface(core_result, 0, 2, grid_size=24),
        compute_2d_decision_surface(core_result, 1, 3, grid_size=24),
    ]
    core_result["barangay_summary"] = _build_barangay_summary(scored)
    return core_result


def _detect_split_clusters(
    reg: pd.DataFrame,
    eval_ids: set[str],
    flagged_ids: set[str],
    eval_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Detect non-overlapping 30-day same-vendor, same-category split-risk clusters in 2024–2025."""
    scores_lookup = eval_scores or {}
    recent = reg.loc[
        reg["award_date"].astype(str).str[:4].isin({"2024", "2025"}) & (reg["_vkey"] != "")
    ].copy()
    if recent.empty:
        return []

    pair_sizes = recent.groupby(["_vkey", "_ckey"], sort=False)["record_id"].transform("size")
    eligible = recent.loc[pair_sizes >= SPLIT_CLUSTER_MIN_AWARDS].sort_values(
        ["_vkey", "_ckey", "_day", "record_id"]
    ).reset_index(drop=True)
    if eligible.empty:
        return []

    clusters: list[dict[str, Any]] = []
    for (_, _), grp in eligible.groupby(["_vkey", "_ckey"], sort=False):
        days = grp["_day"].to_numpy(dtype="datetime64[D]")
        amts = grp["contract_amount"].to_numpy(dtype=float)
        rids = grp["record_id"].astype(str).to_numpy()
        date_strs = grp["award_date"].astype(str).str[:10].to_numpy()
        vendor_name = str(grp["awardee_name"].iloc[0])
        cat_name = str(grp["_cat"].iloc[0])

        candidates: list[tuple[tuple[int, int, int, int, float], int, int, dict[str, Any]]] = []
        right_bounds = np.searchsorted(days, days + np.timedelta64(SPLIT_CLUSTER_WINDOW_DAYS, "D"), side="right")
        for start_i, end_excl in enumerate(right_bounds):
            count = int(end_excl - start_i)
            if count < SPLIT_CLUSTER_MIN_AWARDS:
                continue
            total_amt = float(amts[start_i:end_excl].sum())
            if total_amt < SPLIT_CLUSTER_MIN_TOTAL_PHP:
                continue
            win_ids = list(rids[start_i:end_excl])
            in_eval = [rid for rid in win_ids if rid in eval_ids]
            other_ids = [rid for rid in win_ids if rid not in eval_ids]
            ordered_ids = in_eval + other_ids
            n_flagged = sum(1 for rid in win_ids if rid in flagged_ids)
            peak_score = max((scores_lookup.get(rid, 0.0) for rid in in_eval), default=0.0)
            near_svp = SPLIT_CLUSTER_SVP_LOW_PHP <= total_amt <= SPLIT_CLUSTER_SVP_HIGH_PHP
            priority = (int(len(in_eval) > 0), int(n_flagged > 0), int(near_svp), count, total_amt)
            span_days = int((days[end_excl - 1] - days[start_i]) / np.timedelta64(1, "D"))
            candidates.append((
                priority,
                start_i,
                int(end_excl - 1),
                {
                    "awardee_name": vendor_name,
                    "category": cat_name,
                    "start_date": str(date_strs[start_i]),
                    "end_date": str(date_strs[end_excl - 1]),
                    "span_days": span_days,
                    "award_count": count,
                    "total_amount_php": round(total_amt, 2),
                    "near_svp_ceiling": bool(near_svp),
                    "lead_record_id": ordered_ids[0],
                    "record_ids": ordered_ids,
                    "eval_record_ids": in_eval,
                    "flagged_count": int(n_flagged),
                    "max_anomaly_score": round(float(peak_score), 6),
                },
            ))

        candidates.sort(key=lambda item: item[0], reverse=True)
        used_indices: set[int] = set()
        for _, s_idx, e_idx, info in candidates:
            span_set = set(range(s_idx, e_idx + 1))
            if span_set & used_indices:
                continue
            used_indices.update(span_set)
            clusters.append(info)

    clusters.sort(
        key=lambda c: (
            len(c["eval_record_ids"]) > 0,
            c["flagged_count"],
            c["near_svp_ceiling"],
            c["award_count"],
            c["total_amount_php"],
        ),
        reverse=True,
    )
    return clusters[:MAX_SPLIT_CLUSTERS]


def _build_overlap_leads(
    integrated_data: IntegratedData,
    award_eval_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Deduplicate and rank unresolved archive-to-official overlap pairs, prioritizing 2025 queue."""
    unresolved = integrated_data.unresolved_links.copy()
    if unresolved.empty:
        return []

    reg_vendor = dict(
        zip(
            integrated_data.award_register["record_id"].astype(str),
            integrated_data.award_register["awardee_name"].astype(str),
        )
    )
    eval_scores = dict(
        zip(award_eval_df["record_id"].astype(str), award_eval_df["anomaly_score"].astype(float))
    )
    eval_flags = dict(
        zip(award_eval_df["record_id"].astype(str), award_eval_df["review"].astype(bool))
    )

    unresolved["contract_amount_php"] = pd.to_numeric(unresolved["contract_amount_php"], errors="coerce").fillna(0.0)
    cand_counts = unresolved.groupby("record_id")["official_award_reference"].size().to_dict()

    # Keep highest-priority candidate per archive record_id (preferring title_agrees=True)
    sorted_unres = unresolved.sort_values(
        ["record_id", "title_agrees", "official_award_reference"],
        ascending=[True, False, True],
    )
    dedup = sorted_unres.drop_duplicates(subset=["record_id"], keep="first")

    leads: list[dict[str, Any]] = []
    for row in dedup.itertuples(index=False):
        rid = str(row.record_id)
        in_2025 = rid in eval_scores
        score = round(float(eval_scores[rid]), 6) if in_2025 else None
        flagged = bool(eval_flags.get(rid, False))
        leads.append({
            "record_id": rid,
            "official_award_reference": str(row.official_award_reference),
            "award_date": str(row.award_date)[:10],
            "awardee_name": reg_vendor.get(rid, ""),
            "contract_amount_php": round(float(row.contract_amount_php), 2),
            "title_agrees": bool(row.title_agrees),
            "archive_notice_title": _clean_text(row.archive_notice_title),
            "official_notice_title": _clean_text(row.official_notice_title),
            "reason": _clean_text(row.reason),
            "candidate_count": int(cand_counts.get(rid, 1)),
            "in_2025_queue": in_2025,
            "review": flagged,
            "anomaly_score": score,
        })

    leads.sort(
        key=lambda item: (
            item["in_2025_queue"],
            item["review"],
            item["contract_amount_php"],
            item["award_date"],
        ),
        reverse=True,
    )
    return leads[:MAX_OVERLAP_LEADS]


def _compute_all_vendor_profiles(
    award_register: pd.DataFrame,
    award_eval_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], float]:
    """Build complete vendor concentration profiles across the integrated award register."""
    reg = award_register.copy()
    reg["_vkey"] = reg["awardee_name"].map(_normalize_key)
    reg["_clean_name"] = reg["awardee_name"].astype(str).str.strip()
    reg["_cat"] = reg["business_category"].replace("", MISSING_CATEGORY).fillna(MISSING_CATEGORY)
    reg["_ckey"] = reg["_cat"].map(_normalize_key)
    reg["contract_amount"] = pd.to_numeric(reg["contract_amount"], errors="coerce").fillna(0.0)
    reg["_day"] = pd.to_datetime(reg["award_date"], errors="coerce").dt.normalize()
    reg["_is_2025"] = reg["award_date"].astype(str).str.startswith("2025")

    total_spend = max(float(reg["contract_amount"].sum()), 1.0)
    valid_reg = reg.loc[reg["_vkey"] != ""].copy()
    if valid_reg.empty:
        return [], total_spend

    eval_cp = award_eval_df.copy()
    eval_cp["_vkey"] = eval_cp["awardee_name"].map(_normalize_key)
    eval_valid = eval_cp.loc[eval_cp["_vkey"] != ""]
    if not eval_valid.empty:
        ev_agg = eval_valid.groupby("_vkey", sort=False).agg(
            flags_2025=("review", "sum"),
            max_score_2025=("anomaly_score", "max"),
        )
        flags_by_v = {str(k): int(v) for k, v in ev_agg["flags_2025"].items()}
        max_score_by_v = {str(k): float(v) for k, v in ev_agg["max_score_2025"].items()}
    else:
        flags_by_v, max_score_by_v = {}, {}

    name_counts = (
        valid_reg.groupby(["_vkey", "_clean_name"], sort=False)
        .size()
        .reset_index(name="_n")
        .sort_values(["_vkey", "_n", "_clean_name"], ascending=[True, False, True])
        .drop_duplicates(subset=["_vkey"], keep="first")
    )
    canonical_by_v = dict(zip(name_counts["_vkey"], name_counts["_clean_name"]))

    cat_stats = (
        valid_reg.groupby(["_vkey", "_cat"], sort=False)["contract_amount"]
        .agg(cnt="size", total="sum")
        .reset_index()
        .sort_values(["_vkey", "cnt", "total"], ascending=[True, False, False])
        .drop_duplicates(subset=["_vkey"], keep="first")
    )
    top_cat_by_v = dict(zip(cat_stats["_vkey"], cat_stats["_cat"]))

    v_totals = valid_reg.groupby("_vkey", sort=False).agg(
        total_awards=("contract_amount", "size"),
        total_amount_php=("contract_amount", "sum"),
        awards_2025=("_is_2025", "sum"),
    )

    burst_by_v: dict[str, int] = {str(k): 1 for k in v_totals.index}
    dated = valid_reg.loc[valid_reg["_day"].notna()].sort_values(["_vkey", "_ckey", "_day"])
    if not dated.empty:
        vk_arr = dated["_vkey"].to_numpy()
        ck_arr = dated["_ckey"].to_numpy()
        day_arr = dated["_day"].to_numpy(dtype="datetime64[D]")
        pair_change = np.empty(len(dated), dtype=bool)
        pair_change[0] = True
        pair_change[1:] = (vk_arr[1:] != vk_arr[:-1]) | (ck_arr[1:] != ck_arr[:-1])
        starts = np.flatnonzero(pair_change)
        ends = np.append(starts[1:], len(dated))
        for s_i, e_i in zip(starts, ends):
            seg_len = int(e_i - s_i)
            if seg_len <= 1:
                continue
            vk = str(vk_arr[s_i])
            if seg_len <= burst_by_v.get(vk, 1):
                continue
            seg_days = day_arr[s_i:e_i]
            rights = np.searchsorted(
                seg_days, seg_days + np.timedelta64(SPLIT_CLUSTER_WINDOW_DAYS, "D"), side="right"
            )
            max_w = int((rights - np.arange(seg_len)).max())
            if max_w > burst_by_v.get(vk, 1):
                burst_by_v[vk] = max_w

    profiles: list[dict[str, Any]] = []
    for vkey, row in v_totals.iterrows():
        vk = str(vkey)
        spend = float(row["total_amount_php"])
        flags_cnt = int(flags_by_v.get(vk, 0))
        profiles.append({
            "vendor_key": vk,
            "awardee_name": str(canonical_by_v.get(vk, vk)),
            "total_awards": int(row["total_awards"]),
            "total_amount_php": round(spend, 2),
            "awards_2025": int(row["awards_2025"]),
            "flags_2025": flags_cnt,
            "flagged_awards": flags_cnt,
            "max_burst_30d": int(burst_by_v.get(vk, 1)),
            "top_category": str(top_cat_by_v.get(vk, MISSING_CATEGORY)),
            "share_pct": round(100.0 * spend / total_spend, 4),
            "max_score_2025": round(float(max_score_by_v.get(vk, 0.0)), 6),
        })

    profiles.sort(key=lambda v: (-v["flags_2025"], -v["total_amount_php"], v["awardee_name"]))
    return profiles, total_spend


def build_procurement_intelligence(
    integrated_data: IntegratedData,
    award_eval_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build vendor watchlists, 30-day split-risk clusters, and deduplicated overlap leads.

    @param integrated_data: Reconciled archive + official PhilGEPS IntegratedData bundle.
    @param award_eval_df: Scored 2025 held-out evaluation awards DataFrame.
    @returns: Dictionary containing vendors, split_clusters, overlaps, and summary statistics.
    """
    all_profiles, total_spend = _compute_all_vendor_profiles(integrated_data.award_register, award_eval_df)
    top_vendors = [
        {k: v for k, v in prof.items() if k != "vendor_key"}
        for prof in all_profiles[:MAX_TOP_VENDORS]
    ]

    reg = integrated_data.award_register.copy()
    reg["_vkey"] = reg["awardee_name"].map(_normalize_key)
    reg["_cat"] = reg["business_category"].replace("", MISSING_CATEGORY).fillna(MISSING_CATEGORY)
    reg["_ckey"] = reg["_cat"].map(_normalize_key)
    reg["contract_amount"] = pd.to_numeric(reg["contract_amount"], errors="coerce").fillna(0.0)
    reg["_day"] = pd.to_datetime(reg["award_date"], errors="coerce").dt.normalize()

    eval_ids = set(award_eval_df["record_id"].astype(str))
    flagged_ids = set(award_eval_df.loc[award_eval_df["review"], "record_id"].astype(str))
    eval_scores = dict(zip(award_eval_df["record_id"].astype(str), award_eval_df["anomaly_score"].astype(float)))
    split_clusters = _detect_split_clusters(reg, eval_ids, flagged_ids, eval_scores)
    overlaps = _build_overlap_leads(integrated_data, award_eval_df)

    top10_spend = sum(v["total_amount_php"] for v in sorted(all_profiles, key=lambda x: -x["total_amount_php"])[:10])
    summary = {
        "total_register_awards": int(len(integrated_data.award_register)),
        "distinct_vendors": int(len(all_profiles)),
        "top_vendors_returned": int(len(top_vendors)),
        "top10_vendor_spend_share_pct": round(100.0 * top10_spend / total_spend, 2),
        "split_cluster_count": int(len(split_clusters)),
        "svp_ceiling_cluster_count": int(sum(1 for c in split_clusters if c["near_svp_ceiling"])),
        "unresolved_pair_count": int(len(integrated_data.unresolved_links)),
        "unresolved_unique_records": int(integrated_data.unresolved_links["record_id"].nunique()),
        "overlaps_returned": int(len(overlaps)),
        "overlaps_in_2025_queue": int(sum(1 for o in overlaps if o["in_2025_queue"])),
    }
    return {
        "vendors": top_vendors,
        "split_clusters": split_clusters,
        "overlaps": overlaps,
        "summary": summary,
    }


class LiveXAIEngine:
    """Thread-safe lazy inference engine for live tender/award XAI triage and what-if simulation."""

    def __init__(self, data_dir: Path = ROOT / "data") -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._initialized = False

        self.integrated_data: IntegratedData | None = None
        self.award_screen: dict[str, Any] | None = None
        self.tender_screen: dict[str, Any] | None = None
        self.dpwh_screen: dict[str, Any] | None = None
        self.award_eval_with_links: pd.DataFrame | None = None
        self.vendor_index: list[dict[str, Any]] = []
        self._vendor_map: dict[str, dict[str, Any]] = {}
        self._explainers: dict[str, shap.TreeExplainer] = {}
        self._global_xai: dict[str, dict[str, Any]] = {}
        self._surface_cache: dict[tuple[str, int, int, int], dict[str, Any]] = {}

    @property
    def is_ready(self) -> bool:
        """Return True once the underlying Tagum models and indices have been loaded."""
        return self._initialized

    @property
    def _loaded(self) -> bool:
        """Compatibility property checked by XAIProvider.is_warmed in app.py."""
        return self._initialized

    def ensure_loaded(self) -> None:
        """Thread-safely initialize Tagum datasets, dual forests, SHAP explainers, and indices."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            data = build_integrated_data(self.data_dir)
            award = run_screen(self.data_dir)
            tender = run_tender_screen(data.tenders)
            dpwh_raw = pd.read_csv(self.data_dir / "tagum_dpwh.csv", dtype={"dpwh_contract_id": str})
            dpwh = run_dpwh_screen(dpwh_raw, data.flood_links)

            award_eval = award["eval"].merge(
                data.award_links[["record_id", *OFFICIAL_EVIDENCE]],
                on="record_id",
                how="left",
                validate="one_to_one",
            )
            all_vendors, _ = _compute_all_vendor_profiles(data.award_register, award_eval)
            vendor_clean = [{k: v for k, v in p.items() if k != "vendor_key"} for p in all_vendors]
            vendor_clean.sort(key=lambda v: (-v["total_awards"], -v["total_amount_php"], v["awardee_name"]))

            self.integrated_data = data
            self.award_screen = award
            self.tender_screen = tender
            self.dpwh_screen = dpwh
            self.award_eval_with_links = award_eval
            self.vendor_index = vendor_clean
            self._vendor_map = {p["vendor_key"]: {k: v for k, v in p.items() if k != "vendor_key"} for p in all_vendors}

            for kind, scr in (("award", award), ("tender", tender), ("dpwh", dpwh)):
                self._explainers[kind] = shap.TreeExplainer(
                    scr["forest"], feature_perturbation="tree_path_dependent"
                )
                x_ev = scr["views"]["Evaluation"]
                loc_d = scr.get("local_diffi")
                if loc_d is None:
                    loc_d = compute_local_diffi(scr["forest"], x_ev)
                self._global_xai[kind] = compute_global_xai_summary(
                    scr["forest"],
                    x_ev,
                    scr["effects"],
                    loc_d,
                    scr["exiffi"],
                    list(scr["features"].values()),
                    float(scr["cutoffs"]["Isolation Forest"]),
                )

            self._initialized = True

    def _explain_single(self, kind: str, scr: dict[str, Any], x_vec: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Compute exact single-row TreeSHAP path attributions and Local-DIFFI weights."""
        row_2d = np.asarray(x_vec, dtype=float).reshape(1, -1)
        explainer = self._explainers[kind]
        raw_shap = np.asarray(explainer.shap_values(row_2d))
        if raw_shap.ndim == 3 and raw_shap.shape[-1] == 1:
            raw_shap = raw_shap[..., 0]
        base = float(np.asarray(explainer.expected_value).item())
        loc_d = compute_local_diffi(scr["forest"], row_2d)[0]
        return raw_shap[0], base, loc_d

    def get_meta(self) -> dict[str, Any]:
        """Return dropdown vocabularies, calibrated cutoffs, global XAI importances, and 4 Tagum presets."""
        self.ensure_loaded()
        assert self.integrated_data is not None
        assert self.award_screen is not None
        assert self.tender_screen is not None
        assert self.dpwh_screen is not None

        categories = sorted({
            str(c).strip()
            for c in self.award_screen["processed"]["category"].dropna().unique()
            if str(c).strip() and str(c).strip() != MISSING_CATEGORY
        })
        modes = sorted({
            str(m).strip()
            for m in self.integrated_data.tenders["procurement_mode"].dropna().unique()
            if str(m).strip()
        })
        classifications = sorted({
            str(c).strip()
            for c in self.integrated_data.tenders["classification"].dropna().unique()
            if str(c).strip()
        })
        active_2025_flagged = [v for v in self.vendor_index if v.get("awards_2025", 0) > 0 and v.get("flags_2025", 0) > 0]
        active_2025_routine = [v for v in self.vendor_index if v.get("awards_2025", 0) > 0 and v.get("flags_2025", 0) == 0]
        top_vendor = (
            active_2025_flagged[0]["awardee_name"]
            if active_2025_flagged
            else (self.vendor_index[0]["awardee_name"] if self.vendor_index else "CM'S GENERAL MERCHANDISE AND CONSTRUCTION SUPPLY")
        )
        routine_vendor = (
            active_2025_routine[0]["awardee_name"]
            if active_2025_routine
            else (self.vendor_index[min(5, len(self.vendor_index) - 1)]["awardee_name"] if self.vendor_index else top_vendor)
        )

        presets = [
            {
                "id": "tender_svp_ceiling_rush",
                "kind": "tender",
                "label": "SVP Near ₱2M Ceiling · 3-Day Window",
                "description": "Pre-publication check on a Small Value Procurement notice near RA 12009 ₱2M threshold at quarter-end.",
                "payload": {
                    "notice_title": "RFQ - Supply & Delivery of Emergency Drainage & Road Rehab Materials (Tagum City)",
                    "approved_budget_php": 1_965_000.0,
                    "published_date": "2026-03-28",
                    "closing_date": "2026-03-31",
                    "notice_window_days": 3,
                    "procurement_mode": "Negotiated Procurement - Small Value Procurement (Sec. 53.9)",
                    "classification": "Goods",
                },
            },
            {
                "id": "tender_standard_public_bidding",
                "kind": "tender",
                "label": "Standard Competitive Public Bidding · 21 Days",
                "description": "Routine competitive goods tender with standard 21-day PhilGEPS posting window.",
                "payload": {
                    "notice_title": "PB - Procurement of Medical & Laboratory Supplies for City Health Office",
                    "approved_budget_php": 1_450_000.0,
                    "published_date": "2026-02-10",
                    "closing_date": "2026-03-03",
                    "notice_window_days": 21,
                    "procurement_mode": "Public Bidding",
                    "classification": "Goods",
                },
            },
            {
                "id": "award_split_burst_svp",
                "kind": "award",
                "label": "30-Day Same-Category Burst Near SVP Threshold",
                "description": "Repeat category award with tight ABC margin and elevated 30-day same-vendor award burst.",
                "payload": {
                    "award_title": "SUPPLY & DELIVERY OF CONSTRUCTION MATERIALS FOR BARANGAY DRAINAGE REHAB",
                    "awardee_name": top_vendor,
                    "category": "Construction Materials and Supplies",
                    "contract_amount": 982_500.0,
                    "award_date": "2025-09-29",
                    "approved_budget_php": 985_000.0,
                    "procurement_mode": "Negotiated Procurement - Small Value Procurement (Sec. 53.9)",
                    "vendor_category_30d_override": 5,
                    "vendor_share_pct_override": None,
                },
            },
            {
                "id": "award_routine_peer_supply",
                "kind": "award",
                "label": "Routine Peer-Median Supply Award",
                "description": "Competitive mid-quarter goods award priced near historical category median.",
                "payload": {
                    "award_title": "PROCUREMENT OF OFFICE EQUIPMENT & CONSUMABLES FOR CITY HALL OPERATIONS",
                    "awardee_name": routine_vendor,
                    "category": "General Merchandise",
                    "contract_amount": 185_000.0,
                    "award_date": "2025-05-14",
                    "approved_budget_php": 215_000.0,
                    "procurement_mode": "Public Bidding",
                    "vendor_category_30d_override": None,
                    "vendor_share_pct_override": None,
                },
            },
        ]

        return {
            "categories": categories,
            "modes": modes,
            "procurement_modes": modes,
            "classifications": classifications,
            "cutoffs": {
                "award": self.award_screen["cutoffs"],
                "tender": self.tender_screen["cutoffs"],
                "dpwh": self.dpwh_screen["cutoffs"],
            },
            "features": {
                "award": list(FEATURES.values()),
                "tender": list(TENDER_FEATURES.values()),
                "dpwh": list(DPWH_FEATURES.values()),
            },
            "feature_keys": {
                "award": list(FEATURES.keys()),
                "tender": list(TENDER_FEATURES.keys()),
                "dpwh": list(DPWH_FEATURES.keys()),
            },
            "global_xai": self._global_xai,
            "presets": presets,
        }

    def search_vendors(self, query: str = "", limit: int = 25) -> list[dict[str, Any]]:
        """Search historical Tagum awardees by prefix/substring with spend and 2025 flag stats."""
        self.ensure_loaded()
        cap = max(1, min(int(limit), 100))
        q_norm = _normalize_key(query)
        if not q_norm:
            return self.vendor_index[:cap]

        prefix_hits: list[dict[str, Any]] = []
        sub_hits: list[dict[str, Any]] = []
        for item in self.vendor_index:
            name_norm = item["awardee_name"].casefold()
            if name_norm.startswith(q_norm):
                prefix_hits.append(item)
            elif q_norm in name_norm:
                sub_hits.append(item)
        return (prefix_hits + sub_hits)[:cap]

    def score_custom_tender(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score a new or hypothetical Tagum tender notice with full 4-pillar XAI and peer context."""
        self.ensure_loaded()
        assert self.integrated_data is not None
        assert self.tender_screen is not None

        abc = _opt_float(payload.get("approved_budget_php"))
        if abc is None or abc <= 0:
            raise ValueError("approved_budget_php must be a positive number.")

        pub_raw = _clean_text(payload.get("published_date"))
        if not pub_raw:
            pub_raw = str(self.integrated_data.counts["latest_tender_publication_date"])
        try:
            pub_day = pd.Timestamp(pub_raw).normalize()
        except ValueError as err:
            raise ValueError(f"Invalid published_date: {pub_raw!r}") from err

        window_val = _opt_float(payload.get("notice_window_days"))
        close_raw = _clean_text(payload.get("closing_date"))
        if window_val is None and close_raw:
            try:
                close_day = pd.Timestamp(close_raw).normalize()
            except ValueError as err:
                raise ValueError(f"Invalid closing_date: {close_raw!r}") from err
            window_val = float((close_day - pub_day).days)
        if window_val is None or window_val <= 0:
            raise ValueError("notice_window_days must be positive (or closing_date must be after published_date).")
        window_days = int(max(1, round(window_val)))

        mode_str = _clean_text(payload.get("procurement_mode")) or "Public Bidding"
        class_str = _clean_text(payload.get("classification")) or "Goods"
        title_str = _clean_text(payload.get("notice_title")) or "(Untitled Tender Notice)"
        mode_grp = comparable_mode(mode_str)

        hist = self.tender_screen["processed"]
        hist_days = hist["published_day"].to_numpy(dtype="datetime64[D]")
        target_np = pub_day.to_datetime64().astype("datetime64[D]")
        max_hist_day = hist_days.max()
        anchor_day = (max_hist_day + np.timedelta64(1, "D")) if target_np > max_hist_day else target_np

        lo = int(np.searchsorted(hist_days, anchor_day - np.timedelta64(365, "D"), side="left"))
        hi = int(np.searchsorted(hist_days, anchor_day, side="left"))
        if hi - lo < MIN_PEERS:
            lo, hi = 0, max(MIN_PEERS, len(hist_days))

        sub_modes = hist["mode_group"].to_numpy()[lo:hi]
        sub_class = hist["classification"].to_numpy()[lo:hi]
        sub_bud = hist["approved_budget_php"].to_numpy(dtype=float)[lo:hi]
        sub_win = hist["notice_window_days"].to_numpy(dtype=float)[lo:hi]

        by_mode = sub_modes == mode_grp
        by_class = sub_class == class_str
        both = by_mode & by_class
        enough_mode = int(np.count_nonzero(by_mode)) >= MIN_PEERS
        enough_both = int(np.count_nonzero(both)) >= MIN_PEERS

        bud_peers = sub_bud[both] if enough_both else (sub_bud[by_mode] if enough_mode else sub_bud)
        win_peers = sub_win[by_mode] if enough_mode else sub_win
        bud_med = float(np.median(bud_peers))
        win_med = float(np.median(win_peers))
        prior_mode_share = float(100.0 * np.mean(by_mode))
        prior_class_share = float(100.0 * np.mean(by_class))
        q_days = int(((pub_day + pd.offsets.QuarterEnd(0)) - pub_day).days)

        x_vec = np.array([
            np.log10(abc / bud_med),
            np.log10(window_days / win_med),
            prior_mode_share,
            prior_class_share,
            float(q_days),
        ], dtype=float)

        scr = self.tender_screen
        if_score = float(-scr["forest"].score_samples(x_vec[None, :])[0])
        eif_arr, exiffi_arr = scr["extended"].score(scr["scaler"].transform(x_vec[None, :]), explain=True)
        eif_score = float(eif_arr[0])
        shap_vals, base_path, loc_d = self._explain_single("tender", scr, x_vec)
        _, imb_d = compute_imbalance_diffi(scr["forest"], x_vec[None, :], outlier_mask=np.array([True]))

        if_cut = float(scr["cutoffs"]["Isolation Forest"])
        eif_cut = float(scr["cutoffs"]["Extended Isolation Forest"])
        is_if = if_score > if_cut
        is_eif = eif_score > eif_cut
        feat_names = list(TENDER_FEATURES.values())
        contexts = [
            f"₱{abc:,.0f} vs ₱{bud_med:,.0f} peer median",
            f"{window_days}d vs {win_med:.0f}d peer median",
            f"{prior_mode_share:.1f}% prior mode share",
            f"{prior_class_share:.1f}% prior class share",
            f"{q_days}d to quarter end",
        ]

        row_dict = {
            "notice_title": title_str,
            "approved_budget_php": round(abc, 2),
            "published_date": pub_day.strftime("%Y-%m-%d"),
            "closing_date": close_raw or (pub_day + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d"),
            "notice_window_days": window_days,
            "procurement_mode": mode_str,
            "mode_group": mode_grp,
            "classification": class_str,
            "budget_peer_median_php": round(bud_med, 2),
            "window_peer_median_days": round(win_med, 2),
            "prior_mode_share": round(prior_mode_share, 4),
            "prior_class_share": round(prior_class_share, 4),
            "days_to_quarter_end": q_days,
        }

        eval_df = scr["eval"]
        peer_ids = find_nearest_peers(
            eval_df, scr["views"]["Evaluation"], None, "mode_group", mode_grp,
            "bid_reference", k=4, target_vec=x_vec,
        )
        peer_map = {str(r["bid_reference"]): r for r in _sanitize_json_records(eval_df)}
        peers = [
            {
                "bid_reference": pid,
                "notice_title": peer_map[pid]["notice_title"],
                "approved_budget_php": round(float(peer_map[pid]["approved_budget_php"]), 2),
                "notice_window_days": int(peer_map[pid]["notice_window_days"]),
                "procurement_mode": peer_map[pid]["procurement_mode"],
                "classification": peer_map[pid]["classification"],
                "published_date": str(peer_map[pid]["published_date"])[:10],
                "anomaly_score": round(float(peer_map[pid]["anomaly_score"]), 6),
                "review": bool(peer_map[pid]["review"]),
            }
            for pid in peer_ids
            if pid in peer_map
        ]

        recourse_obj = compute_counterfactual_recourse("tender", row_dict, x_vec, scr)
        checks_obj = statutory_procurement_checks("tender", row_dict)
        return {
            **row_dict,
            "kind": "tender",
            "input": row_dict,
            "feature_names": feat_names,
            "feature_vector": [round(float(v), 6) for v in x_vec],
            "anomaly_score": round(if_score, 6),
            "extended_score": round(eif_score, 6),
            "cutoff": round(if_cut, 6),
            "extended_cutoff": round(eif_cut, 6),
            "review": bool(is_if),
            "extended_review": bool(is_eif),
            "tree_base": round(float(base_path), 6),
            "tree_path_length": round(float(base_path + float(np.sum(shap_vals))), 6),
            "tree_shap": [round(float(v), 6) for v in shap_vals],
            "local_diffi": [round(float(v), 6) for v in loc_d],
            "imbalance_diffi": [round(float(v), 6) for v in imb_d[0]],
            "exiffi": [round(float(v), 6) for v in exiffi_arr[0]],
            "waterfall": build_shap_waterfall(
                shap_vals, base_path, int(scr["forest"].max_samples_), feat_names, if_cut, contexts
            ),
            "consensus": compute_xai_consensus(shap_vals, loc_d, exiffi_arr[0], feat_names, is_if, is_eif),
            "recourse": recourse_obj,
            "counterfactual": recourse_obj,
            "statutory_checks": checks_obj,
            "statutory_flags": checks_obj,
            "peer_ids": peer_ids,
            "peers": peers,
            "nearest_peers": peers,
        }

    def score_custom_award(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Score a new or hypothetical Tagum contract award using real vendor & category history."""
        self.ensure_loaded()
        assert self.award_screen is not None
        assert self.award_eval_with_links is not None

        amount = _opt_float(payload.get("contract_amount"))
        if amount is None or amount <= 0:
            raise ValueError("contract_amount must be a positive number.")

        date_raw = _clean_text(payload.get("award_date")) or "2025-12-15"
        try:
            award_day = pd.Timestamp(date_raw).normalize()
        except ValueError as err:
            raise ValueError(f"Invalid award_date: {date_raw!r}") from err

        title_str = _clean_text(payload.get("award_title")) or "(Untitled Contract Award)"
        vendor_str = _clean_text(payload.get("awardee_name")) or "(Unspecified Vendor)"
        cat_str = _clean_text(payload.get("category")) or MISSING_CATEGORY
        abc_opt = _opt_float(payload.get("approved_budget_php"))
        mode_opt = _clean_text(payload.get("procurement_mode")) or None

        v_key = _normalize_key(vendor_str)
        c_key = _normalize_key(cat_str)

        hist = self.award_screen["processed"]
        hist_days = hist["award_date"].dt.normalize().to_numpy(dtype="datetime64[D]")
        target_np = award_day.to_datetime64().astype("datetime64[D]")
        max_hist_day = hist_days.max()
        anchor_day = (max_hist_day + np.timedelta64(1, "D")) if target_np > max_hist_day else target_np

        lo_365 = int(np.searchsorted(hist_days, anchor_day - np.timedelta64(365, "D"), side="left"))
        hi = int(np.searchsorted(hist_days, anchor_day, side="left"))
        lo_30 = int(np.searchsorted(hist_days, anchor_day - np.timedelta64(30, "D"), side="left"))
        if hi - lo_365 < MIN_PEERS:
            lo_365, hi = 0, max(MIN_PEERS, len(hist_days))
            lo_30 = max(0, hi - 30)

        sub_cats = hist["category_key"].to_numpy()[lo_365:hi]
        sub_vendors = hist["vendor_key"].to_numpy()[lo_365:hi]
        sub_money = hist["contract_amount"].to_numpy(dtype=float)[lo_365:hi]

        cat_matches = sub_cats == c_key
        cat_365_count = int(np.count_nonzero(cat_matches))
        peer_money = sub_money[cat_matches] if cat_365_count >= MIN_PEERS else sub_money
        peer_med = float(np.median(peer_money))
        peer_scope = "Category" if cat_365_count >= MIN_PEERS else "Citywide fallback"

        hist_share = float(100.0 * np.mean(sub_vendors == v_key)) if v_key else 0.0
        rec_cats = hist["category_key"].to_numpy()[lo_30:hi]
        rec_vendors = hist["vendor_key"].to_numpy()[lo_30:hi]
        hist_30d = int(np.sum((rec_vendors == v_key) & (rec_cats == c_key))) if v_key else 0

        share_override = _opt_float(payload.get("vendor_share_pct_override"))
        burst_override = _opt_float(payload.get("vendor_category_30d_override"))
        vendor_share = max(0.0, share_override) if share_override is not None else hist_share
        vendor_30d = max(0, int(round(burst_override))) if burst_override is not None else hist_30d
        q_days = int(((award_day + pd.offsets.QuarterEnd(0)) - award_day).days)

        x_vec = np.array([
            np.log10(amount / peer_med),
            vendor_share,
            float(vendor_30d),
            float(q_days),
            float(cat_365_count),
        ], dtype=float)

        scr = self.award_screen
        if_score = float(-scr["forest"].score_samples(x_vec[None, :])[0])
        eif_arr, exiffi_arr = scr["extended"].score(scr["scaler"].transform(x_vec[None, :]), explain=True)
        eif_score = float(eif_arr[0])
        shap_vals, base_path, loc_d = self._explain_single("award", scr, x_vec)
        _, imb_d = compute_imbalance_diffi(scr["forest"], x_vec[None, :], outlier_mask=np.array([True]))

        if_cut = float(scr["cutoffs"]["Isolation Forest"])
        eif_cut = float(scr["cutoffs"]["Extended Isolation Forest"])
        is_if = if_score > if_cut
        is_eif = eif_score > eif_cut
        feat_names = list(FEATURES.values())
        contexts = [
            f"₱{amount:,.0f} vs ₱{peer_med:,.0f} ({peer_scope})",
            f"{vendor_share:.2f}% prior 365d share",
            f"{vendor_30d} same-category awards in 30d",
            f"{q_days}d to quarter end",
            f"{cat_365_count} prior category awards",
        ]

        row_dict = {
            "award_title": title_str,
            "awardee_name": vendor_str,
            "category": cat_str,
            "contract_amount": round(amount, 2),
            "award_date": award_day.strftime("%Y-%m-%d"),
            "approved_budget_php": round(abc_opt, 2) if abc_opt is not None and abc_opt > 0 else None,
            "procurement_mode": mode_opt,
            "peer_median_php": round(peer_med, 2),
            "peer_scope": peer_scope,
            "vendor_share_pct": round(vendor_share, 4),
            "vendor_category_30d": int(vendor_30d),
            "days_to_quarter_end": q_days,
            "category_365d": cat_365_count,
        }

        vendor_profile = self._vendor_map.get(v_key)
        if vendor_profile is not None:
            v_prof_out = {"is_known_vendor": True, **vendor_profile}
        else:
            v_prof_out = {
                "is_known_vendor": False,
                "awardee_name": vendor_str,
                "total_awards": 0,
                "total_amount_php": 0.0,
                "awards_2025": 0,
                "flags_2025": 0,
                "max_burst_30d": 0,
                "top_category": cat_str,
                "share_pct": 0.0,
                "max_score_2025": 0.0,
            }

        eval_df = self.award_eval_with_links
        peer_ids = find_nearest_peers(
            eval_df, scr["views"]["Evaluation"], None, "category", cat_str,
            "record_id", k=4, target_vec=x_vec,
        )
        peer_map = {str(r["record_id"]): r for r in _sanitize_json_records(eval_df)}
        peers = [
            {
                "record_id": pid,
                "award_title": peer_map[pid]["award_title"],
                "awardee_name": peer_map[pid]["awardee_name"],
                "category": peer_map[pid]["category"],
                "contract_amount": round(float(peer_map[pid]["contract_amount"]), 2),
                "award_date": str(peer_map[pid]["award_date"])[:10],
                "anomaly_score": round(float(peer_map[pid]["anomaly_score"]), 6),
                "review": bool(peer_map[pid]["review"]),
            }
            for pid in peer_ids
            if pid in peer_map
        ]

        recourse_obj = compute_counterfactual_recourse("award", row_dict, x_vec, scr)
        checks_obj = statutory_procurement_checks("award", row_dict)
        return {
            **row_dict,
            "kind": "award",
            "input": row_dict,
            "vendor_profile": v_prof_out,
            "feature_names": feat_names,
            "feature_vector": [round(float(v), 6) for v in x_vec],
            "anomaly_score": round(if_score, 6),
            "extended_score": round(eif_score, 6),
            "cutoff": round(if_cut, 6),
            "extended_cutoff": round(eif_cut, 6),
            "review": bool(is_if),
            "extended_review": bool(is_eif),
            "tree_base": round(float(base_path), 6),
            "tree_path_length": round(float(base_path + float(np.sum(shap_vals))), 6),
            "tree_shap": [round(float(v), 6) for v in shap_vals],
            "local_diffi": [round(float(v), 6) for v in loc_d],
            "imbalance_diffi": [round(float(v), 6) for v in imb_d[0]],
            "exiffi": [round(float(v), 6) for v in exiffi_arr[0]],
            "waterfall": build_shap_waterfall(
                shap_vals, base_path, int(scr["forest"].max_samples_), feat_names, if_cut, contexts
            ),
            "consensus": compute_xai_consensus(shap_vals, loc_d, exiffi_arr[0], feat_names, is_if, is_eif),
            "recourse": recourse_obj,
            "counterfactual": recourse_obj,
            "statutory_checks": checks_obj,
            "statutory_flags": checks_obj,
            "peer_ids": peer_ids,
            "peers": peers,
            "nearest_peers": peers,
        }

    def _screen_and_id_col(self, kind: str) -> tuple[dict[str, Any], pd.DataFrame, str]:
        """Resolve screen dictionary, evaluation DataFrame, and key column for kind."""
        assert self.award_screen is not None
        assert self.tender_screen is not None
        assert self.dpwh_screen is not None
        assert self.award_eval_with_links is not None

        if kind == "award":
            return self.award_screen, self.award_eval_with_links, "record_id"
        if kind == "tender":
            return self.tender_screen, self.tender_screen["eval"], "bid_reference"
        if kind == "dpwh":
            return self.dpwh_screen, self.dpwh_screen["eval"], "dpwh_contract_id"
        raise ValueError(f"Unsupported screen kind: {kind!r}")

    def simulate_whatif(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply live domain-unit overrides to an existing record or median baseline and recompute XAI."""
        self.ensure_loaded()
        kind = _clean_text(payload.get("kind")).casefold()
        scr, eval_df, id_col = self._screen_and_id_col(kind)
        rec_id_raw = _clean_text(payload.get("record_id"))
        overrides = payload.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be a dictionary.")

        if rec_id_raw:
            matches = eval_df.index[eval_df[id_col].astype(str) == rec_id_raw].to_numpy()
            if len(matches) == 0:
                raise ValueError(f"Unknown {kind} {id_col}: {rec_id_raw!r}")
            pos = int(matches[0])
            x_base = scr["views"]["Evaluation"][pos].copy()
            base_row = eval_df.iloc[pos].to_dict()
        else:
            x_base = np.median(scr["views"]["Training"], axis=0).copy()
            base_row = {}

        orig_score = float(-scr["forest"].score_samples(x_base[None, :])[0])
        x_mod = x_base.copy()

        if kind == "award":
            peer_med = max(_opt_float(base_row.get("peer_median_php")) or 150_000.0, 1.0)
            if (amt := _opt_float(overrides.get("contract_amount"))) is not None:
                if amt <= 0:
                    raise ValueError("contract_amount override must be positive.")
                x_mod[0] = np.log10(amt / peer_med)
            elif (alr := _opt_float(overrides.get("amount_log_ratio"))) is not None:
                x_mod[0] = alr
            if (vs := _opt_float(overrides.get("vendor_share_pct"))) is not None:
                x_mod[1] = max(0.0, vs)
            if (vc := _opt_float(overrides.get("vendor_category_30d"))) is not None:
                x_mod[2] = max(0.0, vc)
            if (dq := _opt_float(overrides.get("days_to_quarter_end"))) is not None:
                x_mod[3] = max(0.0, dq)
            if (c365 := _opt_float(overrides.get("category_365d"))) is not None:
                x_mod[4] = max(0.0, c365)

        elif kind == "tender":
            bud_med = max(_opt_float(base_row.get("budget_peer_median_php")) or 300_000.0, 1.0)
            win_med = max(_opt_float(base_row.get("window_peer_median_days")) or 7.0, 1.0)
            if (abc := _opt_float(overrides.get("approved_budget_php"))) is not None:
                if abc <= 0:
                    raise ValueError("approved_budget_php override must be positive.")
                x_mod[0] = np.log10(abc / bud_med)
            elif (blr := _opt_float(overrides.get("budget_log_ratio"))) is not None:
                x_mod[0] = blr
            if (win := _opt_float(overrides.get("notice_window_days"))) is not None:
                if win <= 0:
                    raise ValueError("notice_window_days override must be positive.")
                x_mod[1] = np.log10(win / win_med)
            elif (wlr := _opt_float(overrides.get("window_log_ratio"))) is not None:
                x_mod[1] = wlr
            if (pms := _opt_float(overrides.get("prior_mode_share"))) is not None:
                x_mod[2] = float(np.clip(pms, 0.0, 100.0))
            if (pcs := _opt_float(overrides.get("prior_class_share"))) is not None:
                x_mod[3] = float(np.clip(pcs, 0.0, 100.0))
            if (dq := _opt_float(overrides.get("days_to_quarter_end"))) is not None:
                x_mod[4] = max(0.0, dq)

        else:  # dpwh
            abc_med = max(_opt_float(base_row.get("abc_peer_median_php")) or 9_000_000.0, 1.0)
            for r_key in ("cost_abc_ratio", "cost_abc_ratio_feat"):
                if (rat := _opt_float(overrides.get(r_key))) is not None:
                    if rat <= 0:
                        raise ValueError("cost_abc_ratio override must be positive.")
                    x_mod[0] = rat
                    break
            if (abc_p := _opt_float(overrides.get("abc_php"))) is not None:
                if abc_p <= 0:
                    raise ValueError("abc_php override must be positive.")
                x_mod[1] = np.log10(abc_p / abc_med)
            elif (alr := _opt_float(overrides.get("abc_log_ratio"))) is not None:
                x_mod[1] = alr
            for v_key in ("variance_days", "variance_days_feat"):
                if (vd := _opt_float(overrides.get(v_key))) is not None:
                    x_mod[2] = vd
                    break
            if (dur := _opt_float(overrides.get("planned_duration_days"))) is not None:
                x_mod[3] = max(1.0, dur)
            if (dens := _opt_float(overrides.get("barangay_project_density"))) is not None:
                x_mod[4] = max(0.0, dens)

        if_score = float(-scr["forest"].score_samples(x_mod[None, :])[0])
        eif_arr, exiffi_arr = scr["extended"].score(scr["scaler"].transform(x_mod[None, :]), explain=True)
        eif_score = float(eif_arr[0])
        shap_vals, base_path, loc_d = self._explain_single(kind, scr, x_mod)

        if_cut = float(scr["cutoffs"]["Isolation Forest"])
        eif_cut = float(scr["cutoffs"]["Extended Isolation Forest"])
        is_if = if_score > if_cut
        is_eif = eif_score > eif_cut
        feat_names = list(scr["features"].values())

        return {
            "kind": kind,
            "record_id": rec_id_raw or None,
            "original_score": round(orig_score, 6),
            "anomaly_score": round(if_score, 6),
            "extended_score": round(eif_score, 6),
            "review": bool(is_if),
            "extended_review": bool(is_eif),
            "cutoff": round(if_cut, 6),
            "extended_cutoff": round(eif_cut, 6),
            "feature_vector": [round(float(v), 6) for v in x_mod],
            "waterfall": build_shap_waterfall(
                shap_vals, base_path, int(scr["forest"].max_samples_), feat_names, if_cut
            ),
            "tree_shap": [round(float(v), 6) for v in shap_vals],
            "local_diffi": [round(float(v), 6) for v in loc_d],
            "exiffi": [round(float(v), 6) for v in exiffi_arr[0]],
            "consensus": compute_xai_consensus(shap_vals, loc_d, exiffi_arr[0], feat_names, is_if, is_eif),
        }

    def get_2d_surface(
        self,
        kind: str,
        fx: int,
        fy: int,
        grid_size: int = 24,
    ) -> dict[str, Any]:
        """Return (and cache) the 2D IF vs EIF + ExIFFI decision surface for (kind, fx, fy)."""
        self.ensure_loaded()
        norm_kind = _clean_text(kind).casefold()
        scr, _, _ = self._screen_and_id_col(norm_kind)
        cache_key = (norm_kind, int(fx), int(fy), int(grid_size))
        with self._lock:
            cached = self._surface_cache.get(cache_key)
        if cached is not None:
            return cached

        surface = compute_2d_decision_surface(scr, int(fx), int(fy), grid_size=int(grid_size))
        with self._lock:
            self._surface_cache[cache_key] = surface
        return surface
