"""Retrospective tender screening using official publication fields."""

import numpy as np
import pandas as pd

from screen import MIN_PEERS, compare_stress_scenarios, evaluate, fit_screen

TENDER_FEATURES = {
    "budget_log_ratio": "Budget vs. recent peers",
    "window_log_ratio": "Bid window vs. recent peers",
    "prior_mode_share": "Prior tenders in this procurement mode",
    "prior_class_share": "Prior tenders in this classification",
    "days_to_quarter_end": "Days to quarter end",
}
FIT_START = pd.Timestamp("2024-04-01")
CALIBRATION_START = pd.Timestamp("2025-01-01")
EVALUATION_START = pd.Timestamp("2026-01-01")


def comparable_mode(value):
    """Keep official labels for display and group SVP labels for historical comparison."""
    mode = " ".join(str(value).split())
    if "small value procurement" in mode.casefold():
        return "Small Value Procurement"
    return mode


def tender_features(tenders):
    frame = tenders.copy()
    frame["published_day"] = pd.to_datetime(frame.published_date, format="%Y-%m-%d")
    frame = frame.sort_values(["published_day", "bid_reference"], ignore_index=True)
    frame["mode_group"] = frame.procurement_mode.map(comparable_mode)
    day = frame.published_day.to_numpy(dtype="datetime64[D]")
    start = np.searchsorted(day, day - np.timedelta64(365, "D"), side="left")
    end = np.searchsorted(day, day, side="left")
    mode = frame.mode_group.to_numpy()
    tender_class = frame.classification.to_numpy()
    budget = frame.approved_budget_php.to_numpy(dtype=float)
    window = frame.notice_window_days.to_numpy(dtype=float)
    n = len(frame)
    budget_baseline = np.full(n, np.nan)
    window_baseline = np.full(n, np.nan)
    mode_share = np.full(n, np.nan)
    class_share = np.full(n, np.nan)
    budget_context = np.full(n, "Insufficient history", dtype=object)
    window_context = np.full(n, "Insufficient history", dtype=object)
    for i, (first, last) in enumerate(zip(start, end)):
        if first == last:
            continue
        by_mode = mode[first:last] == mode[i]
        by_class = tender_class[first:last] == tender_class[i]
        both = by_mode & by_class
        mode_share[i] = 100 * by_mode.mean()
        class_share[i] = 100 * by_class.mean()
        enough_mode = np.count_nonzero(by_mode) >= MIN_PEERS
        enough_both = np.count_nonzero(both) >= MIN_PEERS
        mask = both if enough_both else by_mode if enough_mode else slice(None)
        budget_baseline[i] = np.median(budget[first:last][mask])
        budget_context[i] = (
            "Same mode and classification" if enough_both else "Same mode" if enough_mode else "All city tenders"
        )
        window_mask = by_mode if enough_mode else slice(None)
        window_baseline[i] = np.median(window[first:last][window_mask])
        window_context[i] = "Same mode" if enough_mode else "All city tenders"

    frame["budget_peer_median_php"] = budget_baseline
    frame["window_peer_median_days"] = window_baseline
    frame["budget_peer_group"] = budget_context
    frame["window_peer_group"] = window_context
    frame["budget_log_ratio"] = np.log10(budget / budget_baseline)
    frame["window_log_ratio"] = np.log10(window / window_baseline)
    frame["prior_mode_share"] = mode_share
    frame["prior_class_share"] = class_share
    frame["days_to_quarter_end"] = (
        (frame.published_day + pd.offsets.QuarterEnd(0)) - frame.published_day
    ).dt.days
    periods = [
        frame.published_day < FIT_START, frame.published_day < CALIBRATION_START,
        frame.published_day < EVALUATION_START,
    ]
    frame["split"] = np.select(periods, ["Warm-up", "Training", "Calibration"], default="Evaluation")
    return frame


def run_tender_screen(tenders):
    return fit_screen(tender_features(tenders), TENDER_FEATURES, "Budget-only baseline")


def tender_window_scenario(row, x, result):
    train = result["groups"]["Training"]
    peer_windows = train.loc[train.mode_group.eq(row.mode_group), "notice_window_days"].to_numpy()
    if len(peer_windows) < MIN_PEERS:
        return {"scope": "Same procurement mode", "points": [], "closest": None}
    values = np.unique(np.quantile(peer_windows, np.linspace(0.05, 0.95, 21), method="nearest"))
    trial = np.repeat(x[None, :], len(values), axis=0)
    trial[:, 1] = np.log10(values / row.window_peer_median_days)
    scores = -result["forest"].score_samples(trial)
    points = [{"days": int(days), "score": float(score)} for days, score in zip(values, scores)]
    cutoff = result["cutoffs"]["Isolation Forest"]
    candidates = [
        point for point in points if point["days"] != row.notice_window_days and point["score"] <= cutoff
    ]
    closest = None
    if bool(row.review) and candidates:
        closest = min(candidates, key=lambda p: abs(p["days"] - row.notice_window_days))
    return {"scope": "Same procurement mode", "points": points, "closest": closest}


def tender_stress(result, rounds):
    test = result["views"]["Evaluation"]
    rows = result["eval"]
    shorter = np.zeros_like(test)
    windows = rows.notice_window_days.to_numpy()
    shorter[:, 1] = np.log10(np.maximum(1, windows // 2) / windows)
    bigger = np.array([np.log10(4), 0, 0, 0, 0])
    eligible = np.arange(len(rows))
    return compare_stress_scenarios(result, [
        ("Budget ×4", eligible, bigger),
        ("Bid window halved", eligible, shorter),
        ("Budget ×4 and window halved", eligible, shorter + bigger),
    ], rounds)


def evaluate_tenders(result, rounds=8):
    metrics = evaluate(result, rounds, compare=tender_stress, scenario_for=tender_window_scenario)
    metrics["window_scenarios"] = metrics.pop("amount_scenarios")
    return metrics
