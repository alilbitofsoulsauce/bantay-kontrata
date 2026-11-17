"""Fit retrospective Tagum procurement screens and explain held-out awards."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent
FEATURES = {
    "amount_log_ratio": "Amount vs. recent peers",
    "vendor_share_pct": "Vendor's prior award share",
    "vendor_category_30d": "Recent same-category vendor awards",
    "days_to_quarter_end": "Days to quarter end",
    "category_365d": "Prior awards in category",
}
TRAIN_FROM = pd.Timestamp("2018-01-01")
CALIBRATE_FROM = pd.Timestamp("2024-01-01")
TEST_FROM = pd.Timestamp("2025-01-01")
REVIEW_TAIL = 0.05
SEED = 42
TREES = 160
MIN_PEERS = 15
MISSING_CATEGORY = "(Category missing)"


def load_awards(data_dir=ROOT / "data"):
    awards = pd.read_csv(
        data_dir / "tagum_awards.csv",
        parse_dates=["award_date"],
        dtype={"reference_id": "string", "contract_no": "string", "record_id": "string"},
    )
    return awards.sort_values(["award_date", "record_id"]).reset_index(drop=True)


def make_features(awards):
    """Use the previous 365/30 calendar days, excluding every award on the scored day."""
    frame = awards.copy()
    frame["category"] = frame.business_category.fillna(MISSING_CATEGORY)
    frame["category_key"] = frame.category.str.casefold().str.split().str.join(" ")
    frame["vendor_key"] = frame.awardee_name.fillna("").str.casefold().str.split().str.join(" ")
    parsed_day = frame.award_date.dt.normalize()
    day = parsed_day.to_numpy(dtype="datetime64[D]")
    vendor = frame.vendor_key.to_numpy()
    category = frame.category_key.to_numpy()
    money = frame.contract_amount.to_numpy(dtype=float)
    hist_median = np.full(len(frame), np.nan)
    cat_count = np.zeros(len(frame), dtype=int)
    vendor_share = np.full(len(frame), np.nan)
    vendor_recent = np.full(len(frame), np.nan)
    recent_other_vendors = np.zeros(len(frame), dtype=int)
    lower = np.searchsorted(day, day - np.timedelta64(365, "D"), side="left")
    upper = np.searchsorted(day, day, side="left")
    recent = np.searchsorted(day, day - np.timedelta64(30, "D"), side="left")

    for i, (start, end, month_start) in enumerate(zip(lower, upper, recent)):
        if start == end:
            continue
        category_history = category[start:end] == category[i]
        cat_count[i] = int(category_history.sum())
        peers = money[start:end][category_history] if cat_count[i] >= MIN_PEERS else money[start:end]
        hist_median[i] = float(np.median(peers))
        if vendor[i]:
            vendor_share[i] = 100 * np.mean(vendor[start:end] == vendor[i])
            recent_category = category[month_start:end] == category[i]
            recent_vendor = vendor[month_start:end]
            vendor_recent[i] = np.sum((recent_vendor == vendor[i]) & recent_category)
            recent_other_vendors[i] = np.sum(
                (recent_vendor != vendor[i]) & (recent_vendor != "") & recent_category
            )

    frame["peer_median_php"] = hist_median
    frame["peer_scope"] = np.where(cat_count >= MIN_PEERS, "Category", "Citywide fallback")
    frame["amount_log_ratio"] = np.log10(money / hist_median)
    frame["vendor_share_pct"] = vendor_share
    frame["vendor_category_30d"] = vendor_recent
    frame["days_to_quarter_end"] = ((parsed_day + pd.offsets.QuarterEnd(0)) - parsed_day).dt.days
    frame["category_365d"] = cat_count
    frame["prior_award_count"] = upper - lower
    frame["recent_other_vendors"] = recent_other_vendors

    split = [
        frame.award_date < TRAIN_FROM,
        frame.award_date < CALIBRATE_FROM,
        frame.award_date < TEST_FROM,
    ]
    frame["split"] = np.select(split, ["Warm-up", "Training", "Calibration"], default="Evaluation")
    return frame


def new_iforest(seed=SEED):
    return IsolationForest(
        n_estimators=TREES, max_samples=256, max_features=1.0, contamination="auto",
        random_state=seed, n_jobs=1,
    )


def path_adjustment(count):
    """The same average unfinished path adjustment as scikit-learn."""
    count = np.asarray(count, dtype=float)
    result = np.zeros(count.shape)
    result[count == 2] = 1.0
    rich = count > 2
    n = count[rich]
    result[rich] = 2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n
    return result.item() if result.ndim == 0 else result


def explain_paths(forest, x):
    explainer = shap.TreeExplainer(forest, feature_perturbation="tree_path_dependent")
    effects = np.asarray(explainer.shap_values(x))
    if effects.ndim == 3 and effects.shape[-1] == 1:
        effects = effects[..., 0]
    base = float(np.asarray(explainer.expected_value).item())
    h_from_score = -np.log2(-forest.score_samples(x)) * path_adjustment(forest.max_samples_)
    error = np.abs(h_from_score - (base + effects.sum(axis=1)))
    np.testing.assert_allclose(base + effects.sum(axis=1), h_from_score, atol=1e-8, rtol=0)
    return effects, base, error


def local_diffi(forest, x):
    """Carletti et al., Algorithm 3, using unadjusted edge depth for the local weights."""
    x = np.asarray(x, dtype=np.float32)
    importance = np.zeros((len(x), x.shape[1]))
    visits = np.zeros_like(importance)
    max_depth = int(np.ceil(np.log2(forest.max_samples_)))
    for estimator, cols in zip(forest.estimators_, forest.estimators_features_):
        route = estimator.decision_path(x[:, cols]).tocoo()
        splits = estimator.tree_.feature[route.col]
        internal = splits >= 0
        visited_rows = route.row[internal]
        visited_features = cols[splits[internal]]
        depths = np.bincount(visited_rows, minlength=len(x))
        weights = 1 / depths[visited_rows] - 1 / max_depth
        np.add.at(importance, (visited_rows, visited_features), weights)
        np.add.at(visits, (visited_rows, visited_features), 1)
    return np.divide(importance, visits, out=np.zeros_like(importance), where=visits > 0)


@dataclass
class Node:
    count: int
    normal: np.ndarray | None = None
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None


def grow_oblique_tree(x, remaining, rng):
    node = Node(count=len(x))
    if len(x) <= 1 or remaining == 0:
        return node
    lower = x.min(axis=0)
    upper = x.max(axis=0)
    if np.all(lower == upper):
        return node
    # Hariri et al. draw a Gaussian normal and a point from the node bounding box.
    normal = rng.normal(size=x.shape[1])
    normal /= np.linalg.norm(normal)
    projection = x @ normal
    for _ in range(12):
        threshold = float(rng.uniform(lower, upper) @ normal)
        left_mask = projection <= threshold
        if left_mask.any() and (~left_mask).any():
            node.normal = normal
            node.threshold = threshold
            node.left = grow_oblique_tree(x[left_mask], remaining - 1, rng)
            node.right = grow_oblique_tree(x[~left_mask], remaining - 1, rng)
            break
    return node


def walk_oblique(node, x, rows, depth, paths, importance=None, normals=None):
    if len(rows) == 0:
        return
    if node.normal is None:
        paths[rows] += depth + path_adjustment(node.count)
        return

    mask = x[rows] @ node.normal <= node.threshold
    weight = np.abs(node.normal)
    for child, chosen in [(node.left, rows[mask]), (node.right, rows[~mask])]:
        if importance is not None:
            # Arcudi et al. reference implementation uses child_count + 1.
            importance[chosen] += weight * node.count / (child.count + 1)
            normals[chosen] += weight
        walk_oblique(child, x, chosen, depth + 1, paths, importance, normals)


class ExtendedForest:
    """Full oblique EIF with depth correction and ExIFFI local importance."""

    def __init__(self, x, seed=SEED, trees=TREES):
        rng = np.random.default_rng(seed)
        self.sample_size = min(256, len(x))
        self.depth_limit = int(np.ceil(np.log2(self.sample_size)))
        self.roots = [
            grow_oblique_tree(x[rng.choice(len(x), self.sample_size, replace=False)], self.depth_limit, rng)
            for _ in range(trees)
        ]

    def score(self, x, explain=False):
        paths = np.zeros(len(x))
        importance = np.zeros_like(x, dtype=float) if explain else None
        normals = np.zeros_like(x, dtype=float) if explain else None
        rows = np.arange(len(x))
        for root in self.roots:
            walk_oblique(root, x, rows, 0, paths, importance, normals)
        scores = 2 ** (-(paths / len(self.roots)) / path_adjustment(self.sample_size))
        if explain:
            importance = np.divide(
                importance, normals, out=np.zeros_like(importance), where=normals > 0,
            )
        return scores, importance


def score_cutoff(scores):
    return float(np.quantile(scores, 1 - REVIEW_TAIL, method="higher"))


def run_screen(data_dir=ROOT / "data"):
    return fit_screen(make_features(load_awards(data_dir)), FEATURES, "Amount-only baseline")


def fit_screen(processed, features, baseline_name):
    """Fit both forests on the training window and set separate calibration cutoffs."""
    finite = np.isfinite(processed[list(features)]).all(axis=1)
    usable = processed.loc[finite & (processed.split != "Warm-up")].copy()
    groups = {
        name: usable.loc[usable.split.eq(name)].reset_index(drop=True)
        for name in ("Training", "Calibration", "Evaluation")
    }
    if min(map(len, groups.values())) < 30:
        raise ValueError("Every chronological period needs at least 30 complete rows.")
    views = {name: rows[list(features)].to_numpy(dtype=float) for name, rows in groups.items()}
    fit, calibration, test = (views[name] for name in ("Training", "Calibration", "Evaluation"))
    forest = new_iforest().fit(fit)
    scaler = RobustScaler().fit(fit)
    extended = ExtendedForest(scaler.transform(fit))
    calibration_scores = {
        "Isolation Forest": -forest.score_samples(calibration),
        "Extended Isolation Forest": extended.score(scaler.transform(calibration))[0],
        baseline_name: np.abs(calibration[:, 0]),
    }
    cutoffs = {method: score_cutoff(scores) for method, scores in calibration_scores.items()}

    score = -forest.score_samples(test)
    extended_score, exiffi = extended.score(scaler.transform(test), explain=True)
    effects, tree_base, path_errors = explain_paths(forest, test)
    eval_rows = groups["Evaluation"].copy()
    eval_rows["anomaly_score"] = score
    eval_rows["extended_score"] = extended_score
    eval_rows["review"] = score > cutoffs["Isolation Forest"]
    eval_rows["extended_review"] = extended_score > cutoffs["Extended Isolation Forest"]
    eval_rows["tree_path_length"] = tree_base + effects.sum(axis=1)
    eval_rows["top_tree_shap"] = np.array(list(features.values()))[effects.argmin(axis=1)]

    return {
        "processed": processed, "groups": groups, "views": views,
        "forest": forest, "scaler": scaler, "extended": extended, "cutoffs": cutoffs,
        "calibration_scores": calibration_scores, "eval": eval_rows,
        "effects": effects, "exiffi": exiffi, "tree_base": tree_base, "path_errors": path_errors,
        "baseline_name": baseline_name, "features": features,
    }


def scenario_amount(row, x, result):
    """Score observed peer amounts as hypothetical inputs with all other features fixed."""
    groups = result["groups"]
    train = groups["Training"]
    candidates = train.loc[train.category_key.eq(row.category_key), "contract_amount"].to_numpy()
    scope = "Same business category"
    if len(candidates) < MIN_PEERS:
        return {"scope": scope, "points": [], "closest": None}
    amounts = np.unique(np.quantile(candidates, np.linspace(0.05, 0.95, 19), method="nearest"))
    trial = np.repeat(x[None, :], len(amounts), axis=0)
    trial[:, 0] = np.log10(amounts / row.peer_median_php)
    scores = -result["forest"].score_samples(trial)
    choices = [{"amount": float(a), "score": float(s)} for a, s in zip(amounts, scores)]
    cutoff = result["cutoffs"]["Isolation Forest"]
    passing = [
        p for p in choices
        if p["score"] <= cutoff and not np.isclose(p["amount"], row.contract_amount, rtol=0, atol=0.005)
    ]
    closest = None
    if bool(row.review) and passing:
        closest = min(passing, key=lambda p: abs(p["amount"] - row.contract_amount))
    return {"scope": scope, "points": choices, "closest": closest}


def stress_comparison(result, rounds):
    test = result["views"]["Evaluation"]
    rows = result["eval"]
    burst_count = 12
    changes = np.zeros_like(test)
    changes[:, 1] = 100 * burst_count / rows.prior_award_count
    changes[:, 2] = burst_count
    choices = [
        ("Amount ×0.25", np.arange(len(test)), np.array([np.log10(0.25), 0, 0, 0, 0])),
        ("Amount ×4", np.arange(len(test)), np.array([np.log10(4.0), 0, 0, 0, 0])),
        ("Vendor burst", np.flatnonzero(rows.recent_other_vendors.to_numpy() >= burst_count), changes),
    ]
    return compare_stress_scenarios(result, choices, rounds)


def compare_stress_scenarios(result, choices, rounds):
    """Compare synthetic cases with untouched records from the same eligible cohort."""
    forest, extended, scaler = (result[key] for key in ("forest", "extended", "scaler"))
    test = result["views"]["Evaluation"]
    rows = result["eval"]
    cutoffs = result["cutoffs"]
    baseline = result["baseline_name"]
    base_scores = {
        "Isolation Forest": rows.anomaly_score.to_numpy(),
        "Extended Isolation Forest": rows.extended_score.to_numpy(),
        baseline: np.abs(test[:, 0]),
    }
    rng = np.random.default_rng(73)
    scenario_metrics = []
    for name, cohort, change in choices:
        if len(cohort) < 20:
            continue
        eligible_changes = change[cohort] if change.ndim == 2 else change
        simulated = test[cohort] + eligible_changes
        subs = max(1, round(REVIEW_TAIL * len(cohort)))
        trial_draws = [rng.choice(len(cohort), size=subs, replace=False) for _ in range(rounds)]
        # Use all synthetic variants once per model. Each trial mixes only its selected rows.
        hypothetical_scores = {
            "Isolation Forest": -forest.score_samples(simulated),
            "Extended Isolation Forest": extended.score(scaler.transform(simulated))[0],
            baseline: np.abs(simulated[:, 0]),
        }
        for positions in trial_draws:
            mask = np.zeros(len(cohort), dtype=bool)
            mask[positions] = True
            for method, values in hypothetical_scores.items():
                base = base_scores[method][cohort]
                mixed = base.copy()
                mixed[mask] = values[mask]
                paired = mixed[mask] - base[mask]
                scenario_metrics.append({
                    "scenario": name,
                    "method": method,
                    "eligible_awards": len(cohort),
                    "substitutions_per_trial": subs,
                    "ap": average_precision_score(mask, mixed),
                    "recall": float(np.mean(mixed[mask] > cutoffs[method])),
                    "paired_win": float(np.mean((paired > 0) + 0.5 * (paired == 0))),
                })
    return pd.DataFrame(scenario_metrics).groupby(
        ["scenario", "method", "eligible_awards", "substitutions_per_trial"], sort=False
    ).mean().reset_index()


def evaluate(result, rounds=8, *, compare=stress_comparison, scenario_for=scenario_amount):
    forest = result["forest"]
    fit, test = (result["views"][key] for key in ("Training", "Evaluation"))
    rows = result["eval"]
    scenarios = compare(result, rounds)

    flags = rows.index[rows.review].to_numpy()
    faithfulness = {}
    stability = {}
    scenario_check = {}
    if len(flags) > 0:
        x = test[flags]
        effects = result["effects"][flags]
        diffi = local_diffi(forest, x)
        amount_rank = np.argmax(-effects, axis=1)
        diffi_rank = np.argmax(diffi, axis=1)
        imputed = np.repeat(x[:, None, :], x.shape[1], axis=1)
        for i in range(x.shape[1]):
            imputed[:, i, i] = np.median(fit[:, i])
        deltas = (-forest.score_samples(x))[:, None] + forest.score_samples(
            imputed.reshape(-1, x.shape[1])
        ).reshape(len(x), x.shape[1])

        def feature_probe(rank):
            top = deltas[np.arange(len(x)), rank]
            others = (deltas.sum(axis=1) - top) / (x.shape[1] - 1)
            return {
                "mean_top_drop": float(top.mean()),
                "mean_other_drop": float(others.mean()),
                "top_drop_positive": float(np.mean(top > 0)),
            }

        faithfulness = {"TreeSHAP": feature_probe(amount_rank), "Local-DIFFI": feature_probe(diffi_rank)}
        rank_agreement, diffi_agreement, score_rank = [], [], []
        for seed in range(rounds):
            alternate = new_iforest(seed).fit(fit)
            alternate_effect, _, _ = explain_paths(alternate, x)
            rank_agreement.append(np.mean(alternate_effect.argmin(axis=1) == amount_rank))
            diffi_agreement.append(np.mean(local_diffi(alternate, x).argmax(axis=1) == diffi_rank))
            score_rank.append(spearmanr(rows.anomaly_score, -alternate.score_samples(test)).statistic)
        stability = {
            "refits": rounds, "tree_shap_top_feature_agreement": float(np.mean(rank_agreement)),
            "local_diffi_top_feature_agreement": float(np.mean(diffi_agreement)),
            "score_rank_spearman": float(np.mean(score_rank)),
        }
        any_candidate, flips = 0, 0
        for i in flags:
            scenario = scenario_for(rows.iloc[i], test[i], result)
            any_candidate += bool(scenario["points"])
            flips += scenario["closest"] is not None
        scenario_check = {"review_rows": len(flags), "eligible": int(any_candidate), "flipped": int(flips)}

    return {
        "stress_trials_per_scenario": rounds,
        "scenarios": scenarios.to_dict(orient="records"),
        "faithfulness": faithfulness, "stability": stability, "amount_scenarios": scenario_check,
        "maximum_shap_error": float(result["path_errors"].max()),
        "n_flags": int(rows.review.sum()), "n_extended_flags": int(rows.extended_review.sum()),
        "n_both": int((rows.review & rows.extended_review).sum()),
    }
