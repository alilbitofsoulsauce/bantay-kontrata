"""Unit and integration tests for the Four-Pillar XAI & Procurement Intelligence Engine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from screen import ExtendedForest, explain_paths, path_adjustment, score_cutoff
from xai_engine import (
    DPWH_FEATURES,
    LiveXAIEngine,
    build_procurement_intelligence,
    build_shap_waterfall,
    compute_2d_decision_surface,
    compute_counterfactual_recourse,
    compute_imbalance_diffi,
    compute_xai_consensus,
    find_nearest_peers,
    statutory_procurement_checks,
)

ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_screen(kind: str = "award", zero_iqr_col: bool = False) -> dict:
    """Build a compact deterministic screen fixture for fast unit testing."""
    rng = np.random.default_rng(123)
    n_train, n_cal, n_eval = 180, 60, 20
    x_train = rng.normal(loc=0.0, scale=0.35, size=(n_train, 5))
    x_cal = rng.normal(loc=0.0, scale=0.35, size=(n_cal, 5))
    x_eval = rng.normal(loc=0.0, scale=0.35, size=(n_eval, 5))

    if kind == "award":
        x_train[:, 1] = np.abs(x_train[:, 1]) * 4.0
        x_cal[:, 1] = np.abs(x_cal[:, 1]) * 4.0
        x_eval[:, 1] = np.abs(x_eval[:, 1]) * 4.0
        x_train[:, 2] = np.zeros(n_train) if zero_iqr_col else rng.integers(0, 3, size=n_train)
        x_cal[:, 2] = rng.integers(0, 3, size=n_cal)
        x_eval[:, 2] = rng.integers(0, 3, size=n_eval)
        x_train[:, 3] = rng.integers(5, 85, size=n_train)
        x_cal[:, 3] = rng.integers(5, 85, size=n_cal)
        x_eval[:, 3] = rng.integers(5, 85, size=n_eval)
        x_train[:, 4] = rng.integers(20, 200, size=n_train)
        x_cal[:, 4] = rng.integers(20, 200, size=n_cal)
        x_eval[:, 4] = rng.integers(20, 200, size=n_eval)
        # Plant clear mutable anomaly at row 0 and locked-context anomaly at row 1
        x_eval[0] = np.array([1.8, 22.0, 9.0, 40.0, 80.0])
        x_eval[1] = np.array([0.0, 1.0, 0.0, 950.0, 9500.0])
        features = {
            "amount_log_ratio": "Amount vs. recent peers",
            "vendor_share_pct": "Vendor's prior award share",
            "vendor_category_30d": "Recent same-category vendor awards",
            "days_to_quarter_end": "Days to quarter end",
            "category_365d": "Prior awards in category",
        }
    elif kind == "tender":
        x_train[:, 2] = rng.uniform(10.0, 60.0, size=n_train)
        x_cal[:, 2] = rng.uniform(10.0, 60.0, size=n_cal)
        x_eval[:, 2] = rng.uniform(10.0, 60.0, size=n_eval)
        x_train[:, 3] = rng.uniform(20.0, 80.0, size=n_train)
        x_cal[:, 3] = rng.uniform(20.0, 80.0, size=n_cal)
        x_eval[:, 3] = rng.uniform(20.0, 80.0, size=n_eval)
        x_train[:, 4] = rng.integers(5, 85, size=n_train)
        x_cal[:, 4] = rng.integers(5, 85, size=n_cal)
        x_eval[:, 4] = rng.integers(5, 85, size=n_eval)
        x_eval[0] = np.array([1.7, -0.8, 1.5, 50.0, 35.0])
        features = {
            "budget_log_ratio": "Budget vs. recent peers",
            "window_log_ratio": "Bid window vs. recent peers",
            "prior_mode_share": "Prior tenders in this procurement mode",
            "prior_class_share": "Prior tenders in this classification",
            "days_to_quarter_end": "Days to quarter end",
        }
    else:
        x_train[:, 0] = rng.uniform(0.96, 1.0, size=n_train)
        x_cal[:, 0] = rng.uniform(0.96, 1.0, size=n_cal)
        x_eval[:, 0] = rng.uniform(0.96, 1.0, size=n_eval)
        x_train[:, 2] = np.zeros(n_train) if zero_iqr_col else rng.normal(0.0, 3.0, size=n_train)
        x_cal[:, 2] = rng.normal(0.0, 3.0, size=n_cal)
        x_eval[:, 2] = rng.normal(0.0, 3.0, size=n_eval)
        x_train[:, 3] = rng.integers(60, 240, size=n_train)
        x_cal[:, 3] = rng.integers(60, 240, size=n_cal)
        x_eval[:, 3] = rng.integers(60, 240, size=n_eval)
        x_train[:, 4] = rng.integers(3, 25, size=n_train)
        x_cal[:, 4] = rng.integers(3, 25, size=n_cal)
        x_eval[:, 4] = rng.integers(3, 25, size=n_eval)
        x_eval[0] = np.array([1.15, 0.1, 140.0, 620.0, 10.0])
        features = dict(DPWH_FEATURES)

    forest = IsolationForest(
        n_estimators=64,
        max_samples=min(256, len(x_train)),
        max_features=1.0,
        contamination="auto",
        random_state=42,
        n_jobs=1,
    ).fit(x_train)
    scaler = RobustScaler().fit(x_train)
    extended = ExtendedForest(scaler.transform(x_train), seed=42, trees=64)
    cal_if = -forest.score_samples(x_cal)
    cal_eif = extended.score(scaler.transform(x_cal))[0]
    cutoffs = {
        "Isolation Forest": score_cutoff(cal_if),
        "Extended Isolation Forest": score_cutoff(cal_eif),
    }
    return {
        "views": {"Training": x_train, "Calibration": x_cal, "Evaluation": x_eval},
        "groups": {
            "Training": pd.DataFrame(x_train, columns=list(features)),
            "Calibration": pd.DataFrame(x_cal, columns=list(features)),
            "Evaluation": pd.DataFrame(x_eval, columns=list(features)),
        },
        "forest": forest,
        "scaler": scaler,
        "extended": extended,
        "cutoffs": cutoffs,
        "features": features,
    }


class ImbalanceDiffiTests(unittest.TestCase):
    def test_identifies_planted_anomaly_feature_globally_and_locally(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 0.5, size=(220, 5))
        # Plant strong anomalies purely along feature index 0
        x[-12:, 0] = rng.uniform(6.0, 9.0, size=12)
        forest = IsolationForest(
            n_estimators=80, max_samples=min(256, 180), random_state=42, n_jobs=1
        ).fit(x[:180])

        global_diffi, local_imb = compute_imbalance_diffi(forest, x)
        self.assertEqual(global_diffi.shape, (5,))
        self.assertEqual(local_imb.shape, (220, 5))
        self.assertTrue(np.isfinite(global_diffi).all())
        self.assertTrue(np.isfinite(local_imb).all())
        self.assertTrue((local_imb >= 0).all())
        self.assertEqual(int(np.argmax(global_diffi)), 0)
        self.assertGreater(float(local_imb[-12:, 0].mean()), float(local_imb[:180, 0].mean()))

    def test_empty_input_and_degenerate_masks_do_not_divide_by_zero(self) -> None:
        rng = np.random.default_rng(11)
        x_fit = rng.normal(size=(64, 4))
        sub_forest = IsolationForest(
            n_estimators=32, max_samples=32, max_features=0.5, random_state=11, n_jobs=1
        ).fit(x_fit)

        empty_g, empty_l = compute_imbalance_diffi(sub_forest, np.empty((0, 4)))
        self.assertEqual(empty_g.shape, (4,))
        self.assertEqual(empty_l.shape, (0, 4))
        self.assertTrue(np.all(empty_g == 0.0))

        all_false_mask = np.zeros(len(x_fit), dtype=bool)
        g_false, l_false = compute_imbalance_diffi(sub_forest, x_fit, outlier_mask=all_false_mask)
        self.assertTrue(np.isfinite(g_false).all())
        self.assertTrue(np.all(g_false == 0.0))
        self.assertEqual(l_false.shape, (64, 4))


class ShapWaterfallTests(unittest.TestCase):
    def test_exact_path_to_score_mapping_and_ordering(self) -> None:
        screen_res = _make_synthetic_screen("award")
        forest = screen_res["forest"]
        x_row = screen_res["views"]["Evaluation"][0:1]
        effects, base_path, _ = explain_paths(forest, x_row)
        expected_score = float(-forest.score_samples(x_row)[0])
        cutoff = screen_res["cutoffs"]["Isolation Forest"]
        names = list(screen_res["features"].values())
        contexts = [f"ctx-{i}" for i in range(5)]

        wf = build_shap_waterfall(
            effects[0],
            base_path=base_path,
            sample_size=int(forest.max_samples_),
            feature_names=names,
            cutoff=cutoff,
            input_contexts=contexts,
        )
        self.assertAlmostEqual(wf["final_score"], expected_score, places=5)
        self.assertAlmostEqual(wf["path_c"], float(path_adjustment(forest.max_samples_)), places=5)
        self.assertEqual(wf["above_cutoff"], expected_score > cutoff)
        self.assertEqual(len(wf["steps"]), 5)

        splits = [step["shap_splits"] for step in wf["steps"]]
        self.assertEqual(splits, sorted(splits))
        self.assertAlmostEqual(wf["steps"][0]["path_before"], wf["base_path"], places=5)
        self.assertAlmostEqual(wf["steps"][-1]["path_after"], wf["final_path"], places=5)
        for idx in range(1, len(wf["steps"])):
            self.assertAlmostEqual(wf["steps"][idx]["path_before"], wf["steps"][idx - 1]["path_after"], places=5)
            self.assertAlmostEqual(wf["steps"][idx]["score_before"], wf["steps"][idx - 1]["score_after"], places=5)
        for step in wf["steps"]:
            self.assertEqual(step["context"], f"ctx-{step['feature_index']}")

    def test_missing_contexts_and_sample_size_two_edge_cases(self) -> None:
        wf = build_shap_waterfall(
            shap_values=[0.0, 0.0, 0.0],
            base_path=1.0,
            sample_size=2,
            feature_names=["A", "B", "C"],
            cutoff=0.6,
            input_contexts=None,
        )
        self.assertEqual(wf["path_c"], 1.0)
        self.assertEqual(wf["base_score"], 0.5)
        self.assertEqual(wf["final_score"], 0.5)
        self.assertFalse(wf["above_cutoff"])
        self.assertTrue(all(step["context"] == "" and step["score_delta"] == 0.0 for step in wf["steps"]))


class XAIConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = ["Amount", "Vendor Share", "30d Burst", "Quarter End", "Category Count"]

    def test_all_five_consensus_tiers(self) -> None:
        normal = compute_xai_consensus(
            [-0.1, 0.2, 0.1, 0.0, 0.1],
            [0.3, 0.1, 0.0, 0.0, 0.0],
            [0.4, 0.1, 0.1, 0.1, 0.1],
            self.names,
            if_flag=False,
            eif_flag=False,
        )
        self.assertEqual(normal["tier"], "normal")
        self.assertEqual(normal["label"], "Within normal peer range")
        self.assertFalse(normal["both_forests_flag"])

        oblique_only = compute_xai_consensus(
            [-0.1, -0.4, 0.1, 0.0, 0.1],
            [0.1, 0.2, 0.5, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.8, 0.1],
            self.names,
            if_flag=False,
            eif_flag=True,
        )
        self.assertEqual(oblique_only["tier"], "oblique_only")
        self.assertEqual(oblique_only["label"], "EIF oblique interaction flag")
        self.assertFalse(oblique_only["both_forests_flag"])

        high = compute_xai_consensus(
            [-1.8, -0.2, 0.1, 0.0, 0.2],
            [0.9, 0.2, 0.1, 0.0, 0.0],
            [0.7, 0.2, 0.1, 0.1, 0.1],
            self.names,
            if_flag=True,
            eif_flag=True,
        )
        self.assertEqual(high["tier"], "high_consensus")
        self.assertEqual(high["agreement_count"], 3)
        self.assertEqual(high["top_index"], 0)
        self.assertEqual(high["top_feature"], "Amount")
        self.assertTrue(high["both_forests_flag"])

        moderate = compute_xai_consensus(
            [-1.5, -0.2, 0.1, 0.0, 0.1],
            [0.1, 0.2, 0.8, 0.0, 0.0],
            [0.1, 0.2, 0.9, 0.1, 0.1],
            self.names,
            if_flag=True,
            eif_flag=False,
        )
        self.assertEqual(moderate["tier"], "moderate_consensus")
        self.assertEqual(moderate["agreement_count"], 2)
        self.assertEqual(moderate["top_index"], 2)
        self.assertEqual(moderate["top_feature"], "30d Burst")

        divergent = compute_xai_consensus(
            [-1.5, 0.0, 0.1, 0.0, 0.1],
            [0.1, 0.9, 0.2, 0.0, 0.0],
            [0.1, 0.2, 0.8, 0.1, 0.1],
            self.names,
            if_flag=True,
            eif_flag=True,
        )
        self.assertEqual(divergent["tier"], "oblique_interaction")
        self.assertEqual(divergent["agreement_count"], 1)
        self.assertTrue(len(divergent["summary"]) > 20)


class StatutoryChecksTests(unittest.TestCase):
    def test_triggers_ra12009_and_ra9184_procurement_checks(self) -> None:
        award_row = {
            "procurement_mode": "Negotiated Procurement - Small Value Procurement (Sec. 53.9)",
            "contract_amount": 1_950_000.0,
            "approved_budget_php": 1_955_000.0,
            "notice_window_days": 2,
            "vendor_category_30d": 4,
            "vendor_share_pct": 8.5,
            "days_to_quarter_end": 3,
        }
        checks = statutory_procurement_checks("award", award_row)
        codes = {c["code"] for c in checks}
        self.assertEqual(
            codes,
            {
                "ra12009_svp_ceiling",
                "compressed_bid_window",
                "tight_abc_margin",
                "vendor_30d_burst",
                "vendor_concentration",
                "quarter_end_rush",
            },
        )
        for item in checks:
            self.assertIn(item["severity"], {"review", "info"})
            self.assertTrue(item["title"] and item["detail"] and item["document_to_check"])

        legacy_svp_tender = {
            "procurement_mode": "Small Value Procurement",
            "approved_budget_php": 920_000.0,
            "notice_window_days": 5,
            "days_to_quarter_end": 25,
        }
        t_codes = {c["code"] for c in statutory_procurement_checks("tender", legacy_svp_tender)}
        self.assertIn("ra12009_svp_ceiling", t_codes)
        self.assertNotIn("compressed_bid_window", t_codes)

        pub_bid_compressed = {
            "procurement_mode": "Public Bidding",
            "approved_budget_php": 5_000_000.0,
            "notice_window_days": 6,
            "days_to_quarter_end": 40,
        }
        self.assertIn(
            "compressed_bid_window",
            {c["code"] for c in statutory_procurement_checks("tender", pub_bid_compressed)},
        )

        dpwh_delayed = {
            "abc_php": 10_000_000.0,
            "contract_cost_php": 9_999_000.0,
            "variance_days": 45,
        }
        d_codes = {c["code"] for c in statutory_procurement_checks("dpwh", dpwh_delayed)}
        self.assertIn("dpwh_schedule_delay", d_codes)
        self.assertIn("tight_abc_margin", d_codes)

    def test_clean_and_null_rows_produce_no_false_positives(self) -> None:
        clean_tender = {
            "procurement_mode": "Public Bidding",
            "approved_budget_php": 3_000_000.0,
            "notice_window_days": 21,
            "days_to_quarter_end": 45,
        }
        self.assertEqual(statutory_procurement_checks("tender", clean_tender), [])

        null_dpwh = {
            "abc_php": 5_000_000.0,
            "contract_cost_php": None,
            "cost_abc_ratio": np.nan,
            "variance_days": None,
        }
        self.assertEqual(statutory_procurement_checks("dpwh", null_dpwh), [])


class CounterfactualRecourseTests(unittest.TestCase):
    def test_recourse_across_award_tender_and_dpwh(self) -> None:
        configs = [
            (
                "award",
                {"contract_amount": 6_300_000.0, "peer_median_php": 100_000.0},
            ),
            (
                "tender",
                {
                    "approved_budget_php": 5_000_000.0,
                    "budget_peer_median_php": 100_000.0,
                    "notice_window_days": 2,
                    "window_peer_median_days": 14.0,
                },
            ),
            (
                "dpwh",
                {"abc_php": 10_000_000.0, "abc_peer_median_php": 8_000_000.0},
            ),
        ]
        for kind, row in configs:
            with self.subTest(kind=kind):
                res = _make_synthetic_screen(kind)
                x_anom = res["views"]["Evaluation"][0]
                t0 = time.perf_counter()
                cf = compute_counterfactual_recourse(kind, row, x_anom, res)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertTrue(cf["valid"])
                self.assertGreaterEqual(cf["sparsity"], 1)
                self.assertLessEqual(cf["sparsity"], 3)
                self.assertLessEqual(cf["recourse_score"], cf["cutoff"])
                self.assertEqual(len(cf["changes"]), cf["sparsity"])
                self.assertLess(elapsed_ms, 80.0)
                json.dumps(cf, allow_nan=False)

    def test_below_cutoff_and_locked_immutable_outlier_edge_cases(self) -> None:
        res = _make_synthetic_screen("award", zero_iqr_col=True)
        # Normal point below cutoff
        x_normal = np.median(res["views"]["Training"], axis=0)
        cf_ok = compute_counterfactual_recourse(
            "award", {"contract_amount": 100_000.0, "peer_median_php": 100_000.0}, x_normal, res
        )
        self.assertTrue(cf_ok["valid"])
        self.assertEqual(cf_ok["sparsity"], 0)
        self.assertEqual(cf_ok["changes"], [])

        # Extreme locked context features [3, 4] that keep score high or test zero-IQR safety
        x_locked = res["views"]["Evaluation"][1]
        cf_locked = compute_counterfactual_recourse(
            "award", {"contract_amount": 100_000.0, "peer_median_php": 100_000.0}, x_locked, res
        )
        self.assertIn("valid", cf_locked)
        self.assertIn("narrative", cf_locked)
        json.dumps(cf_locked, allow_nan=False)


class DecisionSurfaceAndPeersTests(unittest.TestCase):
    def test_2d_decision_surface_contract_and_validation(self) -> None:
        res = _make_synthetic_screen("tender")
        surf = compute_2d_decision_surface(res, fx=0, fy=1, grid_size=12)
        self.assertEqual(surf["fx"], 0)
        self.assertEqual(surf["fy"], 1)
        self.assertEqual(len(surf["x_grid"]), 12)
        self.assertEqual(len(surf["y_grid"]), 12)
        self.assertEqual(len(surf["if_scores"]), 12)
        self.assertEqual(len(surf["if_scores"][0]), 12)
        self.assertEqual(len(surf["eif_scores"]), 12)
        self.assertGreaterEqual(surf["exiffi_pair_share"], 0.0)
        self.assertLessEqual(surf["exiffi_pair_share"], 1.0)
        json.dumps(surf, allow_nan=False)

        with self.assertRaises(ValueError):
            compute_2d_decision_surface(res, fx=1, fy=1)
        with self.assertRaises(ValueError):
            compute_2d_decision_surface(res, fx=-1, fy=2)
        with self.assertRaises(ValueError):
            compute_2d_decision_surface(res, fx=0, fy=1, grid_size=1)

    def test_find_nearest_peers_same_group_and_fallback(self) -> None:
        rows = pd.DataFrame({
            "id": ["R1", "R2", "R3", "R4", "R5"],
            "cat": ["Goods", "Goods", "Civil Works", "Civil Works", "Civil Works"],
        })
        x = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [5.0, 5.0],
            [8.0, 8.0],
        ])
        # R1 is Goods; only R2 is also Goods, so fallback fills remaining 2 from closest overall
        peers_fallback = find_nearest_peers(rows, x, 0, "cat", "Goods", "id", exclude_id="R1", k=3)
        self.assertEqual(len(peers_fallback), 3)
        self.assertEqual(peers_fallback[0], "R2")
        self.assertNotIn("R1", peers_fallback)

        # Civil Works has 3 rows; excluding R3 leaves 2, target_vec supported
        peers_vec = find_nearest_peers(
            rows, x, None, "cat", "Civil Works", "id", k=2, target_vec=np.array([0.25, 0.25])
        )
        self.assertEqual(peers_vec, ["R3", "R4"])

        # Empty DataFrame edge case
        empty_peers = find_nearest_peers(
            rows.iloc[:0], np.empty((0, 2)), None, "cat", "Goods", "id", k=4, target_vec=np.zeros(2)
        )
        self.assertEqual(empty_peers, [])


class TagumSnapshotAndLiveEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = ROOT / "data"
        cls.engine = LiveXAIEngine(cls.data_dir)
        cls.engine.ensure_loaded()

    def test_dpwh_screen_scores_all_336_projects_and_23_barangays(self) -> None:
        dpwh = self.engine.dpwh_screen
        self.assertEqual(len(dpwh["eval"]), 336)
        self.assertEqual(len(dpwh["records"]), 336)
        self.assertEqual(list(dpwh["features"].keys()), list(DPWH_FEATURES.keys()))
        self.assertEqual(len(dpwh["barangay_summary"]), 23)
        self.assertEqual(len(dpwh["surfaces"]), 2)

        flood_linked = sum(1 for r in dpwh["records"] if r.get("flood_map_project_id"))
        self.assertEqual(flood_linked, 10)
        flagged = [r for r in dpwh["records"] if r["review"]]
        self.assertGreater(len(flagged), 0)
        for sample in flagged[:3]:
            self.assertIn("waterfall", sample)
            self.assertIn("consensus", sample)
            self.assertIn("statutory_checks", sample)
            self.assertIn("peer_ids", sample)
            self.assertIsNotNone(sample.get("recourse"))

        # Verify strict JSON serializability (allow_nan=False)
        serializable_view = {
            "records": dpwh["records"],
            "barangay_summary": dpwh["barangay_summary"],
            "surfaces": dpwh["surfaces"],
            "global_xai": dpwh["global_xai"],
            "cutoffs": dpwh["cutoffs"],
        }
        json.dumps(serializable_view, allow_nan=False)

    def test_procurement_intelligence_vendors_clusters_and_overlaps(self) -> None:
        intel = build_procurement_intelligence(
            self.engine.integrated_data, self.engine.award_eval_with_links
        )
        self.assertLessEqual(len(intel["vendors"]), 60)
        self.assertGreater(len(intel["vendors"]), 20)
        # Check sorting: (flags_2025 desc, total_amount_php desc)
        pairs = [(v["flags_2025"], v["total_amount_php"]) for v in intel["vendors"]]
        self.assertEqual(pairs, sorted(pairs, reverse=True))

        self.assertGreater(len(intel["split_clusters"]), 0)
        self.assertLessEqual(len(intel["split_clusters"]), 40)
        for cluster in intel["split_clusters"]:
            self.assertGreaterEqual(cluster["award_count"], 3)
            self.assertGreaterEqual(cluster["total_amount_php"], 500_000.0)
            self.assertTrue(len(cluster["record_ids"]) >= 3)

        self.assertGreater(len(intel["overlaps"]), 0)
        self.assertLessEqual(len(intel["overlaps"]), 120)
        self.assertTrue(any(item["in_2025_queue"] for item in intel["overlaps"]))
        self.assertIn("summary", intel)
        json.dumps(intel, allow_nan=False)

    def test_live_engine_meta_presets_and_vendor_search(self) -> None:
        meta = self.engine.get_meta()
        self.assertGreater(len(meta["categories"]), 10)
        self.assertGreater(len(meta["modes"]), 3)
        self.assertGreater(len(meta["classifications"]), 2)
        self.assertEqual(len(meta["presets"]), 4)
        kinds = [p["kind"] for p in meta["presets"]]
        self.assertEqual(kinds.count("tender"), 2)
        self.assertEqual(kinds.count("award"), 2)
        json.dumps(meta, allow_nan=False)

        # Every preset must execute cleanly through its live scorer
        for preset in meta["presets"]:
            if preset["kind"] == "tender":
                scored = self.engine.score_custom_tender(preset["payload"])
            else:
                scored = self.engine.score_custom_award(preset["payload"])
            self.assertIn("waterfall", scored)
            self.assertIn("consensus", scored)
            self.assertIn("recourse", scored)
            self.assertGreater(len(scored["peers"]), 0)
            json.dumps(scored, allow_nan=False)

        top_vendors = self.engine.search_vendors("", limit=10)
        self.assertEqual(len(top_vendors), 10)
        filtered = self.engine.search_vendors("construction", limit=5)
        self.assertGreater(len(filtered), 0)
        self.assertTrue(all("construction" in v["awardee_name"].casefold() for v in filtered))
        self.assertEqual(self.engine.search_vendors("zzzz-nonexistent-vendor-9999"), [])

    def test_live_custom_tender_award_validation_and_whatif_consistency(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.score_custom_tender({
                "notice_title": "Bad budget",
                "approved_budget_php": -100.0,
                "published_date": "2026-03-01",
                "notice_window_days": 7,
                "procurement_mode": "Public Bidding",
                "classification": "Goods",
            })
        with self.assertRaises(ValueError):
            self.engine.score_custom_tender({
                "notice_title": "Bad window",
                "approved_budget_php": 500_000.0,
                "published_date": "2026-03-10",
                "closing_date": "2026-03-05",
                "procurement_mode": "Public Bidding",
                "classification": "Goods",
            })
        with self.assertRaises(ValueError):
            self.engine.score_custom_award({
                "award_title": "Zero amount",
                "awardee_name": "Test Supplier",
                "category": "General Merchandise",
                "contract_amount": 0.0,
                "award_date": "2025-06-01",
            })

        # Pick a flagged 2025 award, get its counterfactual recourse, and verify simulate_whatif agrees!
        eval_df = self.engine.award_eval_with_links
        flagged_idx = int(eval_df.index[eval_df["review"]][0])
        flagged_row = eval_df.iloc[flagged_idx].to_dict()
        x_vec = self.engine.award_screen["views"]["Evaluation"][flagged_idx]
        cf = compute_counterfactual_recourse("award", flagged_row, x_vec, self.engine.award_screen)
        if cf["valid"] and cf["changes"]:
            overrides = {c["key"]: c["to_val"] for c in cf["changes"]}
            sim = self.engine.simulate_whatif({
                "kind": "award",
                "record_id": str(flagged_row["record_id"]),
                "overrides": overrides,
            })
            self.assertFalse(sim["review"])
            self.assertLessEqual(sim["anomaly_score"], sim["cutoff"] + 1e-5)
            json.dumps(sim, allow_nan=False)

        # Also test tender & dpwh simulate_whatif + 2D surface + thread safety
        sim_dpwh = self.engine.simulate_whatif({
            "kind": "dpwh",
            "record_id": None,
            "overrides": {"variance_days": 90, "cost_abc_ratio": 0.9999},
        })
        self.assertIn("waterfall", sim_dpwh)
        json.dumps(sim_dpwh, allow_nan=False)

        surf = self.engine.get_2d_surface("award", 0, 1, grid_size=10)
        self.assertEqual(len(surf["if_scores"]), 10)

        def _worker(idx: int) -> float:
            out = self.engine.simulate_whatif({
                "kind": "tender",
                "record_id": None,
                "overrides": {"notice_window_days": 3 + idx},
            })
            return float(out["anomaly_score"])

        with ThreadPoolExecutor(max_workers=4) as pool:
            scores = list(pool.map(_worker, range(6)))
        self.assertEqual(len(scores), 6)
        self.assertTrue(all(np.isfinite(s) for s in scores))


if __name__ == "__main__":
    unittest.main()
