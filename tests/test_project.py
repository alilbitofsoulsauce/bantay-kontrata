import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from integration import build_integrated_data, reconcile_awards
from screen import run_screen
from tender_screen import TENDER_FEATURES, run_tender_screen, tender_features

ROOT = Path(__file__).resolve().parents[1]


class LinkageTest(unittest.TestCase):
    def test_link_requires_one_to_one_title_match(self):
        day = "2024-06-14"
        archive = pd.DataFrame([
            ("A", day, 125.0, "Road repair (PR #123)", "", "Company A"),
            ("B", day, 125.0, "Other work", "", "Company A"),
            ("C", day, 175.0, "Two matching rows", "", "Company A"),
            ("D", day, 175.0, "Two matching rows", "", "Company A"),
        ], columns=["record_id", "award_date", "contract_amount", "notice_title", "award_title", "awardee_name"])
        official = pd.DataFrame([
            ("101", day, 125.0, "Road repair (PR123)", "", "COMPANY A"),
            ("102", day, 125.0, "Different title", "", "Company A"),
            ("103", day, 175.0, "Two matching rows", "", "Company A"),
        ], columns=[
            "award_reference", "award_date", "contract_amount_php", "notice_title", "award_title", "awardee",
        ])
        accepted, pending = reconcile_awards(archive, official)
        self.assertEqual(list(zip(accepted.record_id, accepted.official_award_reference)), [("A", "101")])
        self.assertEqual(set(pending.record_id), {"B", "C", "D"})
        self.assertEqual(len(pending), 3)


class TemporalContextTest(unittest.TestCase):
    def test_tender_history_excludes_same_day_and_award_outcomes(self):
        raw = pd.DataFrame({
            "published_date": ["2024-03-20", "2024-04-04", "2024-04-04", "2024-04-07"],
            "bid_reference": ["1", "2", "3", "4"],
            "procurement_mode": ["Public Bidding"] * 4,
            "classification": ["Goods"] * 4,
            "approved_budget_php": [10000, 100000, 10000, 10000],
            "notice_window_days": [5, 7, 11, 5],
            "observed_official_awards": [0, 1, 2, 0],
        })
        a = tender_features(raw)
        b = tender_features(raw.assign(observed_official_awards=102))
        self.assertEqual(a.loc[1:2, "budget_peer_median_php"].tolist(), [10000.0, 10000.0])
        self.assertEqual(a.loc[1:2, "window_peer_median_days"].tolist(), [5.0, 5.0])
        pd.testing.assert_frame_equal(a[list(TENDER_FEATURES)], b[list(TENDER_FEATURES)])


class SnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_integrated_data(ROOT / "data")

    def test_register_totals_and_keys(self):
        counts = self.data.counts
        self.assertEqual(counts["award_register_rows"], 12_300)
        self.assertEqual(counts["linked_awards"], 2_198)
        self.assertEqual(counts["unresolved_register_rows"], 326)
        self.assertEqual(counts["archive_awards"] + counts["official_awards"] - counts["linked_awards"], 12_300)
        self.assertTrue(self.data.award_register.record_id.is_unique)
        self.assertEqual(len(self.data.award_links[self.data.award_links.award_date.str.startswith("2025")]), 803)

    def test_multi_award_tender_and_flood_links(self):
        p = self.data.tenders.loc[self.data.tenders.bid_reference.eq("11424537")].iloc[0]
        self.assertEqual(p.observed_official_awards, 2)
        self.assertAlmostEqual(p.observed_official_award_total_php, 1_023_136.0)
        self.assertAlmostEqual(p.approved_budget_php, 1_026_667.0)
        projects = pd.read_csv(ROOT / "data" / "tagum_dpwh.csv", dtype={"dpwh_contract_id": str})
        self.assertTrue(self.data.flood_links.dpwh_contract_id.isin(projects.dpwh_contract_id).all())
        self.assertEqual(len(self.data.flood_links), 10)

    def test_both_forests_and_shap_with_frozen_periods(self):
        models = [run_screen(ROOT / "data"), run_tender_screen(self.data.tenders)]
        expected = [(1_317, 106, 7_807), (516, 48, 1_101)]
        for result, (held_out, flags, training) in zip(models, expected):
            with self.subTest(rows=held_out):
                rows = result["eval"]
                cutoff = result["cutoffs"]["Isolation Forest"]
                self.assertEqual(len(result["groups"]["Training"]), training)
                self.assertEqual(len(rows), held_out)
                self.assertEqual(rows.review.sum(), flags)
                np.testing.assert_array_equal(rows.review, rows.anomaly_score > cutoff)
                self.assertTrue(np.isfinite(rows.extended_score).all())
                self.assertLess(result["path_errors"].max(), 1e-8)


if __name__ == "__main__":
    unittest.main()
