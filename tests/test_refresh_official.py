import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from integration import BUYER, ROOT
from refresh_official import COLUMNS, write_snapshot


class RefreshSnapshotTests(unittest.TestCase):
    def make_release(self, path, **extra):
        fields = {
            "Procuring Entity (PE)": BUYER,
            "Bid Reference No.": "99990000",
            "Notice Title": "Refresh test fixture",
            "Classification": "Goods",
            "Procurement Mode": "Public Bidding",
            "Published Date": "01/10/2026",
            "Closing Date": "19/10/2026",
            "Approved Budget of the Contract": "32,000.50",
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow(fields)
            writer.writerow({**fields, **extra})

    def test_new_notice_stages_with_source_trace(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            intake = root / "quarter.csv"
            output = root / "new_snapshot"
            self.make_release(intake, **{"Procuring Entity (PE)": "TAGUM WATER DISTRICT"})
            summary = write_snapshot(ROOT / "data", intake, "2026Q4", "https://philgeps.gov.ph/", output)
            self.assertEqual(summary["official_tenders"], 3396)
            self.assertEqual(summary["archive_awards"], 11305)
            self.assertEqual(summary["included_tagum_source_rows"], 5843)
            self.assertEqual(summary["latest_tender_publication_date"], "2026-10-01")
            tender_file = output / "official/tagum_philgeps_tenders.csv"
            tender = pd.read_csv(tender_file, dtype=str).set_index("bid_reference").loc["99990000"]
            self.assertEqual(tender.approved_budget_php, "32000.50")
            self.assertEqual(tender.source_releases, "2026Q4")
            provenance = json.loads((output / "official/provenance.json").read_text())
            self.assertEqual(provenance["files"][-1]["count"]["tagum_line_item_rows"], 1)

    def test_conflicted_notice_leaves_no_new_snapshot(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            intake = root / "conflict.csv"
            output = root / "failed_snapshot"
            self.make_release(intake, **{"Closing Date": "20/10/2026"})
            with self.assertRaisesRegex(ValueError, "conflicting scoring fields.*99990000"):
                write_snapshot(ROOT / "data", intake, "2026Q4", "https://philgeps.gov.ph/", output)
            self.assertFalse(output.exists())
