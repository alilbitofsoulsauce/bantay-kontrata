from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "review" / "Bantay_Kontrata_Dashboard.html"
SCHEMA = "bantay-kontrata-tagum-reviews-v1"


def embedded_data(html):
    payload = html.split('<script id="dataset" type="application/json">', 1)[1]
    return payload.split("</script>", 1)[0]


@unittest.skipUnless(find_spec("flask"), "Install requirements-app.txt for app tests")
class ReviewAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.offline_html = DASHBOARD.read_text(encoding="utf-8")
        cls.payload = embedded_data(cls.offline_html)
        data = json.loads(cls.payload)
        cls.snapshot = data["snapshot_id"]
        cls.award_key = "award:" + data["award"]["records"][0]["record_id"]
        cls.tender_key = "tender:" + data["tender"]["records"][0]["bid_reference"]

    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.database = Path(temp.name) / "reviews.sqlite3"
        self.client = self.make_client()
        self.start = self.read_reviews()

    def make_client(self):
        from app import create_app

        return create_app(db_path=self.database, dashboard_path=DASHBOARD).test_client()

    def read_reviews(self, client=None):
        response = (client or self.client).get("/api/reviews", query_string={"snapshot_id": self.snapshot})
        self.assertEqual(response.status_code, 200)
        return response.get_json()

    def write_reviews(self, entries, *, source="edit", headers=None, **overrides):
        return self.client.put(
            "/api/reviews",
            headers=headers,
            json={**self.start, "source": source, "entries": entries, **overrides},
        )

    def reviewed_case(self, key=None, **changes):
        return {
            "key": key or self.award_key,
            "status": "reviewed",
            "reviewer": "QA",
            "evidence_url": "",
            "evidence_ref": "Test evidence reference",
            "notes": "Test verification note",
            "updated_at": "2026-11-17T01:00:00.000Z",
            **changes,
        }

    def test_served_dashboard_enables_server_storage_without_changing_snapshot(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        served = response.get_data(as_text=True)
        self.assertIn('data-review-store="server"', served)
        self.assertNotIn('data-review-store="server"', self.offline_html)
        self.assertEqual(embedded_data(served), self.payload)
        self.assertEqual(self.start["schema"], SCHEMA)
        self.assertEqual(self.start["snapshot_id"], self.snapshot)
        self.assertEqual(self.start["entries"], [])

    def test_review_upserts_persist_across_app_instances(self):
        award = self.reviewed_case()
        tender = self.reviewed_case(self.tender_key)
        first = self.write_reviews([award], headers={"Origin": "http://localhost"})
        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
        after_first = self.read_reviews()
        self.assertGreater(after_first["revision"], self.start["revision"])
        second = self.write_reviews([tender], revision=after_first["revision"])
        self.assertEqual(second.status_code, 200, second.get_data(as_text=True))
        after_second = self.read_reviews()
        updated = self.write_reviews(
            [{**award, "notes": "Updated verification note"}],
            revision=after_second["revision"],
        )
        self.assertEqual(updated.status_code, 200, updated.get_data(as_text=True))

        restarted = self.read_reviews(self.make_client())
        entries = {entry["key"]: entry for entry in restarted["entries"]}
        self.assertEqual(set(entries), {self.award_key, self.tender_key})
        self.assertEqual(entries[self.award_key]["notes"], "Updated verification note")
        self.assertEqual(entries[self.tender_key]["evidence_ref"], tender["evidence_ref"])
        self.assertEqual(restarted, self.read_reviews())

    def test_stale_and_wrong_snapshot_requests_do_not_change_reviews(self):
        wrong_snapshot = ("0" if self.snapshot[0] != "0" else "1") + self.snapshot[1:]
        wrong_get = self.client.get("/api/reviews", query_string={"snapshot_id": wrong_snapshot})
        self.assertEqual(wrong_get.status_code, 409)
        wrong_put = self.write_reviews([self.reviewed_case()], snapshot_id=wrong_snapshot)
        self.assertEqual(wrong_put.status_code, 409)
        self.assertEqual(self.read_reviews(), self.start)

        accepted = self.write_reviews([self.reviewed_case()])
        self.assertEqual(accepted.status_code, 200)
        current = self.read_reviews()
        stale = self.write_reviews([self.reviewed_case(self.tender_key)])
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(self.read_reviews(), current)

    def test_invalid_edits_are_rejected_but_older_backups_still_import(self):
        entry = self.reviewed_case()
        for source, invalid in (
            ("edit", [self.reviewed_case("tender:0")]),
            ("edit", [self.reviewed_case(status="unknown")]),
            ("edit", [self.reviewed_case(evidence_url="http://example.invalid/evidence")]),
            ("backup", [entry, entry]),
            ("backup", [entry, self.reviewed_case("tender:0")]),
            ("edit", [self.reviewed_case(evidence_ref="")]),
        ):
            with self.subTest(source=source, entries=invalid):
                response = self.write_reviews(invalid, source=source)
                self.assertIn(response.status_code, (400, 422))
                self.assertEqual(self.read_reviews(), self.start)

        legacy = self.reviewed_case(notes="")
        legacy.pop("evidence_ref")
        response = self.write_reviews([legacy], source="backup")
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        imported, = self.read_reviews()["entries"]
        self.assertEqual(imported["key"], self.award_key)
        self.assertEqual(imported["status"], "reviewed")
        self.assertEqual(imported["evidence_ref"], "")

    def test_wrong_hosts_and_cross_origin_writes_are_denied(self):
        entry = self.reviewed_case()
        for headers in ({"Host": "example.invalid"}, {"Origin": "https://example.invalid"}):
            with self.subTest(headers=headers):
                response = self.write_reviews([entry], headers=headers)
                self.assertIn(response.status_code, (400, 403))
        self.assertEqual(self.read_reviews(), self.start)


if __name__ == "__main__":
    unittest.main()
