"""Tests for cloud deployment security, audit history, and Live XAI API endpoints."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "review" / "Bantay_Kontrata_Dashboard.html"
SCHEMA = "bantay-kontrata-tagum-reviews-v1"
TOKEN_ERROR = "Enter the team review access token in Saved reviews to save changes."


class FakeXAIEngine:
    """Deterministic in-memory stub implementing the LiveXAIEngine interface for DIP tests."""

    def __init__(self) -> None:
        self.last_vendor_call: tuple[str, int] | None = None
        self.last_surface_call: tuple[str, int, int, int] | None = None

    def get_meta(self) -> dict:
        return {
            "categories": ["Construction Projects", "Drugs and Medicines"],
            "procurement_modes": ["Public Bidding", "Small Value Procurement"],
            "classifications": ["Goods", "Infrastructure"],
            "cutoffs": {"award": 0.5612, "tender": 0.5489, "dpwh": 0.5731},
        }

    def search_vendors(self, query: str, limit: int = 25) -> list[dict]:
        self.last_vendor_call = (query, limit)
        catalog = [
            {"awardee_name": "TAGUM BUILDERS CORP.", "award_count": 14, "total_php": 48_500_000.0},
            {"awardee_name": "DAVAO MED SUPPLY", "award_count": 9, "total_php": 11_200_000.0},
        ]
        needle = query.strip().lower()
        matches = [row for row in catalog if not needle or needle in row["awardee_name"].lower()]
        return matches[:limit]

    def score_custom_tender(self, payload: dict) -> dict:
        budget = payload["approved_budget_php"]
        window = payload["notice_window_days"]
        if not isinstance(budget, (int, float)) or budget <= 0:
            raise ValueError("approved_budget_php must be positive.")
        if not isinstance(window, (int, float)) or window < 0:
            raise ValueError("notice_window_days must be non-negative.")
        return {
            "kind": "tender",
            "anomaly_score": 0.612,
            "extended_score": 0.598,
            "review": True,
            "consensus": "3/3 High Consensus",
            "approved_budget_php": float(budget),
            "notice_window_days": float(window),
        }

    def score_custom_award(self, payload: dict) -> dict:
        amount = payload["contract_amount"]
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("contract_amount must be positive.")
        return {
            "kind": "award",
            "anomaly_score": 0.584,
            "extended_score": 0.571,
            "review": True,
            "awardee_name": payload.get("awardee_name", "NEW VENDOR"),
            "contract_amount": float(amount),
        }

    def simulate_whatif(self, payload: dict) -> dict:
        kind = payload.get("kind")
        if kind not in {"award", "tender", "dpwh"}:
            raise ValueError("kind must be award, tender, or dpwh.")
        return {
            "kind": kind,
            "simulated_score": 0.492,
            "below_cutoff": True,
            "overrides": payload.get("overrides", {}),
        }

    def get_2d_surface(self, kind: str, fx: int, fy: int, grid_size: int) -> dict:
        self.last_surface_call = (kind, fx, fy, grid_size)
        return {
            "kind": kind,
            "fx": fx,
            "fy": fy,
            "grid_size": grid_size,
            "if_grid": [[0.45] * grid_size for _ in range(grid_size)],
            "eif_grid": [[0.44] * grid_size for _ in range(grid_size)],
        }


@unittest.skipUnless(find_spec("flask"), "Install requirements-app.txt for app tests")
class DeployAndLiveXAITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        html = DASHBOARD.read_text(encoding="utf-8")
        raw_payload = html.split('<script id="dataset" type="application/json">', 1)[1].split("</script>", 1)[0]
        data = json.loads(raw_payload)
        cls.snapshot = data["snapshot_id"]
        cls.award_key = "award:" + data["award"]["records"][0]["record_id"]
        cls.tender_key = "tender:" + data["tender"]["records"][0]["bid_reference"]
        cls.dpwh_key = "dpwh:" + data["projects"][0]["dpwh_contract_id"]
        cls.expected_counts = {
            "awards": len(data["award"]["records"]),
            "tenders": len(data["tender"]["records"]),
            "dpwh": len(data["projects"]),
        }

    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.temp_dir = Path(temp.name)
        self.database = self.temp_dir / "reviews.sqlite3"
        self.fake_engine = FakeXAIEngine()

    def make_client(self, **kwargs):
        from app import create_app

        defaults = {
            "db_path": self.database,
            "dashboard_path": DASHBOARD,
            "xai_engine": self.fake_engine,
        }
        defaults.update(kwargs)
        return create_app(**defaults).test_client()

    def sample_review(self, key: str | None = None, **changes) -> dict:
        return {
            "key": key or self.award_key,
            "status": "reviewed",
            "reviewer": "COA-1",
            "evidence_url": "https://philgeps.gov.ph/notice/12345",
            "evidence_ref": "BAC-RES-2025-09",
            "notes": "Verified supporting BAC abstract of quotations.",
            "updated_at": "2026-11-17T08:30:00.000Z",
            **changes,
        }

    def test_health_endpoint_reports_counts_db_and_xai_status(self) -> None:
        client = self.make_client()
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["snapshot_id"], self.snapshot)
        self.assertTrue(payload["db_ready"])
        self.assertTrue(payload["live_xai_available"])
        self.assertTrue(payload["live_xai_warmed"])
        self.assertFalse(payload["review_token_required"])
        self.assertEqual(payload["counts"], self.expected_counts)

        real_lazy_client = self.make_client(xai_engine=None)
        real_lazy_health = real_lazy_client.get("/api/health").get_json()
        self.assertTrue(real_lazy_health["live_xai_available"])
        self.assertFalse(real_lazy_health["live_xai_warmed"])

        with patch.dict("sys.modules", {"xai_engine": None}), patch("app.find_spec", return_value=None):
            unwarmed_client = self.make_client(xai_engine=None, review_token="team-secret")
            unwarmed_health = unwarmed_client.get("/api/health").get_json()
        self.assertFalse(unwarmed_health["live_xai_available"])
        self.assertFalse(unwarmed_health["live_xai_warmed"])
        self.assertTrue(unwarmed_health["review_token_required"])

    def test_dpwh_reviews_and_revision_history_filtering(self) -> None:
        client = self.make_client()

        wrong_snap = client.get("/api/reviews/history", query_string={"snapshot_id": "bad"})
        self.assertEqual(wrong_snap.status_code, 409)

        for bad_limit in ("0", "201", "abc"):
            with self.subTest(limit=bad_limit):
                bad_resp = client.get(
                    "/api/reviews/history",
                    query_string={"snapshot_id": self.snapshot, "limit": bad_limit},
                )
                self.assertEqual(bad_resp.status_code, 400)

        empty_hist = client.get("/api/reviews/history", query_string={"snapshot_id": self.snapshot}).get_json()
        self.assertEqual(empty_hist, {"snapshot_id": self.snapshot, "history": []})

        # Save award (rev 1), dpwh contract (rev 2), and update award (rev 3)
        for expected_rev, case_key, note in (
            (0, self.award_key, "Initial award triage."),
            (1, self.dpwh_key, "Checked DPWH geotagged flood structure."),
            (2, self.award_key, "Closed after COA voucher inspection."),
        ):
            put_resp = client.put(
                "/api/reviews",
                json={
                    "schema": SCHEMA,
                    "snapshot_id": self.snapshot,
                    "revision": expected_rev,
                    "source": "edit",
                    "entries": [self.sample_review(case_key, notes=note)],
                },
            )
            self.assertEqual(put_resp.status_code, 200, put_resp.get_data(as_text=True))

        # Unknown DPWH key must still be rejected
        bad_dpwh = client.put(
            "/api/reviews",
            json={
                "schema": SCHEMA,
                "snapshot_id": self.snapshot,
                "revision": 3,
                "source": "edit",
                "entries": [self.sample_review("dpwh:DOES_NOT_EXIST")],
            },
        )
        self.assertEqual(bad_dpwh.status_code, 400)

        all_hist = client.get("/api/reviews/history", query_string={"snapshot_id": self.snapshot}).get_json()
        self.assertEqual([item["revision"] for item in all_hist["history"]], [3, 2, 1])
        self.assertEqual(all_hist["history"][1]["entries"][0]["key"], self.dpwh_key)

        filtered = client.get(
            "/api/reviews/history",
            query_string={"snapshot_id": self.snapshot, "case_key": self.award_key},
        ).get_json()
        self.assertEqual([item["revision"] for item in filtered["history"]], [3, 1])

        limited = client.get(
            "/api/reviews/history",
            query_string={"snapshot_id": self.snapshot, "case_key": self.award_key, "limit": 1},
        ).get_json()
        self.assertEqual([item["revision"] for item in limited["history"]], [3])

    def test_cloud_trusted_hosts_and_proxy_fix_https_origin_verification(self) -> None:
        from app import create_app

        env_db = self.temp_dir / "cloud.sqlite3"
        with patch.dict(
            os.environ,
            {
                "BANTAY_DB_PATH": str(env_db),
                "BANTAY_DASHBOARD_PATH": str(DASHBOARD),
                "BANTAY_TRUSTED_HOSTS": "*",
                "BANTAY_TRUSTED_PROXY": "true",
            },
            clear=False,
        ):
            wildcard_client = create_app(xai_engine=self.fake_engine).test_client()

        proxy_headers = {
            "Host": "internal-container:8000",
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "bantay-tagum.onrender.com",
            "Origin": "https://bantay-tagum.onrender.com",
        }
        ok_resp = wildcard_client.put(
            "/api/reviews",
            headers=proxy_headers,
            json={
                "schema": SCHEMA,
                "snapshot_id": self.snapshot,
                "revision": 0,
                "source": "edit",
                "entries": [self.sample_review()],
            },
        )
        self.assertEqual(ok_resp.status_code, 200, ok_resp.get_data(as_text=True))
        self.assertTrue(env_db.exists())

        # Scheme mismatch (http Origin vs https X-Forwarded-Proto) must be blocked with 403
        mismatch_resp = wildcard_client.put(
            "/api/reviews",
            headers={**proxy_headers, "Origin": "http://bantay-tagum.onrender.com"},
            json={
                "schema": SCHEMA,
                "snapshot_id": self.snapshot,
                "revision": 1,
                "source": "edit",
                "entries": [self.sample_review()],
            },
        )
        self.assertEqual(mismatch_resp.status_code, 403)

        # Comma-separated custom agency domain list
        with patch.dict(
            os.environ,
            {
                "BANTAY_TRUSTED_HOSTS": "audit.tagum.gov.ph, localhost, 127.0.0.1",
                "BANTAY_TRUSTED_PROXY": "1",
            },
            clear=False,
        ):
            agency_client = create_app(
                db_path=self.database,
                dashboard_path=DASHBOARD,
                xai_engine=self.fake_engine,
            ).test_client()

        allowed_agency = agency_client.get(
            "/api/health",
            headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "audit.tagum.gov.ph"},
        )
        self.assertEqual(allowed_agency.status_code, 200)

        untrusted_agency = agency_client.get(
            "/api/health",
            headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "spoofed.invalid"},
        )
        self.assertEqual(untrusted_agency.status_code, 400)

    def test_review_token_protection_supports_header_and_bearer(self) -> None:
        from app import create_app

        secret = "tagum-bac-audit-token-99"
        with patch.dict(os.environ, {"BANTAY_REVIEW_TOKEN": secret}, clear=False):
            client = create_app(
                db_path=self.database,
                dashboard_path=DASHBOARD,
                xai_engine=self.fake_engine,
            ).test_client()

        # Public read still succeeds without token
        self.assertEqual(
            client.get("/api/reviews", query_string={"snapshot_id": self.snapshot}).status_code,
            200,
        )

        payload = {
            "schema": SCHEMA,
            "snapshot_id": self.snapshot,
            "revision": 0,
            "source": "edit",
            "entries": [self.sample_review()],
        }

        for bad_headers in (
            {},
            {"X-Review-Token": "wrong-token"},
            {"X-Review-Token": "🔑-non-ascii-wrong"},
            {"Authorization": "Bearer wrong-token"},
            {"Authorization": f"Basic {secret}"},
        ):
            with self.subTest(headers=bad_headers):
                denied = client.put("/api/reviews", headers=bad_headers, json=payload)
                self.assertEqual(denied.status_code, 401)
                self.assertEqual(denied.get_json()["error"], TOKEN_ERROR)

        ok_header = client.put("/api/reviews", headers={"X-Review-Token": secret}, json=payload)
        self.assertEqual(ok_header.status_code, 200)

        ok_bearer = client.put(
            "/api/reviews",
            headers={"Authorization": f"Bearer {secret}"},
            json={**payload, "revision": 1, "entries": [self.sample_review(self.tender_key)]},
        )
        self.assertEqual(ok_bearer.status_code, 200)

    def test_live_xai_endpoints_happy_paths_validation_and_security(self) -> None:
        client = self.make_client()

        # 1. /api/xai/meta
        meta = client.get("/api/xai/meta")
        self.assertEqual(meta.status_code, 200)
        self.assertIn("cutoffs", meta.get_json())

        # 2. /api/xai/vendors
        vendors = client.get("/api/xai/vendors", query_string={"q": "tagum", "limit": "999"})
        self.assertEqual(vendors.status_code, 200)
        self.assertEqual(len(vendors.get_json()), 1)
        self.assertEqual(self.fake_engine.last_vendor_call, ("tagum", 50))

        client.get("/api/xai/vendors", query_string={"q": "", "limit": "-5"})
        self.assertEqual(self.fake_engine.last_vendor_call, ("", 1))

        bad_vendor_limit = client.get("/api/xai/vendors", query_string={"limit": "nan"})
        self.assertEqual(bad_vendor_limit.status_code, 400)

        # 3. /api/xai/score-tender
        tender_ok = client.post(
            "/api/xai/score-tender",
            headers={"Origin": "http://localhost"},
            json={"approved_budget_php": 995_000, "notice_window_days": 3},
        )
        self.assertEqual(tender_ok.status_code, 200)
        self.assertEqual(tender_ok.get_json()["kind"], "tender")

        # Missing key -> KeyError -> 400; invalid value -> ValueError -> 400; non-dict body -> 400
        self.assertEqual(client.post("/api/xai/score-tender", json={}).status_code, 400)
        self.assertEqual(
            client.post("/api/xai/score-tender", json={"approved_budget_php": -10, "notice_window_days": 4}).status_code,
            400,
        )
        self.assertEqual(client.post("/api/xai/score-tender", json=["not-an-object"]).status_code, 400)

        # Cross-origin POST rejection -> 403
        for route in ("/api/xai/score-tender", "/api/xai/score-award", "/api/xai/whatif"):
            with self.subTest(route=route):
                cross_origin = client.post(
                    route,
                    headers={"Origin": "https://evil.invalid"},
                    json={"kind": "award", "contract_amount": 100_000},
                )
                self.assertEqual(cross_origin.status_code, 403)

        # 4. /api/xai/score-award
        award_ok = client.post(
            "/api/xai/score-award",
            json={"contract_amount": 1_980_000, "awardee_name": "TAGUM BUILDERS CORP."},
        )
        self.assertEqual(award_ok.status_code, 200)
        self.assertEqual(award_ok.get_json()["kind"], "award")
        self.assertEqual(client.post("/api/xai/score-award", json={"contract_amount": 0}).status_code, 400)

        # 5. /api/xai/whatif
        whatif_ok = client.post(
            "/api/xai/whatif",
            json={"kind": "tender", "overrides": {"notice_window_days": 10}},
        )
        self.assertEqual(whatif_ok.status_code, 200)
        self.assertTrue(whatif_ok.get_json()["below_cutoff"])
        self.assertEqual(client.post("/api/xai/whatif", json={"kind": "invalid"}).status_code, 400)

        # 6. /api/xai/surface
        surface_ok = client.get("/api/xai/surface", query_string={"kind": "dpwh", "fx": "0", "fy": "3"})
        self.assertEqual(surface_ok.status_code, 200)
        self.assertEqual(self.fake_engine.last_surface_call, ("dpwh", 0, 3, 24))

        for bad_params in (
            {"kind": "unknown", "fx": "0", "fy": "1"},
            {"kind": "award", "fx": "-1", "fy": "1"},
            {"kind": "award", "fx": "0", "fy": "5"},
            {"kind": "award", "fx": "2", "fy": "2"},
            {"kind": "award", "fx": "0", "fy": "1", "grid_size": "11"},
            {"kind": "award", "fx": "0", "fy": "1", "grid_size": "37"},
            {"kind": "award", "fx": "zero", "fy": "1"},
        ):
            with self.subTest(params=bad_params):
                self.assertEqual(client.get("/api/xai/surface", query_string=bad_params).status_code, 400)

    def test_graceful_503_when_xai_engine_unavailable(self) -> None:
        with patch.dict("sys.modules", {"xai_engine": None}):
            client = self.make_client(xai_engine=None)
            response = client.get("/api/xai/meta")
            self.assertEqual(response.status_code, 503)
            self.assertIn("error", response.get_json())

    def test_lazy_xai_provider_is_thread_safe_and_wsgi_loads_with_env(self) -> None:
        from concurrent.futures import ThreadPoolExecutor
        import importlib
        import types

        init_calls: list[Path] = []

        class CountingEngine(FakeXAIEngine):
            def __init__(self, data_dir: Path) -> None:
                super().__init__()
                init_calls.append(Path(data_dir))

        fake_module = types.ModuleType("xai_engine")
        fake_module.LiveXAIEngine = CountingEngine  # type: ignore[attr-defined]

        from app import XAIProvider

        provider = XAIProvider(ROOT / "data", injected_engine=None)
        self.assertFalse(provider.is_warmed)
        with patch.dict("sys.modules", {"xai_engine": fake_module}):
            self.assertTrue(provider.is_available())
            with ThreadPoolExecutor(max_workers=6) as pool:
                engines = list(pool.map(lambda _: provider.get_engine(), range(12)))
        self.assertEqual(len(init_calls), 1)
        self.assertTrue(all(eng is engines[0] for eng in engines))
        self.assertTrue(provider.is_warmed)

        wsgi_db = self.temp_dir / "wsgi.sqlite3"
        with patch.dict(
            os.environ,
            {
                "BANTAY_DB_PATH": str(wsgi_db),
                "BANTAY_DASHBOARD_PATH": str(DASHBOARD),
                "BANTAY_TRUSTED_HOSTS": "*",
            },
            clear=False,
        ):
            import wsgi

            reloaded = importlib.reload(wsgi)
            resp = reloaded.app.test_client().get("/api/health", headers={"Host": "paas-check.internal"})
            self.assertEqual(resp.status_code, 200)

    def test_deployment_artifacts_exist_and_are_valid(self) -> None:
        expected_files = (
            "wsgi.py",
            "Dockerfile",
            ".dockerignore",
            "Procfile",
            "render.yaml",
            "railway.json",
            ".env.example",
            "requirements.txt",
            "requirements-app.txt",
        )
        for filename in expected_files:
            with self.subTest(filename=filename):
                self.assertTrue((ROOT / filename).is_file(), f"Missing {filename}")

        railway = json.loads((ROOT / "railway.json").read_text(encoding="utf-8"))
        self.assertEqual(railway["deploy"]["healthcheckPath"], "/api/health")

        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("python:3.12-slim", dockerfile)
        self.assertIn("appuser", dockerfile)
        self.assertIn("/api/health", dockerfile)

        env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
        for env_var in (
            "HOST",
            "PORT",
            "BANTAY_DB_PATH",
            "BANTAY_DASHBOARD_PATH",
            "BANTAY_DATA_DIR",
            "BANTAY_TRUSTED_HOSTS",
            "BANTAY_TRUSTED_PROXY",
            "BANTAY_REVIEW_TOKEN",
        ):
            self.assertIn(env_var, env_example)

    def test_real_live_xai_engine_end_to_end_via_flask_routes(self) -> None:
        client = self.make_client(xai_engine=None)
        meta_resp = client.get("/api/xai/meta")
        self.assertEqual(meta_resp.status_code, 200)
        meta = meta_resp.get_json()
        self.assertGreaterEqual(len(meta["presets"]), 4)
        self.assertTrue(client.get("/api/health").get_json()["live_xai_warmed"])

        tender_preset = next(p for p in meta["presets"] if p["kind"] == "tender")
        tender_resp = client.post("/api/xai/score-tender", json=tender_preset["payload"])
        self.assertEqual(tender_resp.status_code, 200)
        tender_data = tender_resp.get_json()
        self.assertIn("waterfall", tender_data)
        self.assertIn("counterfactual", tender_data)
        self.assertIn("consensus", tender_data)
        self.assertEqual(len(tender_data["tree_shap"]), 5)

        award_preset = next(p for p in meta["presets"] if p["kind"] == "award")
        award_resp = client.post("/api/xai/score-award", json=award_preset["payload"])
        self.assertEqual(award_resp.status_code, 200)
        award_data = award_resp.get_json()
        self.assertIn("vendor_profile", award_data)
        self.assertIn("nearest_peers", award_data)

        first_award_id = self.award_key.split(":", 1)[1]
        whatif_resp = client.post(
            "/api/xai/whatif",
            json={"kind": "award", "record_id": first_award_id, "overrides": {"contract_amount": 150_000}},
        )
        self.assertEqual(whatif_resp.status_code, 200)
        self.assertIn("waterfall", whatif_resp.get_json())

        surf_resp = client.get("/api/xai/surface", query_string={"kind": "tender", "fx": "0", "fy": "1"})
        self.assertEqual(surf_resp.status_code, 200)
        self.assertIn("eif_scores", surf_resp.get_json())


if __name__ == "__main__":
    unittest.main()
