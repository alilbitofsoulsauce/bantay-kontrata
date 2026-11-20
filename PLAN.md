# Bantay-Kontrata Tagum — Deployable Explainable AI (XAI) Web App Plan

## Vision & Grounded Real-World Purpose
Transform **Bantay-Kontrata Tagum** from a localhost-only static snapshot viewer into a **complete, cloud-deployable, XAI-centered public procurement and infrastructure audit web application** grounded in real workflows for:
- **Tagum City Bids and Awards Committee (BAC) & Secretariat** (pre-publication tender screening, RA 12009 ceiling & bid-window checks),
- **Commission on Audit (COA) Resident Auditors & Internal Audit Units** (evidence-backed case triage, multi-method XAI root-cause verification, peer contract comparison, audit trail logging, printable verification dossiers),
- **Civil Society Monitors, Researchers, & Civic Journalists** (vendor concentration & 30-day split-contract burst analysis, barangay DPWH infrastructure & flood-control tracking, transparent model validation).

---

## Architectural Workstreams & Strict File Ownership

### Workstream 1: Four-Pillar XAI & Procurement Intelligence Engine
**Files Owned:** `xai_engine.py`, `tests/test_xai_engine.py`
1. **Pillar 1 — Exact TreeSHAP Path-to-Score Waterfall (`build_shap_waterfall`)**:
   - Decomposes expected isolation path length $E[h(x)] = \phi_0 + \sum \phi_j(x)$ and maps every cumulative step via $s(h) = 2^{-h / c(\tilde{n})}$ to show exact score movement relative to the 95th-percentile calibration cutoff.
2. **Pillar 2 — Intrinsic Induced-Imbalance Local & Global DIFFI (`compute_imbalance_diffi`)**:
   - Implements Carletti et al. (2023) Induced Imbalance Coefficient $\lambda(k) \in [0.5, 1.0]$ across isolation tree internal splits to produce both **Global-DIFFI** (outlier-to-inlier cumulative importance ratio) and **Imbalance-Weighted Local-DIFFI**.
3. **Pillar 3 — Extended Isolation Forest (EIF) + ExIFFI & 2D Decision Surfaces (`compute_2d_decision_surface`)**:
   - Generates 2D decision contour grids (`24x24` or configurable) comparing orthogonal **Standard Isolation Forest** vs oblique **Extended Isolation Forest** + **ExIFFI** directional importance across correlated feature pairs (e.g., `Budget vs. Bid Window`, `Amount vs. Vendor Share`, `Cost/ABC Ratio vs. Schedule Variance`).
4. **Pillar 4 — Multi-Feature Constrained Counterfactual Recourse (`compute_counterfactual_recourse`)**:
   - Respects immutable context (classification, category, filing quarter) while searching mutable procurement levers (bid window days, budget/amount vs peer median, vendor concentration, or procurement mode peer group) to find minimal, sparse, policy-plausible recourse below the review cutoff, reporting **validity**, **sparsity**, and human-readable audit narrative.
5. **Cross-Paradigm XAI Consensus & Statutory Checks**:
   - `compute_xai_consensus`: Evaluates rank agreement across TreeSHAP, Local-DIFFI, and ExIFFI (`3/3 High Consensus`, `2/3 Moderate Consensus`, or `Oblique Multivariate Interaction` when EIF/ExIFFI catches diagonal patterns).
   - `statutory_procurement_checks`: Grounded in **RA 9184** & **RA 12009 (NGPA)** — SVP ₱2M/₱1M ceiling proximity, compressed solicitation windows, tight 0.995–1.000 award-to-ABC ratios, 30-day same-vendor category bursts, and quarter-end spend rush.
6. **DPWH Infrastructure XAI Screen (`run_dpwh_screen`)**:
   - Fits Isolation Forest + Extended Isolation Forest + TreeSHAP + DIFFI + ExIFFI + Counterfactual Recourse on Tagum's 336 DPWH infrastructure contracts (`cost_abc_gap`, `abc_peer_log_ratio`, `variance_days`, `planned_duration_days`, `barangay_density`) plus 23-barangay choropleth risk rollups.
7. **Vendor Concentration, Split-Risk Clusters & Overlap Intelligence (`build_procurement_intelligence`)**:
   - Aggregates vendor concentration, 30-day same-category multi-award clusters, and unresolved archive-to-official overlap pairs (`tagum_possible_award_overlaps.csv`).
8. **Live Real-Time XAI Inference Engine (`LiveXAIEngine`)**:
   - Thread-safe, cached engine supporting live scoring & full 4-pillar XAI explanation of new/hypothetical tenders (`score_custom_tender`), new awards with real Tagum vendor history lookup (`score_custom_award`), live multi-slider counterfactual simulation (`simulate_whatif`), 2D surface generation (`get_2d_surface`), and vendor autocomplete (`search_vendors`).

---

### Workstream 2: Production Web Server, Live XAI APIs & Cloud Deployment
**Files Owned:** `app.py`, `wsgi.py`, `Dockerfile`, `.dockerignore`, `render.yaml`, `railway.json`, `Procfile`, `.env.example`, `requirements.txt`, `requirements-app.txt`, `tests/test_deploy_and_live_xai.py`
1. **Cloud-Ready Configuration & Security (`app.py`, `wsgi.py`)**:
   - Preserve 100% compatibility with `tests/test_app.py` when called with defaults (`TRUSTED_HOSTS=["localhost", "127.0.0.1"]`).
   - Support environment variables for cloud & agency deployments: `HOST`, `PORT`, `BANTAY_DB_PATH`, `BANTAY_DASHBOARD_PATH`, `BANTAY_TRUSTED_HOSTS` (comma-separated hosts or `*`), `BANTAY_TRUSTED_PROXY=1` (`Werkzeug ProxyFix`), and optional `BANTAY_REVIEW_TOKEN` for write-protected shared deployments.
2. **REST API Endpoints**:
   - `GET /api/health` — Readiness check with snapshot ID, record counts, DB status, and XAI engine readiness.
   - `GET /api/reviews` & `PUT /api/reviews` — Existing optimistic-concurrency SQLite store (preserved) + optional token check when `BANTAY_REVIEW_TOKEN` is configured.
   - `GET /api/reviews/history` — Immutable audit log from `review_changes` table (supports `snapshot_id`, optional `case_key`, `limit`).
   - `GET /api/xai/meta` — Categories, procurement modes, classifications, model cutoffs, global XAI importances, and sample templates for Live Triage.
   - `GET /api/xai/vendors` — Fast vendor search (`?q=...`) returning historical Tagum award counts, spend share, and recent category activity.
   - `POST /api/xai/score-tender` — Live XAI pre-publication / new notice triage.
   - `POST /api/xai/score-award` — Live XAI contract award triage.
   - `POST /api/xai/whatif` — Live multi-feature sensitivity & counterfactual simulator.
   - `GET /api/xai/surface` — Live 2D IF vs EIF + ExIFFI anomaly contour grid for any feature pair.
3. **One-Click Deployment Artifacts**:
   - `Dockerfile` (Python 3.12-slim, non-root user, healthcheck, Gunicorn/Waitress, persistent `/data` volume), `.dockerignore`, `wsgi.py`, `Procfile`, `render.yaml`, `railway.json`, `.env.example`.

---

### Workstream 3: Enriched Data Pipeline & XAI-First Web Application UI
**Files Owned:** `publish_dashboard.py`, `dashboard_template.html`, `review_workflow.js`
1. **Always-Visible XAI Case Workstation (Award, Tender, & DPWH Inspectors)**:
   - Prominent **XAI Consensus & Dual-Forest Risk Header** (Isolation Forest vs Extended Isolation Forest gauges + XAI agreement badge + explicit methodological guidance banner).
   - **Interactive SHAP Path-to-Score Waterfall Visualization** tracing baseline expected splits → feature pushes → final isolated path & anomaly score vs cutoff.
   - **3-Paradigm XAI Comparison (`TreeSHAP` · `Local-DIFFI` · `ExIFFI`)** with side-by-side normalized comparison view + individual deep-dive modes.
   - **Actionable Counterfactual Recourse Card & Multi-Feature What-If Simulator** (live server API backed + instant client fallback).
   - **Statutory RA 12009 / GPPB Audit Checks & Nearest Historical Tagum Peer Contracts** for direct apples-to-apples verification.
   - **Printable Case Verification Dossier** (`Print / Export Case Brief`) formatted cleanly via `@media print` / dedicated print view.
2. **Live XAI Triage Tab ("Live XAI Scorer")**:
   - Real-time pre-screening form for **New Tenders** and **New Awards** with realistic Tagum quick-fill presets, live Tagum vendor autocomplete, instant 4-pillar XAI waterfall, counterfactual recourse, statutory checks, and comparable Tagum historical records.
3. **XAI Lab & Procurement Intelligence Tab ("XAI & Vendor Lab")**:
   - **2D Oblique vs Axis-Aligned Decision Surface Explorer** (interactive canvas heatmap comparing Standard Isolation Forest vs Extended Isolation Forest + ExIFFI with clickable Tagum records).
   - **Global XAI Feature Importance Matrix** (`Global-DIFFI`, `Mean |TreeSHAP|`, `Mean ExIFFI`) across Award, Tender, and DPWH screens.
   - **Vendor Concentration & 30-Day Split-Risk Watchlist** + **326 Possible Overlap Reconciliation Queue** linking straight to case review.
4. **Upgraded Infrastructure & Saved Reviews Tabs**:
   - **Barangay Choropleth Mode** on the Tagum 23-barangay canvas map (color by DPWH XAI flags, delay variance, project count, or flood-control links) + full DPWH XAI case inspector.
   - **Audit Progress Funnel & Revision History Log** in Saved Reviews (plus optional Review Token input when deployed in protected team mode).

---

### Workstream 4: End-to-End Integration, Rebuild, Automated Tests & Live Browser Verification
1. Rebuild `review/Bantay_Kontrata_Dashboard.html` and CSV/JSON artifacts via `python publish_dashboard.py`.
2. Run complete unit & integration test suite (`unittest discover -s tests -v`).
3. Start the production server locally and verify every tab, XAI waterfall, 2D oblique surface, live XAI tender/award scorer, counterfactual simulator, barangay choropleth map, review persistence, audit history, and responsive layout via the `browser` subagent.
