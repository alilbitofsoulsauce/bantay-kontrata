# Bantay-Kontrata Tagum

**Explainable AI (XAI) Public Procurement & Infrastructure Audit Web Application for the City of Tagum, Davao del Norte.**

Bantay-Kontrata Tagum pairs dual unsupervised anomaly detectors (**Standard Isolation Forest** and **Extended Isolation Forest**) with a **four-pillar Explainable AI (XAI) engine** to help **Bids and Awards Committee (BAC) staff**, **Commission on Audit (COA) resident auditors**, **internal audit units**, and **civic monitors** prioritize records for document-backed verification.

Every anomaly score is decomposed into transparent, human-verifiable audit leads grounded in **Republic Act No. 9184** and **Republic Act No. 12009 (New Government Procurement Act, IRR Feb 2025)** across **12,300 reconciled award register rows**, **3,395 official PhilGEPS tenders**, and **336 DPWH infrastructure projects** mapped across Tagum City's **23 barangays**.

> **Methodological Principle:** Model scores and XAI attributions prioritize records and point reviewers to specific procurement documents—they are never standalone findings of irregularity. Always verify flagged cases against official notices, bid bulletins, BAC resolutions, abstracts of bids/quotations, and signed contracts.

---

## Quick Start (Local Workstation)

From Python 3.12+ in the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-app.txt
python app.py
```

Open **http://127.0.0.1:8000**.
- Case reviews and revision history persist to `instance/bantay-kontrata.sqlite3`.
- All **Live XAI** endpoints (pre-publication tender triage, custom award scoring, multi-feature what-if simulation, and dynamic 2D decision surfaces) run locally with zero external API dependencies.
- For offline field use without a Python server, open `review/Bantay_Kontrata_Dashboard.html` directly in any modern browser.

---

## Deploy Anywhere (Docker, Render, Railway, or Agency Server)

The web application ships ready for containerized, cloud PaaS, or intranet agency deployment:

| Platform | Configuration File | How to Deploy |
|---|---|---|
| **Docker / Cloud Run / Fly.io** | `Dockerfile`, `.dockerignore` | `docker build -t bantay-kontrata . && docker run -p 8000:8000 -v bantay-data:/var/lib/bantay-kontrata bantay-kontrata` |
| **Render** | `render.yaml` | Create a Blueprint instance pointing to this repo; provisions persistent `/var/data` SQLite storage and `/api/health` checks automatically. |
| **Railway** | `railway.json` | Deploy from GitHub; uses `Dockerfile` and `/api/health` readiness probes out of the box. |
| **Gunicorn / Procfile** | `wsgi.py`, `Procfile` | `gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120 wsgi:app` |

### Environment Configuration (`.env.example`)

Copy `.env.example` to `.env` (never commit real tokens):

| Variable | Default | Purpose |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address and port for `python app.py` or container runtime |
| `BANTAY_DB_PATH` | `instance/bantay-kontrata.sqlite3` | SQLite database path for optimistic-concurrency reviews and audit history |
| `BANTAY_DASHBOARD_PATH` | `review/Bantay_Kontrata_Dashboard.html` | Snapshot HTML served at `/` |
| `BANTAY_DATA_DIR` | `data` | Source CSV directory for `LiveXAIEngine` |
| `BANTAY_TRUSTED_HOSTS` | `localhost,127.0.0.1` | Comma-separated allowed `Host` headers, or `*` behind cloud load balancers |
| `BANTAY_TRUSTED_PROXY` | `0` | Set to `1` behind TLS reverse proxies (`ProxyFix` for HTTPS same-origin checks) |
| `BANTAY_REVIEW_TOKEN` | *(empty)* | Optional shared team token required in `X-Review-Token` (or `Bearer`) for `PUT /api/reviews` |

---

## The Four-Pillar Explainable AI (XAI) Architecture

Standard Isolation Forests act as ensembles of random partitioning trees whose raw binary flags fail audit transparency standards. Bantay-Kontrata Tagum implements four complementary XAI paradigms (`xai_engine.py`, `screen.py`, `tender_screen.py`):

1. **Exact TreeSHAP Path-to-Score Waterfall (`shap.TreeExplainer`)**
   - Decomposes expected isolation path depth $E[h(x)] = \phi_0 + \sum_{j=1}^d \phi_j(x)$ to machine precision ($<10^{-13}$ splits error) and maps every cumulative step via $s(h) = 2^{-h / c(\tilde{n})}$ directly onto the calibrated 95th-percentile anomaly score scale.
   - Negative split contributions shorten isolation paths (pushing scores upward toward review); positive contributions lengthen paths (normalizing the record).

2. **Intrinsic Induced-Imbalance Local & Global DIFFI (Carletti et al., 2023)**
   - Model-specific, perturbation-free interpretability weighting shallower isolation depths alongside internal node **Induced Imbalance Coefficients** $\lambda(k) \in [0.5, 1.0]$.
   - Computes per-case **Imbalance-Weighted Local-DIFFI** and model-wide **Global-DIFFI** outlier-to-inlier importance ratios.

3. **Extended Isolation Forest (EIF) + ExIFFI Oblique Hyperplanes & 2D Surfaces**
   - Eliminates axis-aligned orthogonal "ghost zones" using Gaussian random oblique hyperplanes, projecting node sample-inequality gains onto hyperplane normal vectors (**ExIFFI**, Arcudi et al., 2024).
   - Interactive **2D Decision Surface Explorer** contrasts axis-aligned Isolation Forest vs. oblique Extended Isolation Forest contours on correlated feature pairs (e.g., *Approved Budget vs. Bid Window*, *Contract Amount vs. Vendor Share*, *Cost/ABC Ratio vs. Schedule Delay*).

4. **Constrained Multi-Feature Counterfactual Recourse & Statutory RA 12009 Checks**
   - Searches mutable procurement levers while locking immutable context (classification, category, filing period) to identify minimal, sparse, policy-grounded adjustments that move a flagged case to or below the calibrated review cutoff.
   - Pairs algorithmic recourse with deterministic statutory checks under **RA 9184** and **RA 12009**:
     - Small Value Procurement proximity to the **₱2,000,000** (RA 12009) and **₱1,000,000** (RA 9184) ceilings,
     - Compressed PhilGEPS solicitation windows (`<7` days for Public Bidding, `<3` days for alternative modes),
     - Tight `99.5%–100.0%` award-to-ABC price margins,
     - 30-day same-vendor, same-category award bursts (anti-splitting rule),
     - Quarter-end fiscal rush (`≤5` days to quarter close) and DPWH schedule overruns (`≥30` days).

---

## Core Web App Workspaces

1. **Award Review (`1,317` held-out 2025 awards · `106` flagged)**
   - Filter by category, month timeline, review status, XAI consensus/driver, or linked PhilGEPS tender.
   - Inspect dual-forest threshold gauges, XAI consensus tier (`3/3`, `2/3`, or `Oblique Interaction`), interactive stepped TreeSHAP waterfall, side-by-side 3-method XAI matrix, counterfactual recourse, live multi-slider what-if simulation, nearest Tagum peer contracts, and **Printable Case Verification Briefs**.

2. **Tender Review (`516` held-out 2026 tenders · `48` flagged)**
   - Screen published solicitation notices using rolling 365-day publication-time peer medians (strictly excluding same-day notices and post-publication award outcomes), cross-check related solicitation notices, and run live budget/window sensitivity tests.

3. **Live XAI Triage (Pre-Publication & Custom Contract Scorer)**
   - Score and explain **new or draft PhilGEPS tender notices** and **contract awards** in real time before or after publication.
   - Includes realistic Tagum quick-fill presets, live autocomplete across historical Tagum suppliers (automatically pulling 365-day spend share and 30-day category burst counts), full 4-pillar XAI diagnostics, counterfactual recourse, and closest historical peer comparators.

4. **XAI & Vendor Lab**
   - **2D Oblique vs. Axis-Aligned Surface Explorer** across Award, Tender, and DPWH screens with clickable Tagum records.
   - **Global XAI Feature Importance** comparing `Global-DIFFI`, `Mean |TreeSHAP|`, and `Mean ExIFFI`.
   - **Procurement Intelligence**: Top-60 **Vendor Concentration Watchlist**, **30-Day Same-Category Split-Risk Clusters**, and **Archive-to-Official Overlap Candidates** (`326` register rows in `1,765` candidate pairs).

5. **Infrastructure (`336` Tagum DPWH contracts · `23` barangays · `10` flood-map matches)**
   - **Barangay Choropleth Map** shading Tagum's 23 barangays by DPWH XAI flags, mean schedule delay variance, project count, or flood-control links, paired with full 4-pillar DPWH XAI case inspection and review logging.

6. **Saved Reviews & Immutable Audit Trail**
   - Optimistic-concurrency SQLite storage across `award:`, `tender:`, and `dpwh:` cases, review funnel KPIs, chronological **Server Revision History**, CSV export, portable JSON backup/restore with side-by-side conflict resolution, and optional team token authentication.

7. **Validation & Reproducibility**
   - Source reconciliation tables (`11,305 + 3,193 − 2,198 = 12,300` award register rows), chronological Warm-up / Training / Calibration / Evaluation splits, TreeSHAP exactness, feature-deletion faithfulness, 8-seed rank stability, counterfactual recourse validity/sparsity, and synthetic stress benchmarks.

---

## REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | `GET` | Readiness probe (`status`, `snapshot_id`, `db_ready`, `live_xai_available`, `counts`) |
| `/api/reviews` | `GET`, `PUT` | Read or upsert optimistic-concurrency case reviews for the active `snapshot_id` |
| `/api/reviews/history` | `GET` | Chronological revision log from `review_changes` (`?snapshot_id=...&case_key=...&limit=50`) |
| `/api/xai/meta` | `GET` | Categories, procurement modes, classifications, cutoffs, global XAI vectors, and triage presets |
| `/api/xai/vendors` | `GET` | Historical Tagum vendor search (`?q=...&limit=25`) with award counts, spend share, and 30d bursts |
| `/api/xai/score-tender` | `POST` | Live 4-pillar XAI scoring, recourse, statutory checks, and peers for a tender notice |
| `/api/xai/score-award` | `POST` | Live 4-pillar XAI scoring, vendor lookup, recourse, statutory checks, and peers for an award |
| `/api/xai/whatif` | `POST` | Live multi-feature sensitivity & TreeSHAP recalculation (`award`, `tender`, or `dpwh`) |
| `/api/xai/surface` | `GET` | 2D Isolation Forest vs. Extended Isolation Forest contour grid (`?kind=tender&fx=0&fy=1`) |

---

## Rebuild Snapshot & Run Tests

```bash
# Rebuild review/Bantay_Kontrata_Dashboard.html, ranked CSVs, crosswalks, and validation.json
python publish_dashboard.py

# Stage a new quarterly PS-PhilGEPS CSV release
python refresh_official.py \
  --input DATASETS_PHILGEPS_2026_JULY-SEPTEMBER.csv \
  --release 2026Q3 \
  --source-page https://philgeps.gov.ph/CmsHomePages/open-data \
  --output staged-2026Q3
python publish_dashboard.py --data staged-2026Q3 --output review-2026Q3

# Run the full 36-test suite (reconciliation, temporal context, XAI engine, app security, live APIs)
python -m unittest discover -s tests -v
```

### Key Repository Files

| Path | Purpose |
|---|---|
| `app.py`, `wsgi.py` | Flask/WSGI production web server, SQLite review store, revision history, and Live XAI APIs |
| `xai_engine.py` | Four-pillar XAI stack, counterfactual recourse, statutory checks, DPWH screen, intelligence, and `LiveXAIEngine` |
| `screen.py`, `tender_screen.py` | Chronological award and tender Isolation Forest + Extended Isolation Forest pipelines and stress evaluations |
| `integration.py`, `refresh_official.py` | Archive-to-official PhilGEPS reconciliation and quarterly release staging |
| `publish_dashboard.py` | Snapshot compiler generating `review/Bantay_Kontrata_Dashboard.html`, CSV tables, and `validation.json` |
| `dashboard_template.html`, `review_workflow.js` | Self-contained 7-tab XAI web workstation and offline-capable review workflow |
| `Dockerfile`, `render.yaml`, `railway.json`, `Procfile` | Cloud and container deployment configurations |
| `tests/` | Automated unit and end-to-end tests (`test_xai_engine.py`, `test_deploy_and_live_xai.py`, `test_app.py`, `test_project.py`, `test_refresh_official.py`) |
