# Bantay-Kontrata Tagum

An explainable review workflow for City of Tagum awards, published tenders, and DPWH projects. The official procurement extracts use **CITY OF TAGUM, DAVAO DEL NORTE** as the buyer. Scores prioritize records for manual verification.

This release supports a retrospective review pilot on a single workstation or browser profile. The dashboard dates show what the snapshot covers. The latest official award is dated September 29, 2025. The latest included tender was published September 26, 2026.

## Open the project

| Goal | Start here |
|---|---|
| Explore the results | Open **review/Bantay_Kontrata_Dashboard.html** in a browser from the repository or extracted release. The review queues, charts, and map work offline. Official notice and project links open the publishers' websites. |
| Rerun in Google Colab | Open **Bantay_Kontrata_Tagum.ipynb**, select **Runtime → Run all**, and upload **Bantay_Kontrata_Tagum.zip** when prompted. Download the rebuilt files from the **tagum_review** folder. |
| Rerun on a computer | Use Python 3.12 from the project root. See the commands below. |
| Read the methodology | Open **docs/Bantay_Kontrata_Tagum_Guide.pdf**. |

    python -m pip install -r requirements.txt
    python publish_dashboard.py
    python -m unittest discover -s tests -v

## Repository handoff

The release archive has a top-level `Bantay_Kontrata_Tagum` folder. Use this folder as the project root after extraction. Keep the relative paths so the scripts and notebook can find the data and templates.

To create the Colab archive from a committed checkout, run this from the repository root.

    git -c core.autocrlf=false archive --format=zip --prefix=Bantay_Kontrata_Tagum/ --output=../Bantay_Kontrata_Tagum.zip HEAD

| Path | Contents |
|---|---|
| `data/` | Pipeline inputs and source provenance |
| `data/supplemental/Bantay_Kontrata_Additional_Datasets.zip` | The complete standalone dataset supplement, including its own manifest and coverage notes |
| `review/` | The ready-to-open dashboard, ranked tables, crosswalks, and validation report |
| `tests/` | Reconciliation, model, and refresh tests |
| `docs/` | The field guide and the original research brief under `docs/reference/` |

Use the commands above from the repository root. The field guide also works with this archive wherever it refers to the original project ZIP. The supplied Git ignore rules cover local environment files, generated working copies, and reviewer exports.

## Review a real case

1. Open **Award review** or **Tender review** and select a record. Use **Above cutoff** to focus on model flags. Clear it to examine records below the cutoff.
2. When **What to verify first** appears, use it to see the input that contributed most toward a higher score and the documents to check. Open the PhilGEPS notice when linked. Compare the buyer, bid ID, mode, budget, and dates. Check amendments and individual award documents before drawing conclusions. If an old notice cannot be opened, use the bid ID or solicitation number to locate a copy. For archive-only awards, also record the archive record ID and any available local contract number.
3. Open **Record review**, choose a status, and record the evidence checked or missing. **Reviewed** and **Follow up** require reviewer initials, a reason, and an HTTPS evidence link or an internal document reference. Drafts follow the case when you switch records in the open tab. Save the review before closing the tab.
4. Use **Saved reviews** to reopen a case or export all recorded decisions to CSV. Download a JSON backup at the end of a session. Keep the matching dashboard HTML with it. Use **Import backup** on that dashboard to resume on another browser or device. Imports add missing reviews. If two versions of a case differ, open the conflict to compare the status, reviewer, evidence, and notes before choosing a version. Keep the source backup and download a copy of the current reviews before replacing any decision you need to retain.

In **Tender review**, **Other notices with this solicitation** shows related scored notices, including those below the cutoff. Compare their titles, dates, budgets, and snapshot statuses. Open the original notices and check the agency records to establish whether they are reposts, revisions, or separate lots. Each bid keeps its own review. For newer postings and awards, use the [modernized PhilGEPS Bulletin Board](https://philgeps.gov.ph/Indexes/index).

| Review status | Use |
|---|---|
| In review | A reviewer has started examining the case. |
| Needs documents | A specific notice, bid bulletin, award document, or other record still needs to be obtained. |
| Reviewed | The available evidence and the reviewer's conclusion are recorded. |
| Follow up | The evidence suggests a specific question for further review. Enter the question and the supporting document. |

Reviews stay in the browser profile and use the device clock. On a browser that blocks local file storage, the dashboard shows a session-only message and still supports export and restore. Keep backup files with the project, in the same access-controlled location as any internal notes. Open source links only when connected. A model flag is a lead for review. Formal findings require the underlying documents and the office's review process.

For an agency pilot, name a custodian for the working dashboard and backups. Save a dated copy of the reviewer's usual picks before opening the evidence. Run the screen alongside the existing review process, using that process for referrals and official case records. Review cases above and below the cutoff against the underlying documents. Record document availability, time per case, the planned next document check, and evidence-backed follow-up in both groups. Agree on what would justify continued use before reviewing the sample.

## How XAI helps reviewers

Explainable AI (XAI) connects a case’s score to a document check. TreeSHAP shows which inputs pushed the Isolation Forest score up or down. The dashboard shows the strongest push toward a higher score alongside the underlying value or peer comparison and a suggested source check. Budget differences point to the notice and amendments. Prior awards to the same vendor point to related award records or purchase orders.

In Tagum bid **12883096**, the snapshot shows a **25-day bid window** and an **11-day median** among recent tenders in the same mode. The bid window made the largest contribution toward a higher score. A reviewer can start by confirming the published and closing dates and checking any deadline extensions. The peer median describes the model’s comparison group. [PhilGEPS notice 12883096](https://notices.philgeps.gov.ph/GEPSNONPILOT/Tender/PrintableBidNoticeAbstractUI.aspx?refid=12883096)

During the pilot, save the blind review results before showing the explanations. Revisit selected cases and record the original document check, the explanation shown, any check it added or corrected, the document involved, and the extra minutes. Record whether an inspected document supported the change or the reviewer still needs it. This measures how often explanations change the review plan and how often staff can support those changes with evidence.

## What the counts mean

| Dataset or linkage | Count | Unit |
|---|---:|---|
| Historical award archive, 2017–2025 | 11,305 | Award rows after the archive's observation-level duplicate removal |
| Official award table, 2024–2025 | 3,193 | Distinct official numeric award IDs |
| Accepted archive-to-official links | 2,198 | One-to-one links, including 1,395 in 2024 and 803 in 2025 |
| Combined award register | 12,300 | Retained rows after accepted matches were combined |
| Possible overlaps | 326 | Register rows involved in 1,765 candidate pairs held for review |
| Official tenders, 2024–September 2026 | 3,395 | Distinct official numeric bid IDs from 5,842 source rows |
| Infrastructure | 336 | DPWH contract records, with 235 mapped points and 10 flood-map matches |

The award count is **11,305 + 3,193 − 2,198 = 12,300 register rows**. The merge requires the same date, amount to the cent, normalized awardee, and a matching normalized title. It accepts only unique matches in both directions. It preserves uncertain rows and exposes their candidates separately. The archive's 74 populated 2025 values in **reference_id** match its local contract numbers. The official numeric identifiers have separate fields in the integrated outputs.

The official award table ends September 29, 2025, and the tenders extend through September 26, 2026. The 2026 extracts provide solicitation numbers and notice information for Tagum. The award fields for those records lack dated outcomes. The packaged provenance lists eleven official files, their URLs, and their SHA-256 hashes. [Official PS-PhilGEPS releases](https://philgeps.gov.ph/CmsHomePages/open-data) · [Historical award archive](https://huggingface.co/datasets/bettergovph/philgeps-data)

## Model periods and inputs

| Screen | Warm-up | Training | Calibration | Held-out evaluation |
|---|---|---|---|---|
| Historical awards | 2017, 603 rows | 2018–2023, 7,807 | 2024, 1,578 | 2025, 1,317 |
| Official tenders | Jan–Mar 2024, 302 rows | Apr–Dec 2024, 1,101 | 2025, 1,476 | 2026 through September, 516 |

The award model uses the amount relative to the recent category median, the vendor's share of 365-day city awards, the vendor's same-category awards over 30 days, prior category volume, and days to quarter end. For thin categories, the amount comparison uses the citywide median.

The tender model uses budget relative to its recent peers, bid-window length relative to its recent peers, recent mode share, recent classification share, and days to quarter end. Peer calculations use the previous 365 days and exclude all notices published on the same day. Budget peers first use the same mode and classification, then the same mode, then the citywide history. Window peers use the same mode and fall back to the citywide history. A specific peer group needs at least 15 prior rows. Small Value Procurement labels are grouped for statistical comparison. The dashboard displays the publisher's mode labels. The notification and award outcomes appear as evidence fields. The screens are retrospective because the source snapshots can contain later amendments.

Both screens fit Isolation Forest and Extended Isolation Forest on their training periods. Each detector sets its own 95th-percentile cutoff on calibration scores, using the higher order statistic. An evaluation score must exceed that cutoff to enter the review queue. [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17) · [Extended Isolation Forest](https://arxiv.org/abs/1811.02141)

## Review and validation

The award screen flagged **106 of 1,317** historical 2025 rows. Of those 106, 63 link to an official award and 43 link to a tender with source fields. The tender screen flagged **48 of 516** held-out tenders. Matching official evidence occurs independently of the historical model's scoring.

The 516 evaluated tender notices include 40 solicitation numbers shared by multiple notices. Keeping only the latest prior notice for each known solicitation in each 365-day peer window changed one Isolation Forest flag. Both runs flagged 48 notices, with 47 in common. All 2,879 earlier tender rows lack solicitation numbers, so the training and calibration inputs stayed the same. The dashboard uses the published per-notice scores. Confirm reposts and separate lots in agency records before changing how notices count toward the peer comparison.

TreeSHAP explains the standard forest's average isolation path length. Its negative contributions shorten the path. Local-DIFFI measures feature use on shorter paths. ExIFFI reports feature importance for the model with oblique splits. In the what-if panel, the amount or window changes within the observed training peer values and the remaining model inputs are held fixed. [TreeSHAP](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html) · [Local-DIFFI](https://arxiv.org/abs/2007.11117) · [ExIFFI](https://arxiv.org/abs/2310.05468)

The synthetic experiments measure how the ranking responds to controlled amount, vendor history, budget, and window changes. They compare both forests with a univariate baseline and record average precision, recall at the fixed cutoff, and paired score wins. The baseline leads the amount-only and budget-only changes. The forests lead the vendor-history scenario by average precision. The extended forest leads the bid-window-only scenario by average precision. Real audit performance requires record-level labels linked to underlying documents.

## Key outputs and fields

| File | Use |
|---|---|
| **review/tagum_award_register.csv** | All retained award rows with source, official IDs where available, and a possible-overlap marker |
| **review/tagum_award_links.csv** | Accepted archive-to-official pairs, match evidence, and any linked tender's details |
| **review/tagum_possible_award_overlaps.csv** | Review candidates that share the date, amount and vendor, or compete for an exact match |
| **review/tagum_2025_ranked_awards.csv** | Award model rank, scores, local archive contract number, and official tender evidence |
| **review/tagum_2026_ranked_tenders.csv** | Tender model rank, scores, published fields, and peer statistics |
| **review/validation.json** | Coverage counts, stress experiments, path reconstruction, refits, and what-if results |
| **data/official/** | Official extracts and provenance, verified abstract samples, and the DPWH flood crosswalk |
| **tests/test_project.py** | Matching, publication-time context, source counts, tender-level award totals, DPWH links, scores, and TreeSHAP checks |

The **bid_reference** identifies an official tender. The **official_award_reference** identifies one official award. The **record_id** identifies a retained record in the integrated register. **observed_official_award_total_php** sums distinct award IDs linked to a bid in the supplied files. Compare it with the whole tender's approved budget. A single award can represent part of a tender. Bidder counts require an Abstract of Bids or another document that explicitly records bids received.

The documented pipeline was rebuilt from the included files. The notebook in a clean directory reproduced the command-line output byte for byte. The browser checks exercised the award and tender filters, links, explanations and sliders, validation, and the DPWH map on desktop and mobile layouts.

## Stage a new official release

Download a quarterly CSV as published, and record its source page. The command below demonstrates reprocessing the included public 2026 Q3 release. For a new release, use its local filename, release label, and source page, and give the staged sources and dashboard their own new directories. [PS-PhilGEPS downloads](https://philgeps.gov.ph/CmsHomePages/open-data)

    python refresh_official.py --input DATASETS_PHILGEPS_2026_JULY-SEPTEMBER.csv --release 2026Q3 --source-page https://drive.google.com/file/d/1enMAqo1VWFwnf01iHPPw9sjNnwjlTWbt/view --output staged-2026Q3
    python publish_dashboard.py --data staged-2026Q3 --output review-2026Q3

Use `--base-data staged-2026Q3` to stage a later release on top of that copy. Matching quarter labels replace those source rows in the new snapshot. The refresh keeps the fixed historical award and DPWH sources, scopes the new lines to the City of Tagum buyer, records source record numbers and file hashes, and rebuilds the official award and tender tables. It stops before publishing a new source folder when core tender fields conflict or source data fail validation. Open the rebuilt **Validation** tab and compare coverage and review counts with the previous snapshot before using it.

The backup ID depends on the dashboard data and model results. A rebuild in another environment can produce a new ID because of floating-point rounding. Retain the dashboard HTML with its CSV exports and JSON backups so you can reopen prior reviews. The current intake is quarterly and uses release snapshots. Review by an upcoming bid deadline should use the current notice and amendments from the publisher.

## Pilot verification on November 16, 2026

| Notices sampled | Bid IDs | Result |
|---|---|---|
| Award screen, above cutoff | 12022987, 11622916, 12107378 | All three matched the printed notice fields |
| Award screen, below cutoff | 11797139, 11743193, 11969783 | All three matched the printed notice fields |
| Tender screen, above cutoff | 13103239, 13249746, 13157991 | All three matched the printed notice fields |
| Tender screen, below cutoff | 13092850, 13023417, 12765129 | All three matched the printed notice fields |

The checks compared the bid ID, buyer, budget, mode, publication date, and closing date. I chose the first, middle, and last ranked linked notice in each group. Bid 13157991 showed **Awarded** on the portal and **Closed** in the extract. The award amount for that notice still requires the award document. This check measures the usability of source links and displayed fields for the sample. [Official bid 13157991](https://notices.philgeps.gov.ph/GEPSNONPILOT/Tender/PrintableBidNoticeAbstractUI.aspx?refid=13157991)

Browser tests verified offline award and tender review, saved decisions after a browser restart, CSV export, JSON restore in a separate mobile browser profile, rejection of another snapshot's backup, and a session backup when storage was blocked. Seven automated tests passed. Reprocessing the official Q3 CSV reproduced the three source tables byte for byte. Real audit effectiveness still needs independently reviewed cases.
