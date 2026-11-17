# Additional datasets for Bantay-Kontrata Tagum

This directory contains official PS-PhilGEPS quarterly exports filtered to **CITY OF TAGUM, DAVAO DEL NORTE**. It includes rows from eleven releases spanning 2024 Q1 through 2026 Q3. The quarter in a filename identifies the release, and the records can span more than one quarter. **provenance.json** lists the publisher, download pages, source filenames, checksums, and extraction counts. [PS-PhilGEPS Open Data](https://philgeps.gov.ph/CmsHomePages/open-data)

| Event year | Distinct bid references, by publication date | Distinct award references with a positive contract amount, by award date |
|---|---:|---:|
| 2024 | 1,403 | 1,871 |
| 2025 | 1,476 | 1,322 |
| 2026 | 516 | 0 |

All 3,395 tenders in the extract have a positive reported approved budget for the contract (ABC), a publication date, a closing date, and a procurement mode. This includes 2,879 tenders dated 2024–2025. The last recorded award date in this extract is **2025-09-29**. The 2025 Q4 release has 399 rows for this buyer with blank award dates. The 2026 releases contain tender details for Tagum, including solicitation numbers and the reported procurement rules, with blank award dates. This measures what the extracts supplied at download. Later award activity requires a separate source.

## Files

| File | Unit and use |
|---|---|
| **tagum_philgeps_line_items.csv** | 5,842 scoped source rows. A bid or award reference may repeat across line items and releases. The source release and record number trace each row. |
| **tagum_philgeps_tenders.csv** | One row per positive official bid reference with a readable publication date in 2024–2026. Contains ABC, procurement mode, publication and closing dates, and the notice status when the releases agree. |
| **tagum_philgeps_awards.csv** | One row per positive official award reference with a readable award date in the coverage period and a positive reported amount. Contains linked bid reference when present, awardee, reported contract amount, and the administrative contract dates supplied by the publisher. |
| **tagum_2025_prototype_award_links.csv** | 803 matches to the existing 2025 archive. The original record ID, official award reference, bid reference when available, and match evidence are recorded on each row. |
| **tagum_philgeps_verified_abstracts.csv** | 11 official Tagum bid abstracts independently opened and checked. Records the official URL, purchase request, posted budget, dates, and reported mode. |
| **tagum_verified_abstract_award_links.csv** | 14 prototype award rows matched to the sampled abstracts using buyer, purchase request, normalized title, and the award date relative to the portal close. |
| **tagum_dpwh_flood_matches.csv** | Ten flood-map contracts naming Tagum City, linked to existing infrastructure rows by DPWH contract ID. Includes the project and component identifiers, coordinates, and reported contract figures. |
| **coverage.json** | Counts of unique IDs and field coverage. |

## Join to the existing project

The previous archive's 74 filled 2025 values under **reference_id** all equal its local **contract_no** field. For these awards, the official numeric identifiers come from the new exports. The link table matches the buyer, exact award date and amount, and normalized awardee and title. It keeps unique matches in both directions. Of the 1,317 archive rows, **803** match an official award reference and **689** also carry a bid reference. This includes **63** of the 106 review flags, with **43** of those linked to a tender. To attach a tender, use the numeric **bid_reference** in the new award and tender tables.

A purchase request sometimes has more than one notice, and a tender sometimes has multiple award rows. Preserve the award and bid IDs when comparing tender ABC with award amounts. For the Pagsabangan evacuation site, tender **11424537** reports an ABC of **₱1,026,667**. Two matched archive awards total **₱1,023,136**. The extract leaves a tender field blank and names it in **fields_with_conflicting_values** when multiple distinct nonblank values were published for that ID. [Official Pagsabangan notice](https://notices.philgeps.gov.ph/GEPSNONPILOT/Tender/PrintableBidNoticeAbstractUI.aspx?refid=11424537)

The award table has 662 rows without a bid reference. In 658 of those, the reported award-level ABC exactly equals the contract amount. The tender table provides the traceable budget comparison when a bid reference is available. The exports omit bidder counts. Use a bid abstract or equivalent record that reports bids received for any bidder-count analysis. [Example DPWH BAC resolution that states the bids received](https://www.dpwh.gov.ph/dpwh/sites/default/files/webform/civil_works/notices/25lc0022_cancellation.pdf)

## Additional primary sources

- **DPWH Post-Contract Award Disclosures.** Check engineering contract IDs against the district office reports for the amount at award, notice of award and notice to proceed. Compare dates and definitions before using those fields alongside the infrastructure portal. The flood-map extract in this package is a public mirror and includes the project/component identifiers. [Davao del Norte DEO February 2022](https://www.dpwh.gov.ph/dpwh/sites/default/files/webform/civil_works/pcad/pdf/POST-CONTRACT%20AWARD%20DISCLOSURES%20%28FEBRUARY%201-28%2C%202022%29.pdf) · [December 2024](https://www.dpwh.gov.ph/DPWH/sites/default/files/webform/civil_works/pcad/pdf/pcad_summary_report_december_2024.pdf) · [Flood-map data mirror](https://github.com/rukku/sumbongsapangulo.ph-datasets)
- **City of Tagum FY2026 Final Annual Procurement Plan.** Lists planned procurements, estimated budgets, planned modes, funding sources, end-user offices and schedules. Use the plan for planned-versus-recorded comparisons after resolving the line items. [Official FY2026 plan](https://tagumcity.gov.ph/wp-content/uploads/2026/01/NGPA_Final-APP-2026.pdf)
- **COA Tagum compliance audits.** The 2024 listing covers use of the Special Education Fund, and the 2025 listing covers allocation of the disaster risk reduction and management fund. These are audit leads. Any record-level label requires matching a specific finding to the underlying procurement documents. [2024 COA listing](https://www.coa.gov.ph/wpfd_file/tagum-city-compliance-audit-report-2024/) · [2025 COA listing](https://www.coa.gov.ph/wpfd_file/tagum-city-compliance-audit-report-2025/)

The line-item file uses the publisher's selected values and records their source release. The older **Funding Source** and newer **Source of Funds** occupy the same output column. The same applies to **Area of Delivery** and **Area of Delivery/Project Location**. The identifier-level files parse day/month/year dates and positive amounts, and treat zero or blank identifiers as missing. 10 selected cells across 2025 Q4 and the 2026 releases required decoding non-UTF-8 bytes using Windows-1252. The eleven CSV downloads are reproducible from their publisher links and file digests.
