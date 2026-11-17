"""Reconcile the City of Tagum archive and official quarterly extracts."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parent
TENDER_LOOKUP = (
    "https://notices.philgeps.gov.ph/GEPSNONPILOT/Tender/"
    "PrintableBidNoticeAbstractUI.aspx?refid="
)
BUYER = "CITY OF TAGUM, DAVAO DEL NORTE"

ARCHIVE_COLUMNS = [
    "record_id", "award_date", "contract_no", "notice_title", "award_title", "awardee_name",
    "contract_amount", "business_category",
]
TENDER_EVIDENCE = [
    "bid_reference", "approved_budget_php", "published_date", "closing_date", "procurement_mode",
    "bid_notice_status", "official_notice_link", "observed_official_awards",
    "observed_official_award_total_php", "observed_award_total_to_abc",
]
OFFICIAL_EVIDENCE = [
    "official_award_reference", "bid_reference", "award_published_date", "official_procurement_mode",
    "published_date", "closing_date", "procurement_mode", "approved_budget_php", "notice_window_days",
    "bid_notice_status", "official_notice_link", "observed_official_awards",
    "observed_official_award_total_php", "observed_award_total_to_abc",
]


@dataclass
class IntegratedData:
    tenders: pd.DataFrame
    official_awards: pd.DataFrame
    award_register: pd.DataFrame
    award_links: pd.DataFrame
    unresolved_links: pd.DataFrame
    flood_links: pd.DataFrame
    counts: dict


def read_csv(path):
    return pd.read_csv(path, dtype="string", keep_default_na=False)


def require_unique(frame, key, label):
    if frame[key].isna().any() or frame[key].str.strip().eq("").any() or frame[key].duplicated().any():
        raise ValueError(f"{label} requires a present, unique {key}.")


def cents(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        number = Decimal(str(value).replace(",", ""))
        return int(number.quantize(Decimal("0.01")) * 100) if number.is_finite() else None
    except InvalidOperation:
        return None


def normalized(value):
    if pd.isna(value):
        return ""
    return "".join(
        char for char in unicodedata.normalize("NFKC", str(value)).casefold() if char.isalnum()
    )


def titles(record):
    return {normalized(record.notice_title), normalized(record.award_title)} - {""}


def reconcile_awards(archive, official):
    """Match exact core fields, then require a shared title and uniqueness in both directions."""
    candidates = defaultdict(list)
    for row in official.itertuples(index=False):
        key = (row.award_date, cents(row.contract_amount_php), normalized(row.awardee))
        if all(key):
            candidates[key].append((row, titles(row)))

    possible = []
    for row in archive.itertuples(index=False):
        key = (row.award_date, cents(row.contract_amount), normalized(row.awardee_name))
        if not all(key):
            continue
        same_core = candidates.get(key, [])
        archive_titles = titles(row)
        for candidate, official_titles in same_core:
            possible.append({
                "record_id": row.record_id,
                "official_award_reference": candidate.award_reference,
                "award_date": row.award_date,
                "contract_amount_php": key[1] / 100,
                "title_agrees": bool(archive_titles & official_titles),
                "archive_notice_title": row.notice_title,
                "official_notice_title": candidate.notice_title,
            })

    columns = [
        "record_id", "official_award_reference", "award_date", "contract_amount_php",
        "title_agrees", "archive_notice_title", "official_notice_title",
    ]
    possible = pd.DataFrame(possible, columns=columns)
    exact = possible.loc[possible.title_agrees.eq(True)]
    accepted = exact.loc[
        ~exact.record_id.duplicated(keep=False)
        & ~exact.official_award_reference.duplicated(keep=False)
    ]
    links = accepted[
        ["record_id", "official_award_reference", "award_date", "contract_amount_php"]
    ].assign(evidence="Same award date, peso amount, awardee and normalized title")
    unresolved = possible.loc[
        ~possible.record_id.isin(links.record_id)
        & ~possible.official_award_reference.isin(links.official_award_reference)
    ].copy()
    unresolved["reason"] = unresolved.title_agrees.map({
        True: "Multiple records share exact matching fields",
        False: "Shared award date, amount and awardee with a different title",
    })
    return links, unresolved


def tender_award_totals(official):
    approved = official.loc[official.bid_reference.ne(""), ["bid_reference", "contract_amount_php"]].copy()
    approved["amount_cents"] = approved.contract_amount_php.map(cents)
    sums = approved.groupby("bid_reference").amount_cents.agg(["count", "sum"]).reset_index()
    return sums.assign(observed_official_award_total_php=sums["sum"] / 100).rename(
        columns={"count": "observed_official_awards"}
    ).drop(columns="sum")


def build_integrated_data(data_dir=ROOT / "data"):
    official_dir = data_dir / "official"
    archive = read_csv(data_dir / "tagum_awards.csv")
    official = read_csv(official_dir / "tagum_philgeps_awards.csv")
    tenders = read_csv(official_dir / "tagum_philgeps_tenders.csv")
    flood = read_csv(official_dir / "tagum_dpwh_flood_matches.csv")
    require_unique(archive, "record_id", "Historical archive")
    require_unique(official, "award_reference", "Official awards")
    require_unique(tenders, "bid_reference", "Official tenders")
    require_unique(flood, "dpwh_contract_id", "DPWH flood mapping")

    for frame, field in [(official, "award_reference"), (tenders, "bid_reference")]:
        if not frame[field].str.fullmatch(r"[1-9][0-9]*").all() or not frame.buyer.eq(BUYER).all():
            raise ValueError("Official records require their published positive numeric ID and the verified buyer.")
    if not official.loc[official.bid_reference.ne(""), "bid_reference"].isin(tenders.bid_reference).all():
        raise ValueError("The official award table contains a bid reference outside the tender extract.")

    tenders["approved_budget_php"] = pd.to_numeric(tenders.approved_budget_php, errors="raise")
    tenders["notice_window_days"] = (
        pd.to_datetime(tenders.closing_date, format="%Y-%m-%d", errors="raise")
        - pd.to_datetime(tenders.published_date, format="%Y-%m-%d", errors="raise")
    ).dt.days
    if not (tenders.approved_budget_php.gt(0) & tenders.notice_window_days.gt(0)).all():
        raise ValueError("The official tender inputs require a positive ABC and a positive date window.")
    # Construct the public lookup from a validated numeric ID rather than source text.
    tenders["official_notice_link"] = TENDER_LOOKUP + tenders.bid_reference
    tenders = tenders.merge(tender_award_totals(official), on="bid_reference", how="left", validate="one_to_one")
    tenders["observed_official_awards"] = tenders.observed_official_awards.fillna(0).astype(int)
    tenders["observed_award_total_to_abc"] = tenders.observed_official_award_total_php / tenders.approved_budget_php

    links, unresolved = reconcile_awards(archive, official)
    if links.empty:
        raise ValueError("There are no reconciled awards. Verify the two source snapshots.")
    official = official.rename(columns={
        "award_reference": "official_award_reference",
        "approved_budget_php": "award_export_abc_php",
        "procurement_mode": "official_procurement_mode",
    })
    evidence = official.merge(
        tenders[TENDER_EVIDENCE + ["notice_window_days"]],
        on="bid_reference", how="left", validate="many_to_one",
    )
    links = links.merge(
        evidence[OFFICIAL_EVIDENCE], on="official_award_reference", how="left", validate="one_to_one",
    )

    history = archive[ARCHIVE_COLUMNS].merge(
        links[["record_id", *OFFICIAL_EVIDENCE]], on="record_id", how="left", validate="one_to_one",
    )
    history["source"] = history.official_award_reference.notna().map({
        True: "Historical archive + official PhilGEPS", False: "Historical archive",
    })
    history["possible_overlap"] = history.record_id.isin(unresolved.record_id)

    supplemental = evidence.loc[
        ~evidence.official_award_reference.isin(links.official_award_reference)
    ].copy()
    supplemental["record_id"] = "philgeps-award:" + supplemental.official_award_reference
    supplemental["contract_no"] = ""
    supplemental["source"] = "Official PhilGEPS"
    supplemental["possible_overlap"] = supplemental.official_award_reference.isin(unresolved.official_award_reference)
    supplemental = supplemental.rename(columns={
        "awardee": "awardee_name", "contract_amount_php": "contract_amount",
    })
    register_fields = ARCHIVE_COLUMNS + ["source", "possible_overlap", *OFFICIAL_EVIDENCE]
    register = pd.concat([history[register_fields], supplemental[register_fields]], ignore_index=True)
    register["contract_amount"] = pd.to_numeric(register.contract_amount, errors="raise")
    register = register.sort_values(["award_date", "record_id"], ignore_index=True)
    require_unique(register, "record_id", "Integrated award register")
    if len(register) != len(archive) + len(official) - len(links):
        raise AssertionError("The merged award count must reconcile to its source and link counts.")

    archive_counts = Counter(archive.award_date.str[:4])
    award_counts = Counter(official.award_date.str[:4])
    link_counts = Counter(links.award_date.str[:4])
    tender_counts = Counter(tenders.published_date.str[:4])
    counts = {
        "archive_awards": len(archive),
        "official_awards": len(official),
        "official_tenders": len(tenders),
        "linked_awards": len(links),
        "award_register_rows": len(register),
        "unresolved_pairs": len(unresolved),
        "unresolved_register_rows": int(register.possible_overlap.sum()),
        "official_awards_with_tender": int(official.bid_reference.ne("").sum()),
        "latest_official_award_date": official.award_date.max(),
        "latest_tender_publication_date": tenders.published_date.max(),
        "flood_links": len(flood),
        "annual": [{
            "year": year,
            "archive_awards": archive_counts[year],
            "official_awards": award_counts[year],
            "matches": link_counts[year],
            "award_register_rows": archive_counts[year] + award_counts[year] - link_counts[year],
            "official_tenders": tender_counts[year],
        } for year in sorted(archive_counts.keys() | award_counts.keys() | tender_counts.keys())],
    }
    return IntegratedData(tenders, official, register, links, unresolved, flood, counts)
