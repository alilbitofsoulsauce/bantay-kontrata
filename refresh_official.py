"""Stage a new or corrected PS-PhilGEPS quarterly CSV for the Tagum pilot."""

import argparse
from collections import Counter
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from hashlib import file_digest
import json
from pathlib import Path
import re
from shutil import copy2
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import pandas as pd

from integration import BUYER, ROOT, TENDER_LOOKUP, build_integrated_data, read_csv


COLUMNS = {
    "Procuring Entity (PE)": "buyer",
    "Region": "buyer_region",
    "Province": "buyer_province",
    "City/Municipality": "buyer_city",
    "Bid Reference No.": "bid_reference",
    "Solicitation No.": "solicitation_number",
    "Notice Title": "notice_title",
    "Classification": "classification",
    "Procurement Mode": "procurement_mode",
    "Business Category": "business_category",
    "Procurement Category": "procurement_category",
    "Funding Source": "funding_source",
    "Funding Instrument": "funding_instrument",
    "Applicable Procurement Rules": "applicable_procurement_rules",
    "Approved Budget of the Contract": "approved_budget_php",
    "Published Date": "published_date",
    "Closing Date": "closing_date",
    "Area of Delivery": "area_of_delivery",
    "Contract Duration": "contract_duration",
    "Delivery Period": "delivery_period",
    "Calendar Type": "calendar_type",
    "Line Item No": "line_item_number",
    "Item Name": "item_name",
    "Quantity": "quantity",
    "UOM": "unit_of_measure",
    "Item Budget": "item_budget_php",
    "Bid Notice Status": "bid_notice_status",
    "Award Reference No.": "award_reference",
    "Award Title": "award_title",
    "Published Date(Award)": "award_published_date",
    "Award Date": "award_date",
    "Contract Amount": "contract_amount_php",
    "Award Notice Status": "award_notice_status",
    "Notice to Proceed Date": "notice_to_proceed_date",
    "Contract Effectivity Date": "contract_effectivity_date",
    "Contract End Date": "contract_end_date",
    "Awardee Organization Name": "awardee",
}
HEADER_ALIASES = {
    "Funding Source": "Source of Funds",
    "Area of Delivery": "Area of Delivery/Project Location",
}
OPTIONAL_COLUMNS = {
    "Solicitation No.", "Business Category", "Procurement Category", "Applicable Procurement Rules",
    "Contract Duration", "Delivery Period", "Calendar Type",
}

TENDER_FIELDS = [
    "buyer", "solicitation_number", "notice_title", "classification", "business_category",
    "procurement_category", "procurement_mode", "funding_source", "funding_instrument",
    "applicable_procurement_rules", "approved_budget_php", "published_date", "closing_date",
    "area_of_delivery", "contract_duration", "delivery_period", "calendar_type", "bid_notice_status",
]
AWARD_FIELDS = [
    "buyer", "bid_reference", "notice_title", "award_title", "awardee", "award_date",
    "award_published_date", "contract_amount_php", "approved_budget_php", "classification",
    "business_category", "procurement_mode", "award_notice_status", "notice_to_proceed_date",
    "contract_effectivity_date", "contract_end_date",
]
NULLS = {"", "null", "na", "n/a", "-"}


def clean(value):
    if pd.isna(value):
        return ""
    result = str(value).strip()
    return "" if result.casefold() in NULLS else result


def number_id(value):
    value = clean(value)
    return str(int(value)) if value.isascii() and value.isdecimal() and int(value) > 0 else ""


def money(value):
    value = clean(value).replace(",", "").replace("₱", "").strip()
    if not value:
        return ""
    try:
        pay = Decimal(value)
        if pay.is_finite() and pay > 0:
            return str(pay.quantize(Decimal("0.01")))
    except InvalidOperation:
        pass
    return ""


def date_from_source(raw):
    value = clean(raw)
    match = re.match(r"^(\d{1,2}/\d{1,2}/\d{4})(?=$|\s)", value)
    try:
        return datetime.strptime(match.group(1), "%d/%m/%Y").date().isoformat() if match else ""
    except ValueError:
        return ""


def normalized_fields(raw):
    normalized = raw.map(clean)
    for field in ("bid_reference", "award_reference"):
        normalized[field] = normalized[field].map(number_id)
    for field in ("approved_budget_php", "contract_amount_php"):
        normalized[field] = normalized[field].map(money)
    for field in (
        "published_date", "closing_date", "award_date", "award_published_date",
        "notice_to_proceed_date", "contract_effectivity_date", "contract_end_date",
    ):
        normalized[field] = normalized[field].map(date_from_source)
    return normalized


def summarize(rows, key, fields):
    collected = []
    for label, group in rows.loc[rows[key].ne("")].groupby(key, sort=True):
        record = {key: label}
        conflicts = []
        for field in fields:
            values = sorted(set(group[field]) - {""})
            if len(values) > 1:
                conflicts.append(field)
            record[field] = values[0] if len(values) == 1 else ""
        record["source_releases"] = "|".join(sorted(set(group.source_release)))
        record["source_line_item_count"] = len(group)
        record["fields_with_conflicting_values"] = "|".join(conflicts)
        collected.append(record)
    return pd.DataFrame(collected, columns=[
        key, *fields, "source_releases", "source_line_item_count", "fields_with_conflicting_values",
    ])


INVALID_UTF8 = re.compile(r"[\udc80-\udcff]+")
LINE_FILE = "tagum_philgeps_line_items.csv"
COVERAGE_START = "2024-01-01"
SNAPSHOT_INPUTS = ("tagum_awards.csv", "tagum_barangays.geojson", "tagum_dpwh.csv", "provenance.json")


def source_release(path, quarter, source_page):
    """Read only the verified buyer's rows, retaining their source record numbers."""
    with path.open("rb") as handle:
        digest = file_digest(handle, "sha256").hexdigest()
    with path.open(encoding="utf-8-sig", errors="surrogateescape", newline="") as source:
        first = source.readline()
        delimiter = "\t" if first.count("\t") > first.count(",") else ","
        source.seek(0)
        reader = csv.reader(source, delimiter=delimiter)
        header = next(reader, [])
        positions = []
        for field in COLUMNS:
            source_field = field if field in header else HEADER_ALIASES.get(field)
            if source_field in header:
                positions.append(header.index(source_field))
            elif field in OPTIONAL_COLUMNS:
                positions.append(None)
            else:
                raise ValueError(f"{quarter} is missing a required column: {field}.")
        buyer_index = positions[0]
        stats = Counter()
        rows = []
        for record_number, record in enumerate(reader, 1):
            stats["national_records"] += 1
            if len(record) != len(header):
                stats["wrong_number_of_columns"] += 1
                if len(record) > buyer_index and record[buyer_index].strip().upper() == BUYER:
                    raise ValueError(f"{quarter} source record {record_number} has an invalid column count.")
                continue
            if record[buyer_index].strip().upper() != BUYER:
                continue
            cells = [record[j].strip() if j is not None else "" for j in positions]
            for i, value in enumerate(cells):
                if INVALID_UTF8.search(value):
                    stats["tagum_cells_with_windows_encoding_fallback"] += 1
                    cells[i] = INVALID_UTF8.sub(
                        lambda match: match[0].encode("utf-8", "surrogateescape").decode("cp1252", errors="replace"),
                        value,
                    )
            rows.append([quarter, str(record_number), *cells])
    if not rows:
        raise ValueError(f"{quarter} contains no records for {BUYER}. Check the downloaded release.")
    columns = ["source_release", "source_record_number", *COLUMNS.values()]
    stats["tagum_line_item_rows"] = len(rows)
    metadata = {
        "release": quarter, "filename": path.name, "download_page": source_page,
        "source_sha256": digest, "source_bytes": path.stat().st_size, "raw_field_count": len(header),
        "character_encoding": "UTF-8 with Windows-1252 fallback for invalid UTF-8 bytes in selected cells",
        "column_separator": "tab" if delimiter == "\t" else "comma", "count": dict(stats),
    }
    return pd.DataFrame(rows, columns=columns), metadata


def identifier_tables(line_items):
    ready = normalized_fields(line_items)
    tenders = summarize(ready, "bid_reference", TENDER_FIELDS)
    scoring_fields = {"published_date", "closing_date", "approved_budget_php", "procurement_mode", "classification"}
    conflict = tenders.fields_with_conflicting_values.map(lambda fields: bool(set(fields.split("|")) & scoring_fields))
    if conflict.any():
        examples = ", ".join(tenders.loc[conflict, "bid_reference"].head(6))
        raise ValueError(f"Resolve conflicting scoring fields in {int(conflict.sum())} tenders. Bid IDs include {examples}.")
    tenders = tenders.loc[tenders.published_date.ge(COVERAGE_START)].copy()
    if tenders[["procurement_mode", "classification"]].eq("").any().any():
        raise ValueError("A tender with a published bid ID and date is missing its mode or classification.")
    tenders["official_notice_link"] = TENDER_LOOKUP + tenders.bid_reference
    awards = summarize(ready, "award_reference", AWARD_FIELDS)
    awards = awards.loc[awards.award_date.ge(COVERAGE_START) & awards.contract_amount_php.ne("")].copy()
    return tenders, awards


def write_csv(frame, path):
    safe = frame.copy()
    for field in safe.select_dtypes(include=["string", "object"]):
        safe[field] = safe[field].str.replace(r"^(?=\s*[=+\-@\t\r\n])", "'", regex=True)
    safe.to_csv(path, index=False)


def write_snapshot(base_data, source_path, quarter, source_page, destination):
    if not re.fullmatch(r"20\d{2}Q[1-4]", quarter):
        raise ValueError("Use a release label such as 2026Q3.")
    parsed_url = urlparse(source_page)
    if parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.username or parsed_url.password:
        raise ValueError("Supply the public HTTPS download page for this release.")
    if destination.exists():
        raise ValueError(f"Choose a new output directory. {destination} already exists.")
    base_official = base_data / "official"
    current = read_csv(base_official / LINE_FILE)
    incoming, metadata = source_release(source_path, quarter, source_page)
    line_items = pd.concat([current.loc[current.source_release.ne(quarter)], incoming], ignore_index=True)
    line_items = line_items.sort_values(
        ["source_release", "source_record_number"],
        key=lambda field: field.astype(int) if field.name == "source_record_number" else field,
        kind="stable",
    )
    tenders, awards = identifier_tables(line_items)
    if tenders.empty or awards.empty:
        raise ValueError("Check the source releases. The staged extract has an empty tender or award table.")
    provenance = json.loads((base_official / "provenance.json").read_text(encoding="utf-8"))
    files = {item["release"]: item for item in provenance["files"]}
    files[quarter] = metadata
    provenance["files"] = [files[key] for key in sorted(files)]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tagum-staging-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "data"
        official = stage / "official"
        official.mkdir(parents=True)
        for filename in SNAPSHOT_INPUTS:
            copy2(base_data / filename, stage / filename)
        copy2(base_official / "tagum_dpwh_flood_matches.csv", official / "tagum_dpwh_flood_matches.csv")
        for frame, filename in [
            (line_items, LINE_FILE),
            (tenders, "tagum_philgeps_tenders.csv"),
            (awards, "tagum_philgeps_awards.csv"),
        ]:
            write_csv(frame, official / filename)
        (official / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        counts = {
            "included_tagum_source_rows": len(line_items),
            "staged_release": quarter,
            "staged_source_sha256": metadata["source_sha256"],
            **build_integrated_data(stage).counts,
        }
        (official / "coverage.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        stage.rename(destination)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, default=ROOT / "data")
    parser.add_argument("--input", type=Path, required=True, help="Unmodified official quarterly download")
    parser.add_argument("--release", required=True, help="Publisher's release quarter, such as 2026Q3")
    parser.add_argument("--source-page", required=True, help="Public HTTPS URL where the file is available")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the staged sources")
    args = parser.parse_args()
    try:
        result = write_snapshot(args.base_data, args.input, args.release, args.source_page, args.output)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(2, f"Staging stopped. {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
