"""Serve the bundled review dashboard and save reviews on this computer."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request


ROOT = Path(__file__).resolve().parent
DEFAULT_DASHBOARD = ROOT / "review" / "Bantay_Kontrata_Dashboard.html"
DEFAULT_DB = ROOT / "instance" / "bantay-kontrata.sqlite3"
REVIEW_SCHEMA = "bantay-kontrata-tagum-reviews-v1"
STATUSES = {"in_review", "needs_documents", "reviewed", "follow_up"}
FIELD_LIMITS = {"reviewer": 80, "evidence_url": 650, "evidence_ref": 200, "notes": 2000}
MAX_BODY_SIZE = 32 * 1024 * 1024
HTML_ROOT = '<html lang="en">'
DATA_SCRIPT = '<script id="dataset" type="application/json">'


def load_dashboard(path: Path) -> tuple[str, str, set[str]]:
    """Read and validate the snapshot that this server will serve."""
    try:
        html = path.read_text(encoding="utf-8")
        if html.count(HTML_ROOT) != 1 or html.count(DATA_SCRIPT) != 1:
            raise ValueError("the dashboard markup is missing its HTML or dataset marker")
        payload, end, _ = html.split(DATA_SCRIPT, 1)[1].partition("</script>")
        if not end:
            raise ValueError("the dashboard dataset is incomplete")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("the dashboard dataset must be an object")
        snapshot_id = data.pop("snapshot_id")
        canonical = json.dumps(
            data, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if sha256(canonical).hexdigest() != snapshot_id:
            raise ValueError("the dashboard dataset does not match its snapshot ID")
        keys = set()
        for kind, key_field in (("award", "record_id"), ("tender", "bid_reference")):
            records = data[kind]["records"]
            if not isinstance(records, list) or not records:
                raise ValueError(f"the dashboard has no {kind} records")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"the dashboard contains an invalid {kind} record")
                value = record[key_field]
                if not isinstance(value, str) or not value.strip() or f"{kind}:{value}" in keys:
                    raise ValueError(f"the dashboard contains an invalid or duplicate {kind} ID")
                keys.add(f"{kind}:{value}")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Cannot load dashboard at {path}: {error}") from error

    served = html.replace(HTML_ROOT, '<html lang="en" data-review-store="server">', 1)
    return served, snapshot_id, keys


def connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path, timeout=5, isolation_level=None)


def initialize_store(db_path: Path, snapshot_id: str, keys: set[str]) -> None:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(connect(db_path)) as db, db:
            if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise RuntimeError("SQLite reported an integrity error")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS review_snapshots (
                    snapshot_id TEXT PRIMARY KEY NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0)
                );
                CREATE TABLE IF NOT EXISTS review_entries (
                    snapshot_id TEXT NOT NULL,
                    case_key TEXT NOT NULL,
                    entry TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, case_key)
                );
                CREATE TABLE IF NOT EXISTS review_changes (
                    snapshot_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    entries TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, revision)
                );
                """)
            db.execute("INSERT OR IGNORE INTO review_snapshots (snapshot_id) VALUES (?)", (snapshot_id,))
            db.execute("SELECT revision, recorded_at, source, entries FROM review_changes LIMIT 0")
            db.execute("BEGIN")
            state = read_state(db, snapshot_id, read_revision(db, snapshot_id))
            validated_entries(state["entries"], "backup", keys)
    except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError) as error:
        raise RuntimeError(f"Cannot open review database at {db_path}: {error}") from error


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z", value
    ):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def valid_evidence_url(value: str) -> bool:
    if not value:
        return True
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    try:
        url = urlsplit(value)
        host, port = url.hostname, url.port
        return (
            url.scheme == "https" and bool(host) and not url.username and not url.password
            and not any(char.isspace() for char in url.netloc)
            and not any(char in "%<>^|*" for char in host)
            and "\\" not in url.netloc and (port is None or 0 <= port <= 65535)
        )
    except ValueError:
        return False


def validated_entries(items: object, kind: str, keys: set[str]) -> list[dict]:
    if not isinstance(items, list) or len(items) > len(keys) or (kind == "edit" and len(items) != 1):
        raise ValueError("An edit must include exactly one review. A backup must contain a list of reviews.")

    entries = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each review must be an object.")
        key, status = item.get("key"), item.get("status")
        if not isinstance(key, str) or key not in keys or key in seen:
            raise ValueError("The request contains an unknown or duplicate review case.")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("Choose a valid review status.")
        fields = {}
        for field, limit in FIELD_LIMITS.items():
            value = item.get(field, "" if field == "evidence_ref" else None)
            if not isinstance(value, str) or len(value.encode("utf-16-le", errors="surrogatepass")) // 2 > limit:
                raise ValueError(f"The review has an invalid {field.replace('_', ' ')} field.")
            fields[field] = value.strip() if kind == "edit" else value
        if not valid_evidence_url(fields["evidence_url"]):
            raise ValueError("Use an HTTPS evidence link without a username or password.")
        if kind == "edit" and status in {"reviewed", "follow_up"} and (
            not fields["reviewer"] or not fields["notes"] or not (fields["evidence_url"] or fields["evidence_ref"])
        ):
            raise ValueError("Enter your initials, explain the decision, and add an evidence link or document reference.")
        updated_at = item.get("updated_at")
        if kind == "backup" and not valid_timestamp(updated_at):
            raise ValueError("The backup contains an invalid review timestamp.")
        entries.append({"key": key, "status": status, **fields, "updated_at": updated_at if kind == "backup" else None})
        seen.add(key)
    return entries


def read_revision(db: sqlite3.Connection, snapshot_id: str) -> int:
    row = db.execute("SELECT revision FROM review_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
    if row is None:
        raise sqlite3.DatabaseError("Review snapshot metadata is missing.")
    if type(row[0]) is not int or row[0] < 0:
        raise sqlite3.DatabaseError("The saved review revision is invalid.")
    return row[0]


def read_state(db: sqlite3.Connection, snapshot_id: str, revision: int) -> dict:
    rows = db.execute(
        "SELECT case_key, entry FROM review_entries WHERE snapshot_id = ? ORDER BY case_key", (snapshot_id,)
    )
    entries = []
    for case_key, text in rows:
        try:
            entry = json.loads(text)
        except (RecursionError, TypeError, ValueError) as error:
            raise sqlite3.DatabaseError("A saved review contains invalid data.") from error
        if not isinstance(entry, dict) or entry.get("key") != case_key:
            raise sqlite3.DatabaseError("A saved review has an invalid case key.")
        entries.append(entry)
    return {
        "schema": REVIEW_SCHEMA,
        "snapshot_id": snapshot_id,
        "revision": revision,
        "entries": entries,
    }


def create_app(db_path: Path | None = None, dashboard_path: Path | None = None) -> Flask:
    html, snapshot_id, keys = load_dashboard(Path(dashboard_path) if dashboard_path is not None else DEFAULT_DASHBOARD)
    path = Path(db_path).expanduser().resolve() if db_path is not None else DEFAULT_DB
    initialize_store(path, snapshot_id, keys)

    app = Flask(__name__, static_folder=None)
    app.config.update(TRUSTED_HOSTS=["localhost", "127.0.0.1"], MAX_CONTENT_LENGTH=MAX_BODY_SIZE)
    app.json.sort_keys = False

    @app.after_request
    def set_headers(response):
        if request.path == "/" or request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(sqlite3.DatabaseError)
    def database_error(error):
        app.logger.error("Cannot access review database at %s: %s", path, error)
        return jsonify(error="Local review storage is unavailable. Try again or check the app console."), 503

    @app.errorhandler(413)
    def body_too_large(_error):
        return jsonify(error="The review request exceeds 32 MB."), 413

    @app.get("/")
    def dashboard():
        return Response(html, mimetype="text/html")

    @app.get("/api/reviews")
    def get_reviews():
        if request.args.get("snapshot_id") != snapshot_id:
            return jsonify(error="This app serves a different data snapshot. Reopen the app dashboard."), 409
        with closing(connect(path)) as db, db:
            db.execute("BEGIN")
            state = read_state(db, snapshot_id, read_revision(db, snapshot_id))
        return jsonify(state)

    @app.put("/api/reviews")
    def save_reviews():
        origin = request.headers.get("Origin")
        if origin is not None and origin != request.host_url.rstrip("/"):
            return jsonify(error="Save reviews from the same local dashboard address."), 403
        try:
            body = request.get_json(silent=True)
        except RecursionError:
            return jsonify(error="The review request contains invalid JSON."), 400
        if not isinstance(body, dict) or body.get("schema") != REVIEW_SCHEMA:
            return jsonify(error="Send a Bantay-Kontrata review request as JSON."), 400
        if body.get("snapshot_id") != snapshot_id:
            return jsonify(error="This app serves a different data snapshot. Reopen the app dashboard."), 409
        kind, expected = body.get("source"), body.get("revision")
        if kind not in ("edit", "backup") or type(expected) is not int or expected < 0:
            return jsonify(error="The request needs a valid review source and revision."), 400
        try:
            entries = validated_entries(body.get("entries"), kind, keys)
        except ValueError as error:
            return jsonify(error=str(error)), 400

        with closing(connect(path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            revision = read_revision(db, snapshot_id)
            if revision != expected:
                return jsonify(error="The reviews changed on the server. Reload the latest reviews and try again."), 409
            if entries:
                recorded_at = utc_timestamp()
                if kind == "edit":
                    entries[0]["updated_at"] = recorded_at
                db.executemany(
                    """INSERT INTO review_entries (snapshot_id, case_key, entry) VALUES (?, ?, ?)
                    ON CONFLICT (snapshot_id, case_key) DO UPDATE SET entry = excluded.entry""",
                    ((snapshot_id, entry["key"], json.dumps(entry)) for entry in entries),
                )
                revision += 1
                db.execute("UPDATE review_snapshots SET revision = ? WHERE snapshot_id = ?", (revision, snapshot_id))
                db.execute(
                    """INSERT INTO review_changes (snapshot_id, revision, recorded_at, source, entries)
                    VALUES (?, ?, ?, ?, ?)""",
                    (snapshot_id, revision, recorded_at, kind, json.dumps(entries)),
                )
            state = read_state(db, snapshot_id, revision)
        return jsonify(state)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Local port (default 8000)")
    parser.add_argument("--db", type=Path, help="SQLite review database path")
    parser.add_argument("--dashboard", type=Path, help="Generated dashboard HTML for the data snapshot")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        app = create_app(db_path=args.db, dashboard_path=args.dashboard)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")

    from waitress import serve

    print(f"Open http://127.0.0.1:{args.port} in your browser.", flush=True)
    serve(app, host="127.0.0.1", port=args.port, max_request_body_size=MAX_BODY_SIZE)


if __name__ == "__main__":
    main()
