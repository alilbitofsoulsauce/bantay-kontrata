"""Serve the bundled review dashboard, persist audit reviews, and expose Live XAI APIs."""

import argparse
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import hmac
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
from typing import Any, Callable
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix


ROOT = Path(__file__).resolve().parent
DEFAULT_DASHBOARD = ROOT / "review" / "Bantay_Kontrata_Dashboard.html"
DEFAULT_DB = ROOT / "instance" / "bantay-kontrata.sqlite3"
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_TRUSTED_HOSTS = ["localhost", "127.0.0.1"]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

REVIEW_SCHEMA = "bantay-kontrata-tagum-reviews-v1"
STATUSES = {"in_review", "needs_documents", "reviewed", "follow_up"}
FIELD_LIMITS = {"reviewer": 80, "evidence_url": 650, "evidence_ref": 200, "notes": 2000}
MAX_BODY_SIZE = 32 * 1024 * 1024
HTML_ROOT = '<html lang="en">'
DATA_SCRIPT = '<script id="dataset" type="application/json">'

VALID_XAI_KINDS = {"award", "tender", "dpwh"}
DEFAULT_HISTORY_LIMIT = 50
MAX_HISTORY_LIMIT = 200
DEFAULT_VENDOR_LIMIT = 25
MAX_VENDOR_LIMIT = 50
DEFAULT_SURFACE_GRID = 24
MIN_SURFACE_GRID = 12
MAX_SURFACE_GRID = 36
MAX_FEATURE_INDEX = 4

SNAPSHOT_MISMATCH_ERROR = "This app serves a different data snapshot. Reopen the app dashboard."
REVIEW_TOKEN_ERROR = "Enter the team review access token in Saved reviews to save changes."

_UNSET = object()


class XAIEngineUnavailableError(RuntimeError):
    """Raised when the Live XAI inference engine cannot be imported or initialized."""


@dataclass(frozen=True)
class DashboardSnapshot:
    """Validated dashboard HTML, snapshot digest, reviewable case keys, and dataset counts."""

    served_html: str
    snapshot_id: str
    keys: set[str]
    counts: dict[str, int]


@dataclass(frozen=True)
class ServerConfig:
    """Resolved runtime configuration from function parameters and environment variables."""

    db_path: Path
    dashboard_path: Path
    data_dir: Path
    trusted_hosts: list[str] | None
    is_trusted_proxy: bool
    review_token: str


class XAIProvider:
    """Thread-safe lazy provider decoupling Flask routes from concrete LiveXAIEngine setup."""

    def __init__(self, data_dir: Path, injected_engine: Any | None = None) -> None:
        self._data_dir = data_dir
        self._engine = injected_engine
        self._lock = threading.Lock()

    @property
    def is_warmed(self) -> bool:
        """Return True when an XAI engine instance is loaded and warmed in memory."""
        if self._engine is None:
            return False
        loaded_flag = getattr(self._engine, "_loaded", True)
        return bool(loaded_flag) if isinstance(loaded_flag, bool) else True

    def is_available(self) -> bool:
        """Check whether Live XAI can be served without triggering an expensive fit."""
        if self._engine is not None:
            return True
        if "xai_engine" in sys.modules:
            loaded_module = sys.modules["xai_engine"]
            return loaded_module is not None and hasattr(loaded_module, "LiveXAIEngine")
        try:
            return find_spec("xai_engine") is not None
        except (ImportError, ValueError):
            return False

    def get_engine(self) -> Any:
        """Return the active XAI engine, lazily initializing LiveXAIEngine on first request."""
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                try:
                    from xai_engine import LiveXAIEngine

                    self._engine = LiveXAIEngine(self._data_dir)
                except Exception as error:
                    raise XAIEngineUnavailableError(
                        "Live XAI scoring engine is currently unavailable."
                    ) from error
        return self._engine


def parse_trusted_hosts(raw_value: str | list[str] | None) -> list[str] | None:
    """Resolve trusted host configuration into a hostname list or None for wildcard."""
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        cleaned = [host.strip() for host in raw_value if isinstance(host, str) and host.strip()]
        return cleaned or list(DEFAULT_TRUSTED_HOSTS)
    stripped = raw_value.strip()
    if not stripped:
        return list(DEFAULT_TRUSTED_HOSTS)
    if stripped == "*":
        return None
    hosts = [host.strip() for host in stripped.split(",") if host.strip()]
    return hosts or list(DEFAULT_TRUSTED_HOSTS)


def resolve_config(
    *,
    db_path: Path | str | None,
    dashboard_path: Path | str | None,
    data_dir: Path | str | None,
    trusted_hosts: Any,
    trusted_proxy: bool | None,
    review_token: str | None,
) -> ServerConfig:
    """Combine explicit arguments with BANTAY_* environment fallbacks."""
    env_dashboard = os.environ.get("BANTAY_DASHBOARD_PATH", "").strip()
    resolved_dashboard = (
        Path(dashboard_path).expanduser()
        if dashboard_path is not None
        else (Path(env_dashboard).expanduser() if env_dashboard else DEFAULT_DASHBOARD)
    )

    env_db = os.environ.get("BANTAY_DB_PATH", "").strip()
    resolved_db = (
        Path(db_path).expanduser().resolve()
        if db_path is not None
        else (Path(env_db).expanduser().resolve() if env_db else DEFAULT_DB)
    )

    env_data = os.environ.get("BANTAY_DATA_DIR", "").strip()
    resolved_data = (
        Path(data_dir).expanduser().resolve()
        if data_dir is not None
        else (Path(env_data).expanduser().resolve() if env_data else DEFAULT_DATA_DIR.resolve())
    )

    resolved_hosts = (
        parse_trusted_hosts(os.environ.get("BANTAY_TRUSTED_HOSTS", ""))
        if trusted_hosts is _UNSET
        else parse_trusted_hosts(trusted_hosts)
    )

    is_proxy = (
        bool(trusted_proxy)
        if trusted_proxy is not None
        else os.environ.get("BANTAY_TRUSTED_PROXY", "").strip().lower() in {"1", "true"}
    )

    resolved_token = (
        review_token.strip()
        if review_token is not None
        else os.environ.get("BANTAY_REVIEW_TOKEN", "").strip()
    )

    return ServerConfig(
        db_path=resolved_db,
        dashboard_path=resolved_dashboard,
        data_dir=resolved_data,
        trusted_hosts=resolved_hosts,
        is_trusted_proxy=is_proxy,
        review_token=resolved_token,
    )


def _extract_dpwh_keys(data: dict[str, Any], keys: set[str]) -> int:
    """Register optional DPWH contract review keys from projects or dpwh sections."""
    dpwh_ids: set[str] = set()
    candidates: list[Any] = []
    if isinstance(data.get("projects"), list):
        candidates.extend(data["projects"])
    dpwh_section = data.get("dpwh")
    if isinstance(dpwh_section, dict) and isinstance(dpwh_section.get("records"), list):
        candidates.extend(dpwh_section["records"])
    for proj in candidates:
        if isinstance(proj, dict):
            contract_id = proj.get("dpwh_contract_id")
            if isinstance(contract_id, str) and contract_id.strip():
                dpwh_ids.add(contract_id)
                keys.add(f"dpwh:{contract_id}")
    return len(dpwh_ids)


def load_dashboard_snapshot(path: Path) -> DashboardSnapshot:
    """Read, hash-verify, and index the bundled HTML dashboard snapshot."""
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
        keys: set[str] = set()
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
        dpwh_count = _extract_dpwh_keys(data, keys)
        counts = {
            "awards": len(data["award"]["records"]),
            "tenders": len(data["tender"]["records"]),
            "dpwh": dpwh_count,
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Cannot load dashboard at {path}: {error}") from error

    served = html.replace(HTML_ROOT, '<html lang="en" data-review-store="server">', 1)
    return DashboardSnapshot(served_html=served, snapshot_id=snapshot_id, keys=keys, counts=counts)


def load_dashboard(path: Path) -> tuple[str, str, set[str]]:
    """Read and validate the snapshot that this server will serve."""
    snapshot = load_dashboard_snapshot(path)
    return snapshot.served_html, snapshot.snapshot_id, snapshot.keys


def connect(db_path: Path) -> sqlite3.Connection:
    """Open an autocommit SQLite connection with busy timeout."""
    return sqlite3.connect(db_path, timeout=5, isolation_level=None)


def check_database_ready(db_path: Path) -> bool:
    """Run a lightweight SELECT 1 probe against the SQLite store."""
    try:
        with closing(connect(db_path)) as db:
            return db.execute("SELECT 1").fetchone() == (1,)
    except (OSError, sqlite3.Error):
        return False


def initialize_store(db_path: Path, snapshot_id: str, keys: set[str]) -> None:
    """Create review tables if missing and verify integrity of existing rows."""
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
    """Return the current UTC timestamp formatted with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def valid_timestamp(value: object) -> bool:
    """Return True if value is a valid ISO-8601 UTC millisecond timestamp."""
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
    """Return True when value is empty or a well-formed HTTPS URL without credentials."""
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


def validated_entries(items: object, kind: str, keys: set[str]) -> list[dict[str, Any]]:
    """Validate incoming edit or backup review entries against schema and case keys."""
    if not isinstance(items, list) or len(items) > len(keys) or (kind == "edit" and len(items) != 1):
        raise ValueError("An edit must include exactly one review. A backup must contain a list of reviews.")

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each review must be an object.")
        key, status = item.get("key"), item.get("status")
        if not isinstance(key, str) or key not in keys or key in seen:
            raise ValueError("The request contains an unknown or duplicate review case.")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("Choose a valid review status.")
        fields: dict[str, str] = {}
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
    """Read the current optimistic-concurrency revision number for snapshot_id."""
    row = db.execute("SELECT revision FROM review_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
    if row is None:
        raise sqlite3.DatabaseError("Review snapshot metadata is missing.")
    if type(row[0]) is not int or row[0] < 0:
        raise sqlite3.DatabaseError("The saved review revision is invalid.")
    return row[0]


def read_state(db: sqlite3.Connection, snapshot_id: str, revision: int) -> dict[str, Any]:
    """Read all saved review entries for snapshot_id."""
    rows = db.execute(
        "SELECT case_key, entry FROM review_entries WHERE snapshot_id = ? ORDER BY case_key", (snapshot_id,)
    )
    entries: list[dict[str, Any]] = []
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


def read_history(
    db: sqlite3.Connection,
    snapshot_id: str,
    *,
    case_key: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Read immutable audit log revisions in reverse chronological order."""
    if case_key is None:
        rows = db.execute(
            """SELECT revision, recorded_at, source, entries
            FROM review_changes WHERE snapshot_id = ?
            ORDER BY revision DESC LIMIT ?""",
            (snapshot_id, limit),
        )
    else:
        rows = db.execute(
            """SELECT revision, recorded_at, source, entries
            FROM review_changes WHERE snapshot_id = ?
            ORDER BY revision DESC""",
            (snapshot_id,),
        )

    history: list[dict[str, Any]] = []
    for revision, recorded_at, source, raw_entries in rows:
        try:
            parsed_entries = json.loads(raw_entries)
        except (RecursionError, TypeError, ValueError) as error:
            raise sqlite3.DatabaseError("A review history record contains invalid JSON.") from error
        if not isinstance(parsed_entries, list):
            raise sqlite3.DatabaseError("A review history record has an invalid entries list.")
        if case_key is not None and not any(
            isinstance(item, dict) and item.get("key") == case_key for item in parsed_entries
        ):
            continue
        history.append(
            {
                "revision": revision,
                "recorded_at": recorded_at,
                "source": source,
                "entries": parsed_entries,
            }
        )
        if len(history) >= limit:
            break
    return history


def is_same_origin() -> bool:
    """Return True when the request has no Origin header or matches host_url."""
    origin = request.headers.get("Origin")
    return origin is None or origin == request.host_url.rstrip("/")


def has_valid_review_token(expected_token: str) -> bool:
    """Verify X-Review-Token or Bearer token in constant time."""
    provided = request.headers.get("X-Review-Token", "").strip()
    if not provided:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected_token.encode("utf-8"))


def parse_bounded_int(
    raw_value: str | None,
    *,
    default: int | None,
    minimum: int,
    maximum: int,
    name: str,
) -> int:
    """Parse an integer query parameter within [minimum, maximum], raising ValueError otherwise."""
    if raw_value is None or raw_value == "":
        if default is None:
            raise ValueError(f"Query parameter '{name}' is required.")
        return default
    try:
        parsed = int(raw_value)
    except ValueError as error:
        raise ValueError(f"Query parameter '{name}' must be an integer.") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"Query parameter '{name}' must be between {minimum} and {maximum}.")
    return parsed


def _register_xai_routes(app: Flask, xai_provider: XAIProvider) -> None:
    """Attach Live XAI inference endpoints to the Flask app."""

    def execute_post(handler: Callable[[Any, dict[str, Any]], Any]):
        if not is_same_origin():
            return jsonify(error="Send Live XAI requests from the same dashboard address."), 403
        try:
            payload = request.get_json(silent=True)
        except RecursionError:
            return jsonify(error="The request contains invalid JSON."), 400
        if not isinstance(payload, dict):
            return jsonify(error="Send a JSON object payload."), 400
        engine = xai_provider.get_engine()
        try:
            return jsonify(handler(engine, payload))
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400

    @app.get("/api/xai/meta")
    def xai_meta():
        engine = xai_provider.get_engine()
        try:
            return jsonify(engine.get_meta())
        except (KeyError, ValueError) as error:
            return jsonify(error=str(error)), 400

    @app.get("/api/xai/vendors")
    def xai_vendors():
        raw_limit = request.args.get("limit", str(DEFAULT_VENDOR_LIMIT))
        try:
            parsed_limit = int(raw_limit)
        except ValueError:
            return jsonify(error="Query parameter 'limit' must be an integer."), 400
        clamped_limit = min(MAX_VENDOR_LIMIT, max(1, parsed_limit))
        engine = xai_provider.get_engine()
        try:
            return jsonify(engine.search_vendors(request.args.get("q", ""), limit=clamped_limit))
        except (KeyError, ValueError) as error:
            return jsonify(error=str(error)), 400

    @app.post("/api/xai/score-tender")
    def xai_score_tender():
        return execute_post(lambda engine, payload: engine.score_custom_tender(payload))

    @app.post("/api/xai/score-award")
    def xai_score_award():
        return execute_post(lambda engine, payload: engine.score_custom_award(payload))

    @app.post("/api/xai/whatif")
    def xai_whatif():
        return execute_post(lambda engine, payload: engine.simulate_whatif(payload))

    @app.get("/api/xai/surface")
    def xai_surface():
        kind = request.args.get("kind", "")
        if kind not in VALID_XAI_KINDS:
            return jsonify(error="Query parameter 'kind' must be award, tender, or dpwh."), 400
        try:
            fx = parse_bounded_int(
                request.args.get("fx"), default=None, minimum=0, maximum=MAX_FEATURE_INDEX, name="fx"
            )
            fy = parse_bounded_int(
                request.args.get("fy"), default=None, minimum=0, maximum=MAX_FEATURE_INDEX, name="fy"
            )
            if fx == fy:
                raise ValueError("Feature indices 'fx' and 'fy' must be different.")
            grid_size = parse_bounded_int(
                request.args.get("grid_size"),
                default=DEFAULT_SURFACE_GRID,
                minimum=MIN_SURFACE_GRID,
                maximum=MAX_SURFACE_GRID,
                name="grid_size",
            )
            engine = xai_provider.get_engine()
            return jsonify(engine.get_2d_surface(kind, fx, fy, grid_size))
        except (KeyError, TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400


def create_app(
    db_path: Path | str | None = None,
    dashboard_path: Path | str | None = None,
    *,
    data_dir: Path | str | None = None,
    xai_engine: Any | None = None,
    trusted_hosts: str | list[str] | Any = _UNSET,
    trusted_proxy: bool | None = None,
    review_token: str | None = None,
) -> Flask:
    """Create and configure the Bantay-Kontrata Flask application."""
    config = resolve_config(
        db_path=db_path,
        dashboard_path=dashboard_path,
        data_dir=data_dir,
        trusted_hosts=trusted_hosts,
        trusted_proxy=trusted_proxy,
        review_token=review_token,
    )
    snapshot = load_dashboard_snapshot(config.dashboard_path)
    initialize_store(config.db_path, snapshot.snapshot_id, snapshot.keys)
    xai_provider = XAIProvider(config.data_dir, injected_engine=xai_engine)

    app = Flask(__name__, static_folder=None)
    app.config.update(TRUSTED_HOSTS=config.trusted_hosts, MAX_CONTENT_LENGTH=MAX_BODY_SIZE)
    app.json.sort_keys = False
    if config.is_trusted_proxy:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # type: ignore[method-assign]

    @app.after_request
    def set_headers(response: Response) -> Response:
        if request.path == "/" or request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(sqlite3.DatabaseError)
    def database_error(error: sqlite3.DatabaseError):
        app.logger.error("Cannot access review database at %s: %s", config.db_path, error)
        return jsonify(error="Local review storage is unavailable. Try again or check the app console."), 503

    @app.errorhandler(XAIEngineUnavailableError)
    def xai_unavailable_error(error: XAIEngineUnavailableError):
        app.logger.error("Live XAI engine unavailable (data_dir=%s): %s", config.data_dir, error)
        return jsonify(error=str(error)), 503

    @app.errorhandler(413)
    def body_too_large(_error: Any):
        return jsonify(error="The review request exceeds 32 MB."), 413

    @app.get("/")
    def dashboard() -> Response:
        return Response(snapshot.served_html, mimetype="text/html")

    @app.get("/api/health")
    def health():
        is_db_ready = check_database_ready(config.db_path)
        payload = {
            "status": "ok" if is_db_ready else "degraded",
            "snapshot_id": snapshot.snapshot_id,
            "db_ready": is_db_ready,
            "live_xai_available": xai_provider.is_available(),
            "live_xai_warmed": xai_provider.is_warmed,
            "review_token_required": bool(config.review_token),
            "counts": snapshot.counts,
        }
        return jsonify(payload), (200 if is_db_ready else 503)

    @app.get("/api/reviews")
    def get_reviews():
        if request.args.get("snapshot_id") != snapshot.snapshot_id:
            return jsonify(error=SNAPSHOT_MISMATCH_ERROR), 409
        with closing(connect(config.db_path)) as db, db:
            db.execute("BEGIN")
            state = read_state(db, snapshot.snapshot_id, read_revision(db, snapshot.snapshot_id))
        return jsonify(state)

    @app.get("/api/reviews/history")
    def get_review_history():
        if request.args.get("snapshot_id") != snapshot.snapshot_id:
            return jsonify(error=SNAPSHOT_MISMATCH_ERROR), 409
        try:
            limit = parse_bounded_int(
                request.args.get("limit"),
                default=DEFAULT_HISTORY_LIMIT,
                minimum=1,
                maximum=MAX_HISTORY_LIMIT,
                name="limit",
            )
        except ValueError as error:
            return jsonify(error=str(error)), 400

        case_key = request.args.get("case_key", "").strip() or None
        with closing(connect(config.db_path)) as db, db:
            db.execute("BEGIN")
            history = read_history(db, snapshot.snapshot_id, case_key=case_key, limit=limit)
        return jsonify({"snapshot_id": snapshot.snapshot_id, "history": history})

    @app.put("/api/reviews")
    def save_reviews():
        if not is_same_origin():
            return jsonify(error="Save reviews from the same local dashboard address."), 403
        if config.review_token and not has_valid_review_token(config.review_token):
            return jsonify(error=REVIEW_TOKEN_ERROR), 401
        try:
            body = request.get_json(silent=True)
        except RecursionError:
            return jsonify(error="The review request contains invalid JSON."), 400
        if not isinstance(body, dict) or body.get("schema") != REVIEW_SCHEMA:
            return jsonify(error="Send a Bantay-Kontrata review request as JSON."), 400
        if body.get("snapshot_id") != snapshot.snapshot_id:
            return jsonify(error=SNAPSHOT_MISMATCH_ERROR), 409
        kind, expected = body.get("source"), body.get("revision")
        if kind not in ("edit", "backup") or type(expected) is not int or expected < 0:
            return jsonify(error="The request needs a valid review source and revision."), 400
        try:
            entries = validated_entries(body.get("entries"), kind, snapshot.keys)
        except ValueError as error:
            return jsonify(error=str(error)), 400

        with closing(connect(config.db_path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            revision = read_revision(db, snapshot.snapshot_id)
            if revision != expected:
                return jsonify(error="The reviews changed on the server. Reload the latest reviews and try again."), 409
            if entries:
                recorded_at = utc_timestamp()
                if kind == "edit":
                    entries[0]["updated_at"] = recorded_at
                db.executemany(
                    """INSERT INTO review_entries (snapshot_id, case_key, entry) VALUES (?, ?, ?)
                    ON CONFLICT (snapshot_id, case_key) DO UPDATE SET entry = excluded.entry""",
                    ((snapshot.snapshot_id, entry["key"], json.dumps(entry)) for entry in entries),
                )
                revision += 1
                db.execute(
                    "UPDATE review_snapshots SET revision = ? WHERE snapshot_id = ?",
                    (revision, snapshot.snapshot_id),
                )
                db.execute(
                    """INSERT INTO review_changes (snapshot_id, revision, recorded_at, source, entries)
                    VALUES (?, ?, ?, ?, ?)""",
                    (snapshot.snapshot_id, revision, recorded_at, kind, json.dumps(entries)),
                )
            state = read_state(db, snapshot.snapshot_id, revision)
        return jsonify(state)

    _register_xai_routes(app, xai_provider)
    return app


def main() -> None:
    """Run the production Waitress WSGI server from the command line."""
    env_host = os.environ.get("HOST", "").strip() or DEFAULT_HOST
    raw_env_port = os.environ.get("PORT", "").strip()
    try:
        default_port = int(raw_env_port) if raw_env_port else DEFAULT_PORT
    except ValueError:
        default_port = DEFAULT_PORT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=env_host, help=f"Bind address (default {env_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port (default {default_port})")
    parser.add_argument("--db", type=Path, help="SQLite review database path")
    parser.add_argument("--dashboard", type=Path, help="Generated dashboard HTML for the data snapshot")
    parser.add_argument("--data-dir", type=Path, help="Source data directory for Live XAI inference")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        app = create_app(db_path=args.db, dashboard_path=args.dashboard, data_dir=args.data_dir)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")

    from waitress import serve

    browse_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Open http://{browse_host}:{args.port} in your browser.", flush=True)
    serve(app, host=args.host, port=args.port, max_request_body_size=MAX_BODY_SIZE)


if __name__ == "__main__":
    main()
