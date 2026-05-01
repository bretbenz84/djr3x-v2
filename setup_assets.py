#!/usr/bin/env python3
"""
setup_assets.py — Download AI models and initialize the people database.
Safe to run multiple times: never overwrites existing models or wipes existing data.
"""

import bz2
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Import config values ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DB_PATH,
    FACE_MODELS_DIR,
    LOCAL_LLM_ENABLED,
    LOCAL_LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PERSONALITY_DEFAULTS,
    RESEMBLYZER_MODEL_DIR,
    WHISPER_LOCAL_MODEL,
    WHISPER_MODEL_DIR,
)

# ── Directories required by the project ──────────────────────────────────────
REQUIRED_DIRS = [
    "assets/models/wake_word",
    "assets/models/face",
    "assets/models/whisper",
    "assets/models/resemblyzer",
    "assets/audio/clips",
    "assets/audio/startup",
    "assets/audio/tts_cache",
    "assets/music",
    "assets/trivia",
    "assets/memory",
]

# ── dlib model sources (official dlib.net, bz2-compressed) ───────────────────
DLIB_MODELS = [
    {
        "name": "shape_predictor_68_face_landmarks.dat",
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    },
    {
        "name": "dlib_face_recognition_resnet_model_v1.dat",
        "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    },
    {
        "name": "mmod_human_face_detector.dat",
        "url": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
    },
]

# ── Full database schema (mirrors Memory System section of CONTEXT.md) ────────
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id                      INTEGER PRIMARY KEY,
    name                    TEXT,
    nickname                TEXT,
    first_seen              DATETIME,
    last_seen               DATETIME,
    visit_count             INTEGER DEFAULT 0,
    familiarity_score       REAL DEFAULT 0.0,
    friendship_tier         TEXT DEFAULT 'stranger',
    warmth_score            REAL DEFAULT 0.0,
    antagonism_score        REAL DEFAULT 0.0,
    playfulness_score       REAL DEFAULT 0.0,
    curiosity_score         REAL DEFAULT 0.0,
    trust_score             REAL DEFAULT 0.5,
    net_relationship_score  REAL DEFAULT 0.0,
    lifetime_insult_count   INTEGER DEFAULT 0,
    lifetime_apology_count  INTEGER DEFAULT 0,
    height                  TEXT,
    build                   TEXT,
    hair_color              TEXT,
    hair_style              TEXT,
    skin_color              TEXT,
    age_range               TEXT,
    age_category            TEXT DEFAULT 'adult',
    notable_features        TEXT,
    appearance_updated_at   DATETIME
);

CREATE TABLE IF NOT EXISTS biometrics (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER REFERENCES people(id),
    type        TEXT,
    encoding    BLOB,
    created_at  DATETIME
);

CREATE TABLE IF NOT EXISTS person_facts (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER REFERENCES people(id),
    category    TEXT,
    key         TEXT,
    value       TEXT,
    confidence  REAL,
    source      TEXT,
    created_at  DATETIME,
    updated_at  DATETIME,
    last_confirmed_at DATETIME,
    evidence_count INTEGER DEFAULT 1,
    importance REAL DEFAULT 0.5,
    decay_rate TEXT DEFAULT 'normal',
    last_used_at DATETIME,
    stale_after_days INTEGER,
    corrected_at DATETIME
);

CREATE TABLE IF NOT EXISTS person_qa (
    id              INTEGER PRIMARY KEY,
    person_id       INTEGER REFERENCES people(id),
    question_key    TEXT,
    question_text   TEXT,
    answer_text     TEXT,
    asked_at        DATETIME,
    depth_level     INTEGER
);

CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY,
    person_id       INTEGER REFERENCES people(id),
    session_date    DATETIME,
    summary         TEXT,
    emotion_tone    TEXT,
    topics          TEXT
);

CREATE TABLE IF NOT EXISTS person_events (
    id              INTEGER PRIMARY KEY,
    person_id       INTEGER REFERENCES people(id),
    event_name      TEXT,
    event_date      DATE,
    event_notes     TEXT,
    mentioned_at    DATETIME,
    followed_up     BOOLEAN DEFAULT FALSE,
    follow_up_at    DATETIME,
    outcome         TEXT,
    status          TEXT DEFAULT 'planned',
    canceled_at     DATETIME,
    updated_at      DATETIME
);

CREATE TABLE IF NOT EXISTS personality_settings (
    id          INTEGER PRIMARY KEY,
    parameter   TEXT UNIQUE,
    value       INTEGER,
    updated_at  DATETIME,
    updated_by  TEXT
);

CREATE TABLE IF NOT EXISTS person_relationships (
    id              INTEGER PRIMARY KEY,
    from_person_id  INTEGER REFERENCES people(id),
    to_person_id    INTEGER REFERENCES people(id),
    relationship    TEXT,
    described_by    INTEGER REFERENCES people(id),
    created_at      DATETIME,
    updated_at      DATETIME,
    UNIQUE(from_person_id, to_person_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_rel_from ON person_relationships(from_person_id);
CREATE INDEX IF NOT EXISTS idx_rel_to   ON person_relationships(to_person_id);

CREATE TABLE IF NOT EXISTS person_emotional_events (
    id                       INTEGER PRIMARY KEY,
    person_id                INTEGER REFERENCES people(id),
    category                 TEXT,
    valence                  REAL,
    description              TEXT,
    loss_subject             TEXT,
    loss_subject_kind        TEXT,
    loss_subject_name        TEXT,
    mentioned_at             DATETIME,
    last_acknowledged_at     DATETIME,
    checkins_muted_at        DATETIME,
    checkins_muted_reason    TEXT,
    sensitivity_decay_days   INTEGER,
    person_invited_topic     INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_emoevent_person ON person_emotional_events(person_id);

CREATE TABLE IF NOT EXISTS person_conversation_boundaries (
    id              INTEGER PRIMARY KEY,
    person_id       INTEGER REFERENCES people(id),
    behavior        TEXT,
    topic           TEXT,
    description     TEXT,
    source_text     TEXT,
    active          INTEGER DEFAULT 1,
    created_at      DATETIME,
    updated_at      DATETIME,
    UNIQUE(person_id, behavior, topic)
);

CREATE INDEX IF NOT EXISTS idx_boundary_person ON person_conversation_boundaries(person_id);

CREATE TABLE IF NOT EXISTS person_preferences (
    id                  INTEGER PRIMARY KEY,
    person_id           INTEGER REFERENCES people(id),
    domain              TEXT,
    preference_type     TEXT,
    key                 TEXT,
    value               TEXT,
    confidence          REAL DEFAULT 1.0,
    importance          REAL DEFAULT 0.5,
    source              TEXT,
    created_at          DATETIME,
    updated_at          DATETIME,
    last_used_at        DATETIME,
    ask_cooldown_until  DATETIME,
    UNIQUE(person_id, domain, preference_type, key)
);

CREATE INDEX IF NOT EXISTS idx_pref_person ON person_preferences(person_id);
CREATE INDEX IF NOT EXISTS idx_pref_lookup ON person_preferences(person_id, domain, key);

CREATE TABLE IF NOT EXISTS person_interests (
    id                      INTEGER PRIMARY KEY,
    person_id               INTEGER REFERENCES people(id),
    name                    TEXT,
    category                TEXT,
    interest_strength       TEXT,
    confidence              REAL DEFAULT 1.0,
    source                  TEXT,
    first_mentioned_at      DATETIME,
    last_mentioned_at       DATETIME,
    last_asked_about_at     DATETIME,
    ask_cooldown_until      DATETIME,
    notes                   TEXT,
    associated_people       TEXT,
    associated_stories      TEXT,
    UNIQUE(person_id, name)
);

CREATE INDEX IF NOT EXISTS idx_interest_person ON person_interests(person_id);
CREATE INDEX IF NOT EXISTS idx_interest_lookup ON person_interests(person_id, name);
"""


# ── Progress callback for urllib downloads ────────────────────────────────────
def _progress(blocknum: int, blocksize: int, totalsize: int) -> None:
    if totalsize > 0:
        pct = min(blocknum * blocksize / totalsize * 100, 100)
        print(f"\r    {pct:5.1f}%", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Directories
# ─────────────────────────────────────────────────────────────────────────────
def create_directories(root: Path) -> list[str]:
    created = []
    for rel in REQUIRED_DIRS:
        d = root / rel
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(rel)
    return created


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — dlib face recognition models
# ─────────────────────────────────────────────────────────────────────────────
def download_dlib_models(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    created, skipped, failed = [], [], []
    dest_dir = root / FACE_MODELS_DIR

    for model in DLIB_MODELS:
        dest = dest_dir / model["name"]
        label = f"face/{model['name']}"

        if dest.exists():
            skipped.append(label)
            continue

        tmp = dest.with_suffix(".bz2.tmp")
        try:
            print(f"    Downloading {model['name']} ...")
            urllib.request.urlretrieve(model["url"], tmp, _progress)
            print()
            print(f"    Decompressing {model['name']} ...")
            with bz2.open(tmp, "rb") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            tmp.unlink()
            created.append(label)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if dest.exists():
                dest.unlink()
            failed.append(f"{label}: {exc}")

    return created, skipped, failed


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — mlx-whisper large-v3-turbo model
# ─────────────────────────────────────────────────────────────────────────────
def download_whisper_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    local_dir = root / WHISPER_MODEL_DIR
    label = f"whisper/{WHISPER_LOCAL_MODEL}"

    # config.json is the reliable sentinel: present ↔ model is fully downloaded
    if (local_dir / "config.json").exists():
        return [], [label], []

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return [], [], [
            f"{label}: huggingface_hub not installed — "
            "run: pip install huggingface_hub"
        ]

    try:
        print(f"    Downloading {WHISPER_LOCAL_MODEL}")
        print("    (~800 MB, may take several minutes on first run)")
        snapshot_download(
            repo_id=WHISPER_LOCAL_MODEL,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return [label], [], []
    except Exception as exc:
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Resemblyzer pretrained model
# ─────────────────────────────────────────────────────────────────────────────
def download_resemblyzer_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    dest = root / RESEMBLYZER_MODEL_DIR / "pretrained.pt"
    label = "resemblyzer/pretrained.pt"

    if dest.exists():
        return [], [label], []

    # Prefer copying the file bundled with the installed package
    try:
        import resemblyzer
        bundled = Path(resemblyzer.__file__).parent / "pretrained.pt"
        if bundled.exists():
            shutil.copy2(bundled, dest)
            return [f"{label} (copied from package)"], [], []
    except ImportError:
        pass

    # Fall back to downloading from the official GitHub source
    url = (
        "https://github.com/resemble-ai/Resemblyzer"
        "/raw/master/resemblyzer/pretrained.pt"
    )
    tmp = dest.with_suffix(".tmp")
    try:
        print("    Downloading Resemblyzer pretrained model ...")
        urllib.request.urlretrieve(url, tmp, _progress)
        print()
        tmp.rename(dest)
        return [label], [], []
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Ollama local sidecar model
# ─────────────────────────────────────────────────────────────────────────────
def _ollama_url(path: str) -> str:
    return str(OLLAMA_BASE_URL).rstrip("/") + path


def _ollama_api_ready(timeout_secs: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(_ollama_url("/"), timeout=timeout_secs) as resp:
            return resp.status < 500
    except Exception:
        return False


def _start_ollama_server() -> None:
    if _ollama_api_ready():
        return
    if platform.system() == "Darwin":
        try:
            proc = subprocess.run(
                ["open", "-ga", "Ollama", "--args", "hidden"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                return
        except Exception:
            pass
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _wait_for_ollama(timeout_secs: float = 30.0) -> bool:
    _start_ollama_server()
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        if _ollama_api_ready():
            return True
        time.sleep(0.25)
    return _ollama_api_ready()


def _ollama_model_present(model: str) -> bool:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    return any(
        line.split(None, 1)[0] == model
        for line in proc.stdout.splitlines()
        if line.strip() and not line.startswith("NAME")
    )


def install_ollama_model() -> tuple[list[str], list[str], list[str]]:
    created, skipped, failed = [], [], []
    provider = str(LOCAL_LLM_PROVIDER).lower()
    model = str(OLLAMA_MODEL).strip()
    label = f"ollama/{model}"

    if not LOCAL_LLM_ENABLED or provider != "ollama" or not model:
        skipped.append("ollama/local sidecar model disabled")
        return created, skipped, failed

    if shutil.which("ollama") is None:
        failed.append(
            f"{label}: ollama CLI not found — run ./setup_macos.sh or install Ollama"
        )
        return created, skipped, failed

    if not _wait_for_ollama():
        failed.append(f"{label}: Ollama server not reachable at {OLLAMA_BASE_URL}")
        return created, skipped, failed

    if _ollama_model_present(model):
        skipped.append(label)
        return created, skipped, failed

    try:
        print(f"    Pulling Ollama model {model} ...")
        subprocess.run(["ollama", "pull", model], check=True)
        created.append(label)
    except Exception as exc:
        failed.append(f"{label}: {exc}")

    return created, skipped, failed


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Database schema and personality_settings seed
# ─────────────────────────────────────────────────────────────────────────────
def _tables_exist(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='people'"
    ).fetchone()
    return row is not None


def _personality_seeded(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT COUNT(*) FROM personality_settings").fetchone()[0] > 0


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> bool:
    existing = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column in existing:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    return True


def _run_schema_updates(conn: sqlite3.Connection) -> list[str]:
    """Apply idempotent schema additions for DBs created by older setup runs."""
    applied = []
    if _ensure_column(conn, "person_emotional_events", "checkins_muted_at", "DATETIME"):
        applied.append("person_emotional_events.checkins_muted_at")
    if _ensure_column(conn, "person_emotional_events", "checkins_muted_reason", "TEXT"):
        applied.append("person_emotional_events.checkins_muted_reason")
    for column in ("loss_subject", "loss_subject_kind", "loss_subject_name"):
        if _ensure_column(conn, "person_emotional_events", column, "TEXT"):
            applied.append(f"person_emotional_events.{column}")
    if _ensure_column(conn, "person_facts", "last_confirmed_at", "DATETIME"):
        applied.append("person_facts.last_confirmed_at")
    if _ensure_column(conn, "person_facts", "evidence_count", "INTEGER DEFAULT 1"):
        applied.append("person_facts.evidence_count")
    for column, definition in (
        ("status", "TEXT DEFAULT 'planned'"),
        ("canceled_at", "DATETIME"),
        ("updated_at", "DATETIME"),
    ):
        if _ensure_column(conn, "person_events", column, definition):
            applied.append(f"person_events.{column}")
    conn.execute(
        """UPDATE person_events
           SET status = 'planned'
           WHERE status IS NULL OR status = ''"""
    )
    conn.execute(
        """UPDATE person_events
           SET updated_at = COALESCE(updated_at, follow_up_at, mentioned_at)
           WHERE updated_at IS NULL"""
    )
    conn.execute(
        """UPDATE person_facts
           SET last_confirmed_at = COALESCE(last_confirmed_at, updated_at, created_at)
           WHERE last_confirmed_at IS NULL"""
    )
    conn.execute(
        """UPDATE person_facts
           SET evidence_count = 1
           WHERE evidence_count IS NULL OR evidence_count < 1"""
    )
    return applied


def initialize_database(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    created, skipped, failed = [], [], []
    db_path = root / DB_PATH

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")

        schema_existed = _tables_exist(conn)
        conn.executescript(DB_SCHEMA)
        updates = _run_schema_updates(conn)
        conn.commit()

        if not schema_existed:
            created.append("memory/people.db — schema")
        elif updates:
            created.append(
                "memory/people.db — schema updates: " + ", ".join(updates)
            )
        else:
            skipped.append("memory/people.db — schema")

        if not _personality_seeded(conn):
            now = datetime.utcnow().isoformat()
            conn.executemany(
                "INSERT OR IGNORE INTO personality_settings "
                "(parameter, value, updated_at, updated_by) VALUES (?, ?, ?, 'default')",
                [(p, v, now) for p, v in PERSONALITY_DEFAULTS.items()],
            )
            conn.commit()
            created.append("memory/people.db — personality_settings seeded")
        else:
            skipped.append("memory/people.db — personality_settings")

        conn.close()
    except Exception as exc:
        failed.append(f"memory/people.db: {exc}")

    return created, skipped, failed


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(
    dir_created: list[str],
    all_created: list[str],
    all_skipped: list[str],
    all_failed: list[str],
) -> None:
    print()
    print("=" * 62)
    print("  setup_assets.py — complete")
    print("=" * 62)

    if dir_created:
        print(f"\n  [+] Directories created ({len(dir_created)}):")
        for d in dir_created:
            print(f"        {d}/")
    else:
        print("\n  [=] All directories already existed.")

    if all_created:
        print(f"\n  [+] Assets created ({len(all_created)}):")
        for item in all_created:
            print(f"        {item}")

    if all_skipped:
        print(f"\n  [-] Skipped — already present ({len(all_skipped)}):")
        for item in all_skipped:
            print(f"        {item}")

    if all_failed:
        print(f"\n  [!] Failures ({len(all_failed)}):")
        for item in all_failed:
            print(f"        {item}")
        print()
        print("  Check your internet connection and try again.")
        print("=" * 62)
        sys.exit(1)

    print()
    print("  All assets ready.  Run:  python3 main.py")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    root = Path(__file__).parent.resolve()

    print("DJ-R3X v2 — setup_assets.py")
    print()

    print("[1/6] Creating project directories ...")
    dir_created = create_directories(root)
    count = len(dir_created)
    print(f"      {count} created." if count else "      All already exist.")

    all_created: list[str] = []
    all_skipped: list[str] = []
    all_failed:  list[str] = []

    print("[2/6] dlib face recognition models ...")
    c, s, f = download_dlib_models(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[3/6] mlx-whisper large-v3-turbo model ...")
    c, s, f = download_whisper_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[4/6] Resemblyzer speaker-ID model ...")
    c, s, f = download_resemblyzer_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[5/6] Ollama local sidecar model ...")
    c, s, f = install_ollama_model()
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[6/6] Database schema and personality defaults ...")
    c, s, f = initialize_database(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print_summary(dir_created, all_created, all_skipped, all_failed)


def _report(created: list[str], skipped: list[str], failed: list[str]) -> None:
    for item in created:
        print(f"      + {item}")
    for item in skipped:
        print(f"      = {item} (skipped)")
    for item in failed:
        print(f"      ! {item}")


if __name__ == "__main__":
    main()
