#!/usr/bin/env python3
"""
setup_assets.py — Download AI models and initialize the people database.
Safe to run multiple times: never overwrites existing models or wipes existing data.
"""

import bz2
import shutil
import sqlite3
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# ── Import config values ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DB_PATH,
    FACE_MODELS_DIR,
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
    updated_at  DATETIME
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
    outcome         TEXT
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
# Step 5 — Database schema and personality_settings seed
# ─────────────────────────────────────────────────────────────────────────────
def _tables_exist(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='people'"
    ).fetchone()
    return row is not None


def _personality_seeded(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT COUNT(*) FROM personality_settings").fetchone()[0] > 0


def initialize_database(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    created, skipped, failed = [], [], []
    db_path = root / DB_PATH

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")

        if not _tables_exist(conn):
            conn.executescript(DB_SCHEMA)
            conn.commit()
            created.append("memory/people.db — schema")
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

    print("[1/5] Creating project directories ...")
    dir_created = create_directories(root)
    count = len(dir_created)
    print(f"      {count} created." if count else "      All already exist.")

    all_created: list[str] = []
    all_skipped: list[str] = []
    all_failed:  list[str] = []

    print("[2/5] dlib face recognition models ...")
    c, s, f = download_dlib_models(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[3/5] mlx-whisper large-v3-turbo model ...")
    c, s, f = download_whisper_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[4/5] Resemblyzer speaker-ID model ...")
    c, s, f = download_resemblyzer_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[5/5] Database schema and personality defaults ...")
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
