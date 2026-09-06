#!/usr/bin/env python3
"""
setup_assets.py — Download AI models and initialize the people database.
Safe to run multiple times: never overwrites existing models or wipes existing data.
"""

import bz2
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path


# ── Re-exec under the project venv Python ────────────────────────────────────
# So `python3 setup_assets.py` (or any Python on PATH) always runs in the project
# venv — model downloads use the venv's huggingface_hub, and the dependency sync
# targets the venv. Without this, running with a system/pyenv Python that lacks the
# project deps would fail or install into the wrong place. Guarded against loops.
def _reexec_under_venv_python() -> None:
    root = Path(__file__).resolve().parent
    venv_py = root / "venv" / "bin" / "python"
    if not venv_py.exists() or os.environ.get("DJR3X_SETUP_REEXEC"):
        return
    try:
        if Path(sys.executable).resolve() == venv_py.resolve():
            return
    except Exception:
        return
    os.environ["DJR3X_SETUP_REEXEC"] = "1"
    print(f"Re-running setup_assets.py under the project venv Python: {venv_py}")
    os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])


# Only when run as a script — importing setup_assets (e.g. from tests) must NOT
# re-exec the process.
if __name__ == "__main__":
    _reexec_under_venv_python()

# ── Import config values ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DB_PATH,
    REX_DB_PATH,
    ECAPA_MODEL_DIR,
    RFDETR_MODEL_DIR,
    FACE_MODELS_DIR,
    INSIGHTFACE_MODEL_PACK,
    INSIGHTFACE_MODEL_ROOT,
    MEDIAPIPE_OBJECT_DETECTOR_MODEL,
    MEDIAPIPE_FACE_LANDMARKER_MODEL,
    MEDIAPIPE_POSE_LANDMARKER_MODEL,
    LOCAL_LLM_ENABLED,
    LOCAL_LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PERSONALITY_DEFAULTS,
    RESEMBLYZER_MODEL_DIR,
    WHISPER_LOCAL_MODEL,
    WHISPER_MODEL_DIR,
    QWEN_ASR_MODEL_REPO,
    QWEN_ASR_MODEL_DIR,
    QWEN_TTS_MODEL_DIR,
    LOCAL_TTS_MODEL_ID,
    LOCAL_TTS_MODEL_VARIANT,
    PLACE_MODEL_DIR,
    PLACE_OPEN_CLIP_MODEL,
    PLACE_OPEN_CLIP_PRETRAINED,
)

# ── Directories required by the project ──────────────────────────────────────
REQUIRED_DIRS = [
    "assets/models/wake_word",
    "assets/models/face",
    "assets/models/insightface",
    "assets/models/pose",
    "assets/models/object_detection",
    "assets/models/rfdetr",
    "assets/models/mobileclip",
    "assets/models/whisper",
    "assets/models/ecapa",
    "assets/models/resemblyzer",
    "assets/models/qwen_tts",
    "assets/models/qwen_asr",
    "assets/voices/rex",
    "assets/voices/people",
    "assets/voices/famous",
    "assets/audio/clips",
    "assets/audio/startup",
    "assets/audio/tts_cache",
    "assets/music",
    "assets/trivia",
    "assets/memory",
]

# ── dlib model sources (official dlib.net, bz2-compressed) ───────────────────
# Use HTTPS directly. urllib has been observed to fail on the HTTP -> HTTPS
# redirect path for these files on macOS even when the HTTPS URLs succeed.
DLIB_MODELS = [
    {
        "name": "shape_predictor_68_face_landmarks.dat",
        "url": "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    },
    {
        "name": "dlib_face_recognition_resnet_model_v1.dat",
        "url": "https://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    },
    {
        "name": "mmod_human_face_detector.dat",
        "url": "https://dlib.net/files/mmod_human_face_detector.dat.bz2",
    },
]

# ── InsightFace model pack (SCRFD detection + ArcFace recognition) ───────────
# The primary face backend (config.FACE_BACKEND="insightface"). The release zip
# also ships landmark/genderage models we don't use — only these two are
# extracted (~190MB instead of ~330MB). Weights are non-commercial licensed
# (fine for this personal robot).
INSIGHTFACE_PACK_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/"
    f"{INSIGHTFACE_MODEL_PACK}.zip"
)
INSIGHTFACE_ONNX_FILES = ["det_10g.onnx", "w600k_r50.onnx"]

MEDIAPIPE_FACE_LANDMARKER = {
    "name": Path(MEDIAPIPE_FACE_LANDMARKER_MODEL).name,
    "path": MEDIAPIPE_FACE_LANDMARKER_MODEL,
    "url": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}

MEDIAPIPE_POSE_LANDMARKER = {
    "name": Path(MEDIAPIPE_POSE_LANDMARKER_MODEL).name,
    "path": MEDIAPIPE_POSE_LANDMARKER_MODEL,
    "url": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ),
}

MEDIAPIPE_OBJECT_DETECTOR = {
    "name": Path(MEDIAPIPE_OBJECT_DETECTOR_MODEL).name,
    "path": MEDIAPIPE_OBJECT_DETECTOR_MODEL,
    "url": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
    ),
}

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
    lifetime_greeting_count INTEGER DEFAULT 0,
    last_greeted_at         DATETIME,
    greetings_today         INTEGER DEFAULT 0,
    greetings_today_date    TEXT,
    last_milestone_greeted  INTEGER DEFAULT 0,
    last_wellbeing_ask_at   DATETIME,
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

CREATE TABLE IF NOT EXISTS person_aliases (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER REFERENCES people(id),
    alias       TEXT NOT NULL,
    alias_norm  TEXT NOT NULL UNIQUE,
    source      TEXT,
    created_at  DATETIME,
    updated_at  DATETIME
);

CREATE INDEX IF NOT EXISTS idx_alias_person ON person_aliases(person_id);

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
    corrected_at DATETIME,
    fact_kind   TEXT DEFAULT 'fact',
    kindness    REAL,
    told_by     INTEGER REFERENCES people(id)
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

-- Cross-session dedupe for proactive topic asks (e.g. holiday-plans questions), so a
-- date-bound question Rex already raised in a prior run isn't repeated.
CREATE TABLE IF NOT EXISTS proactive_topics_asked (
    person_id   INTEGER NOT NULL,
    topic_key   TEXT NOT NULL,
    asked_at    DATETIME,
    answered    INTEGER DEFAULT 0,
    PRIMARY KEY (person_id, topic_key)
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
    updated_at      DATETIME,
    anticipated_at  DATETIME,
    hedged          INTEGER DEFAULT 0
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
    person_invited_topic     INTEGER DEFAULT 0,
    recency                  TEXT DEFAULT 'unknown'
);

CREATE INDEX IF NOT EXISTS idx_emoevent_person ON person_emotional_events(person_id);

-- One row per spoken turn, for dated conversation recall ("what did we talk
-- about on July 12?"). Mirrors memory/database.py's migration definition so a
-- FRESH install (and every test fixture built from DB_SCHEMA) matches a
-- migrated live DB — this file drifted once (recency was migration-only for
-- months) and every emotional-events read on a fresh DB hit the failure-safe
-- error path.
CREATE TABLE IF NOT EXISTS conversation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          DATETIME NOT NULL,
    day         TEXT NOT NULL,
    session_id  TEXT,
    speaker     TEXT NOT NULL,
    person_id   INTEGER,
    text        TEXT NOT NULL,
    UNIQUE(ts, speaker, text)
);

CREATE INDEX IF NOT EXISTS idx_convlog_day ON conversation_log(day);
CREATE INDEX IF NOT EXISTS idx_convlog_person ON conversation_log(person_id);

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

CREATE TABLE IF NOT EXISTS person_disposition_stats (
    person_id               INTEGER PRIMARY KEY REFERENCES people(id),
    total_samples           INTEGER DEFAULT 0,
    smile_samples           INTEGER DEFAULT 0,
    frown_samples           INTEGER DEFAULT 0,
    neutral_samples         INTEGER DEFAULT 0,
    surprise_samples        INTEGER DEFAULT 0,
    brow_furrow_samples     INTEGER DEFAULT 0,
    other_samples           INTEGER DEFAULT 0,
    smile_score             REAL DEFAULT 0.0,
    frown_score             REAL DEFAULT 0.0,
    neutral_score           REAL DEFAULT 0.0,
    surprise_score          REAL DEFAULT 0.0,
    brow_furrow_score       REAL DEFAULT 0.0,
    dominant_expression     TEXT,
    disposition_label       TEXT,
    confidence              REAL DEFAULT 0.0,
    first_observed_at       DATETIME,
    last_observed_at        DATETIME,
    last_mentioned_at       DATETIME
);

CREATE INDEX IF NOT EXISTS idx_disposition_label
    ON person_disposition_stats(disposition_label);

CREATE TABLE IF NOT EXISTS person_callback_material (
    id              INTEGER PRIMARY KEY,
    person_id       INTEGER REFERENCES people(id),
    premise         TEXT,
    category        TEXT,
    topic_slug      TEXT,
    sensitivity     TEXT DEFAULT 'guarded',
    source          TEXT,
    source_quote    TEXT,
    source_fact_id  INTEGER,
    volunteered_playfully INTEGER DEFAULT 0,
    session_id      TEXT,
    created_at      DATETIME,
    updated_at      DATETIME,
    last_used_at    DATETIME,
    use_count       INTEGER DEFAULT 0,
    retired_at      DATETIME,
    retired_reason  TEXT,
    UNIQUE(person_id, topic_slug)
);

CREATE INDEX IF NOT EXISTS idx_callback_person
    ON person_callback_material(person_id);

-- Voice-primary identity: cross-session memory for recurring UNKNOWN voices.
-- One persisted voice embedding Rex has heard but has no name for yet; person_id
-- stays NULL until the voice is named (memory/voice_signatures.py).
CREATE TABLE IF NOT EXISTS voice_signatures (
    id            INTEGER PRIMARY KEY,
    embedding     BLOB NOT NULL,
    turns         INTEGER DEFAULT 1,
    person_id     INTEGER REFERENCES people(id),
    label         TEXT,
    created_at    DATETIME,
    last_seen_at  DATETIME
);

CREATE INDEX IF NOT EXISTS idx_voice_sig_person
    ON voice_signatures(person_id);

-- Separate space: CAM++ and ECAPA both have 192 dimensions but are incompatible.
CREATE TABLE IF NOT EXISTS voice_signatures_campplus (
    id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    turns INTEGER DEFAULT 1,
    person_id INTEGER REFERENCES people(id),
    label TEXT,
    created_at DATETIME,
    last_seen_at DATETIME
);
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
# Step 2b — InsightFace model pack (SCRFD + ArcFace, the primary face backend)
# ─────────────────────────────────────────────────────────────────────────────
def download_insightface_models(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    # FaceAnalysis(name=PACK, root=ROOT) looks in ROOT/models/PACK/*.onnx
    dest_dir = root / INSIGHTFACE_MODEL_ROOT / "models" / INSIGHTFACE_MODEL_PACK
    missing = [n for n in INSIGHTFACE_ONNX_FILES if not (dest_dir / n).exists()]
    labels = [f"insightface/{n}" for n in INSIGHTFACE_ONNX_FILES]

    if not missing:
        return [], labels, []

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f"{INSIGHTFACE_MODEL_PACK}.zip.tmp"
    try:
        print(f"    Downloading {INSIGHTFACE_MODEL_PACK}.zip (~280MB) ...")
        urllib.request.urlretrieve(INSIGHTFACE_PACK_URL, tmp, _progress)
        print()
        created = []
        with zipfile.ZipFile(tmp) as zf:
            for member in zf.namelist():
                name = Path(member).name  # zip members are flat, but be safe
                if name in INSIGHTFACE_ONNX_FILES:
                    print(f"    Extracting {name} ...")
                    with zf.open(member) as src, open(dest_dir / name, "wb") as out:
                        shutil.copyfileobj(src, out)
                    created.append(f"insightface/{name}")
        tmp.unlink()
        still_missing = [n for n in INSIGHTFACE_ONNX_FILES if not (dest_dir / n).exists()]
        if still_missing:
            return created, [], [f"insightface: zip did not contain {still_missing}"]
        return created, [], []
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return [], [], [f"insightface/{INSIGHTFACE_MODEL_PACK}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — MediaPipe Face Landmarker model
# ─────────────────────────────────────────────────────────────────────────────
def download_mediapipe_face_landmarker(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    dest = root / MEDIAPIPE_FACE_LANDMARKER["path"]
    label = f"face/{MEDIAPIPE_FACE_LANDMARKER['name']}"

    if dest.exists():
        return [], [label], []

    tmp = dest.with_suffix(".tmp")
    try:
        print(f"    Downloading {MEDIAPIPE_FACE_LANDMARKER['name']} ...")
        urllib.request.urlretrieve(MEDIAPIPE_FACE_LANDMARKER["url"], tmp, _progress)
        print()
        tmp.rename(dest)
        return [label], [], []
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        if dest.exists():
            dest.unlink()
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b — MediaPipe Pose Landmarker model (body pose / gesture, wave-back)
# ─────────────────────────────────────────────────────────────────────────────
def download_mediapipe_pose_landmarker(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    dest = root / MEDIAPIPE_POSE_LANDMARKER["path"]
    label = f"pose/{MEDIAPIPE_POSE_LANDMARKER['name']}"

    if dest.exists():
        return [], [label], []

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        print(f"    Downloading {MEDIAPIPE_POSE_LANDMARKER['name']} ...")
        urllib.request.urlretrieve(MEDIAPIPE_POSE_LANDMARKER["url"], tmp, _progress)
        print()
        tmp.rename(dest)
        return [label], [], []
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        if dest.exists():
            dest.unlink()
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — MediaPipe Object Detector model
# ─────────────────────────────────────────────────────────────────────────────
def download_mediapipe_object_detector(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    dest = root / MEDIAPIPE_OBJECT_DETECTOR["path"]
    label = f"object_detection/{MEDIAPIPE_OBJECT_DETECTOR['name']}"

    if dest.exists():
        return [], [label], []

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        print(f"    Downloading {MEDIAPIPE_OBJECT_DETECTOR['name']} ...")
        urllib.request.urlretrieve(MEDIAPIPE_OBJECT_DETECTOR["url"], tmp, _progress)
        print()
        tmp.rename(dest)
        return [label], [], []
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        if dest.exists():
            dest.unlink()
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — mlx-whisper large-v3-turbo model
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
            f"run: {sys.executable} -m pip install huggingface_hub"
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


def download_qwen_asr_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Qwen3-ASR — the primary transcription backend (config.TRANSCRIPTION_BACKEND).
    Same shape as the whisper download: config.json is the completion sentinel."""
    local_dir = root / QWEN_ASR_MODEL_DIR
    label = f"qwen_asr/{QWEN_ASR_MODEL_REPO}"

    if (local_dir / "config.json").exists():
        return [], [label], []

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return [], [], [
            f"{label}: huggingface_hub not installed — "
            f"run: {sys.executable} -m pip install huggingface_hub"
        ]

    try:
        print(f"    Downloading {QWEN_ASR_MODEL_REPO}")
        print("    (~2 GB, may take several minutes on first run)")
        snapshot_download(
            repo_id=QWEN_ASR_MODEL_REPO,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return [label], [], []
    except Exception as exc:
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a2 — RF-DETR nano (the primary local animal/object detector)
# ─────────────────────────────────────────────────────────────────────────────
RFDETR_NANO = {
    "name": "rf-detr-nano.pth",
    "url": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "md5": "fb6504cce7fbdc783f7a46991f07639f",
}


def download_rfdetr_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    import hashlib
    dest = root / RFDETR_MODEL_DIR / RFDETR_NANO["name"]
    label = f"rfdetr/{RFDETR_NANO['name']}"

    if dest.exists():
        return [], [label], []

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        print(f"    Downloading {RFDETR_NANO['name']} (~350 MB) ...")
        urllib.request.urlretrieve(RFDETR_NANO["url"], tmp, _progress)
        print()
        digest = hashlib.md5(tmp.read_bytes()).hexdigest()
        if digest != RFDETR_NANO["md5"]:
            tmp.unlink()
            return [], [], [f"{label}: md5 mismatch ({digest})"]
        tmp.rename(dest)
        return [label], [], []
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a4 — YAMNet sound-event classifier (audio/sound_events.py)
# ─────────────────────────────────────────────────────────────────────────────
YAMNET_MODEL_DIR = "assets/models/yamnet"
YAMNET_FILES = [
    {
        "name": "yamnet.onnx",   # Apache-2.0; waveform-in ONNX export of Google's YAMNet
        "url": "https://huggingface.co/andrelgomes/yamnet-onnx/resolve/main/yamnet.onnx",
    },
    {
        "name": "yamnet_class_map.csv",   # official AudioSet class map (index → display_name)
        "url": "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    },
]


def download_yamnet_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Fetch the YAMNet ONNX sound-event classifier (~16 MB) + its class map.
    audio/sound_events.py disables itself cleanly when these are missing, so a
    failed download degrades to the legacy energy heuristics, not a crash."""
    created: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for item in YAMNET_FILES:
        dest = root / YAMNET_MODEL_DIR / item["name"]
        label = f"yamnet/{item['name']}"
        if dest.exists():
            skipped.append(label)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        try:
            print(f"    Downloading {item['name']} ...")
            urllib.request.urlretrieve(item["url"], tmp, _progress)
            print()
            tmp.rename(dest)
            created.append(label)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            failed.append(f"{label}: {exc}")
    return created, skipped, failed


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a3 — MobileCLIP-S2 image encoder (place recognition, perception/place_*)
# ─────────────────────────────────────────────────────────────────────────────
def download_mobileclip_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Fetch the MobileCLIP-S2 open_clip weights into assets/models/mobileclip. open_clip
    manages its own HF-hub cache under cache_dir, so we hand it the dir and let it skip
    when the weights are already there. Idempotent; degrades to a clear message if
    open_clip_torch isn't installed yet."""
    dest = root / PLACE_MODEL_DIR
    label = f"mobileclip/{PLACE_OPEN_CLIP_MODEL}/{PLACE_OPEN_CLIP_PRETRAINED}"

    # Already downloaded? (any sizeable weight blob under the cache dir)
    if dest.exists() and any(
        p.is_file() and p.stat().st_size > 50_000_000 for p in dest.rglob("*")
    ):
        return [], [label], []

    try:
        import open_clip
    except ImportError:
        return [], [], [
            f"{label}: open_clip_torch not installed "
            f"(pip install open_clip_torch, or re-run after the deps sync)"
        ]

    dest.mkdir(parents=True, exist_ok=True)
    try:
        print(f"    Downloading {label} (~0.4 GB) ...")
        # Triggers the HF-hub download into cache_dir; we discard the returned model.
        open_clip.create_model_and_transforms(
            PLACE_OPEN_CLIP_MODEL,
            pretrained=PLACE_OPEN_CLIP_PRETRAINED,
            cache_dir=str(dest),
        )
        return [label], [], []
    except Exception as exc:
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b — ECAPA-TDNN speaker-ID model (SpeechBrain, the primary voice embedder)
# ─────────────────────────────────────────────────────────────────────────────
ECAPA_REPO_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_FILES = [
    "hyperparams.yaml", "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.txt",
]


def download_campplus_model(root: Path):
    from audio.campplus import download
    try:
        from config import CAMPPLUS_MODEL_PATH
        path = root / CAMPPLUS_MODEL_PATH
        existed = path.exists()
        download(path)
        return ([] if existed else ["campplus"], ["campplus"] if existed else [], [])
    except Exception as exc:
        return [], [], [f"campplus: {exc}"]


def download_ecapa_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    local_dir = root / ECAPA_MODEL_DIR
    label = f"ecapa/{ECAPA_REPO_ID}"

    if (local_dir / "hyperparams.yaml").exists() and (
        local_dir / "embedding_model.ckpt"
    ).exists():
        return [], [label], []

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return [], [], [
            f"{label}: huggingface_hub not installed — "
            f"run: {sys.executable} -m pip install huggingface_hub"
        ]

    try:
        print(f"    Downloading {ECAPA_REPO_ID} (~80 MB)")
        snapshot_download(
            repo_id=ECAPA_REPO_ID,
            local_dir=str(local_dir),
            allow_patterns=ECAPA_FILES,
        )
        return [label], [], []
    except Exception as exc:
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Resemblyzer pretrained model (legacy fallback voice embedder)
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
# Step 6b — Qwen3-TTS voice-clone model (mlx-audio, the on-device TTS engine
#           powering --local-tts, the ElevenLabs fallback, and impersonation)
# ─────────────────────────────────────────────────────────────────────────────
def download_qwen_tts_model(
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    # Weights live under <QWEN_TTS_MODEL_DIR>/<variant>/ so switching variants
    # never collides. The runtime loads this dir by absolute path — fully offline.
    local_dir = root / QWEN_TTS_MODEL_DIR / LOCAL_TTS_MODEL_VARIANT
    label = f"qwen_tts/{LOCAL_TTS_MODEL_ID}"

    # Sentinel: BOTH weight files. config.json alone is a false-positive risk —
    # it's a tiny file snapshot_download fetches before the 2.4 GB weights, so an
    # interrupted first run would leave config.json present but the model broken.
    # This repo also ships a SECOND weight set under speech_tokenizer/ (the
    # vocoder) — without it the model loads but produces no audio, so check both.
    if (local_dir / "model.safetensors").exists() and (
        local_dir / "speech_tokenizer" / "model.safetensors"
    ).exists():
        return [], [label], []

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return [], [], [
            f"{label}: huggingface_hub not installed — "
            f"run: {sys.executable} -m pip install huggingface_hub"
        ]

    try:
        print(f"    Downloading {LOCAL_TTS_MODEL_ID}")
        print("    (~2.9 GB, may take several minutes on first run)")
        # No allow_patterns: the full repo is needed, including the nested
        # speech_tokenizer/ subdir. No local_dir_use_symlinks: huggingface_hub
        # >=1.0 removed that kwarg (passing it raises TypeError) and already
        # writes real files into local_dir.
        snapshot_download(
            repo_id=LOCAL_TTS_MODEL_ID,
            local_dir=str(local_dir),
        )
        return [label], [], []
    except Exception as exc:
        return [], [], [f"{label}: {exc}"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Ollama local sidecar model
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
    else:
        try:
            print(f"    Pulling Ollama model {model} ...")
            subprocess.run(["ollama", "pull", model], check=True)
            created.append(label)
        except Exception as exc:
            failed.append(f"{label}: {exc}")

    # Semantic-recall embedding model (memory/semantic.py). Small (~270MB); recall
    # degrades gracefully to keyword matching without it, but pulling it here is
    # what makes MEMORY_SEMANTIC_RECALL_ENABLED live on a fresh machine.
    embed_model = str(
        getattr(__import__("config"), "MEMORY_SEMANTIC_EMBED_MODEL", "nomic-embed-text")
    ).strip()
    if embed_model:
        embed_label = f"ollama/{embed_model}"
        if _ollama_model_present(embed_model):
            skipped.append(embed_label)
        else:
            try:
                print(f"    Pulling Ollama embed model {embed_model} ...")
                subprocess.run(["ollama", "pull", embed_model], check=True)
                created.append(embed_label)
            except Exception as exc:
                failed.append(f"{embed_label}: {exc}")

    return created, skipped, failed


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Database schema and personality_settings seed
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
    disposition_existed = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='person_disposition_stats'"
    ).fetchone() is not None
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS person_aliases (
            id          INTEGER PRIMARY KEY,
            person_id   INTEGER REFERENCES people(id),
            alias       TEXT NOT NULL,
            alias_norm  TEXT NOT NULL UNIQUE,
            source      TEXT,
            created_at  DATETIME,
            updated_at  DATETIME
        );
        CREATE INDEX IF NOT EXISTS idx_alias_person ON person_aliases(person_id);

        CREATE TABLE IF NOT EXISTS person_disposition_stats (
            person_id               INTEGER PRIMARY KEY REFERENCES people(id),
            total_samples           INTEGER DEFAULT 0,
            smile_samples           INTEGER DEFAULT 0,
            frown_samples           INTEGER DEFAULT 0,
            neutral_samples         INTEGER DEFAULT 0,
            surprise_samples        INTEGER DEFAULT 0,
            brow_furrow_samples     INTEGER DEFAULT 0,
            other_samples           INTEGER DEFAULT 0,
            smile_score             REAL DEFAULT 0.0,
            frown_score             REAL DEFAULT 0.0,
            neutral_score           REAL DEFAULT 0.0,
            surprise_score          REAL DEFAULT 0.0,
            brow_furrow_score       REAL DEFAULT 0.0,
            dominant_expression     TEXT,
            disposition_label       TEXT,
            confidence              REAL DEFAULT 0.0,
            first_observed_at       DATETIME,
            last_observed_at        DATETIME,
            last_mentioned_at       DATETIME
        );
        CREATE INDEX IF NOT EXISTS idx_disposition_label
            ON person_disposition_stats(disposition_label);
        """
    )
    if not disposition_existed:
        applied.append("person_disposition_stats")
    if _ensure_column(conn, "person_emotional_events", "checkins_muted_at", "DATETIME"):
        applied.append("person_emotional_events.checkins_muted_at")
    if _ensure_column(conn, "person_emotional_events", "checkins_muted_reason", "TEXT"):
        applied.append("person_emotional_events.checkins_muted_reason")
    for column in ("loss_subject", "loss_subject_kind", "loss_subject_name"):
        if _ensure_column(conn, "person_emotional_events", column, "TEXT"):
            applied.append(f"person_emotional_events.{column}")
    if _ensure_column(conn, "person_emotional_events", "recency", "TEXT DEFAULT 'unknown'"):
        applied.append("person_emotional_events.recency")
    if _ensure_column(conn, "person_facts", "last_confirmed_at", "DATETIME"):
        applied.append("person_facts.last_confirmed_at")
    if _ensure_column(conn, "person_facts", "evidence_count", "INTEGER DEFAULT 1"):
        applied.append("person_facts.evidence_count")
    for column, definition in (
        ("status", "TEXT DEFAULT 'planned'"),
        ("canceled_at", "DATETIME"),
        ("updated_at", "DATETIME"),
        ("anticipated_at", "DATETIME"),
        ("hedged", "INTEGER DEFAULT 0"),
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


# Schema for Rex's OWN episodic-memory DB (rex.db) — mirrors memory/rex_db.SCHEMA.
# Kept here too so a fresh system gets the file created at setup time.
# (See memory/rex_db.SCHEMA / config.EPISODIC_RECALL_KIND_WEIGHTS for the `kind` enum.)
REX_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS rex_episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    person_id   INTEGER,
    person_name TEXT,
    detail      TEXT,
    salience    REAL    NOT NULL DEFAULT 0.5,
    session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_rex_episodes_created ON rex_episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_rex_episodes_kind    ON rex_episodes(kind);
CREATE INDEX IF NOT EXISTS idx_rex_episodes_person  ON rex_episodes(person_id);

-- Persistent per-object room model (memory/room_model.py), one row per label.
CREATE TABLE IF NOT EXISTS room_objects (
    label           TEXT    PRIMARY KEY,
    location_bucket TEXT,
    first_seen      TEXT    NOT NULL,
    last_seen       TEXT    NOT NULL,
    sighting_count  INTEGER NOT NULL DEFAULT 1
);

-- Per-person comedy-bit cooldown (intelligence/bit_ledger.py).
CREATE TABLE IF NOT EXISTS bit_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER,
    topic       TEXT    NOT NULL,
    quoted      TEXT,
    tokens      TEXT,
    source      TEXT,
    spoken_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bit_ledger_person ON bit_ledger(person_id);
"""


def initialize_rex_database(root: Path) -> tuple[list[str], list[str], list[str]]:
    """Create Rex's episodic-memory DB (rex.db) on a fresh system. Separate file from
    people.db, with its own schema. Idempotent (CREATE TABLE IF NOT EXISTS)."""
    created, skipped, failed = [], [], []
    db_path = root / REX_DB_PATH
    try:
        existed = db_path.exists()
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(REX_DB_SCHEMA)
        conn.commit()
        conn.close()
        (created if not existed else skipped).append("memory/rex.db — episodic schema")
    except Exception as exc:
        failed.append(f"memory/rex.db: {exc}")
    return created, skipped, failed


# ─────────────────────────────────────────────────────────────────────────────
# Step 9 — Summary
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
    print("  All assets ready. Activate the venv, then run:  python main.py")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Python dependencies (so a re-run picks up newly-added packages)
# ─────────────────────────────────────────────────────────────────────────────
def sync_python_dependencies(root: Path) -> tuple[list[str], list[str], list[str]]:
    """Install/refresh Python packages from requirements.txt so re-running setup
    picks up newly-added deps (e.g. mlx-audio for --local-tts) instead of leaving
    the runtime to fail on a missing import.

    Installs into the PROJECT VENV via its own pip regardless of which interpreter
    launched this script — so `python3 setup_assets.py` still targets the venv, not
    whatever Python happens to be on PATH. dlib is excluded: it is an optional
    legacy face-recognition fallback that needs the Apple-Silicon build flags
    setup_macos.sh applies, and building it here could fail the whole step.
    Non-fatal: reports the failure and continues.
    """
    import tempfile

    label = "python packages (requirements.txt)"
    req = root / "requirements.txt"
    if not req.exists():
        return [], [], [f"{label}: requirements.txt not found"]

    venv_pip = root / "venv" / "bin" / "pip"
    pip_cmd = [str(venv_pip)] if venv_pip.exists() else [sys.executable, "-m", "pip"]

    kept = [
        line for line in req.read_text().splitlines()
        if not line.strip().lower().startswith("dlib")
    ]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write("\n".join(kept) + "\n")
            tmp_path = handle.name
        print(f"    Installing from requirements.txt via {pip_cmd[0]} (dlib excluded) ...")
        subprocess.run([*pip_cmd, "install", "-r", tmp_path], check=True)
        return [label], [], []
    except subprocess.CalledProcessError as exc:
        return [], [], [
            f"{label}: pip install failed (exit {exc.returncode}) — run manually: "
            f"{' '.join(pip_cmd)} install -r requirements.txt"
        ]
    except Exception as exc:
        return [], [], [f"{label}: {exc}"]
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    root = Path(__file__).parent.resolve()

    print("DJ-R3X v2 — setup_assets.py")
    print()

    all_created: list[str] = []
    all_skipped: list[str] = []
    all_failed:  list[str] = []

    print("[deps] Syncing Python packages from requirements.txt ...")
    c, s, f = sync_python_dependencies(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[1/16] Creating project directories ...")
    dir_created = create_directories(root)
    count = len(dir_created)
    print(f"      {count} created." if count else "      All already exist.")

    print("[2/16] InsightFace models (SCRFD + ArcFace — primary face backend) ...")
    c, s, f = download_insightface_models(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[3/16] dlib face recognition models (legacy fallback backend) ...")
    c, s, f = download_dlib_models(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[4/16] MediaPipe Face Landmarker model ...")
    c, s, f = download_mediapipe_face_landmarker(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[5/16] MediaPipe Pose Landmarker model ...")
    c, s, f = download_mediapipe_pose_landmarker(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[6/16] MediaPipe Object Detector model ...")
    c, s, f = download_mediapipe_object_detector(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[7/16] RF-DETR object-detector model (primary local detector) ...")
    c, s2, f = download_rfdetr_model(root)
    all_created += c; all_skipped += s2; all_failed += f
    _report(c, s2, f)

    print("[8/16] MobileCLIP-S2 place-recognition image encoder ...")
    c, s, f = download_mobileclip_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[9/16] YAMNet sound-event classifier (audio/sound_events.py) ...")
    c, s, f = download_yamnet_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[10/16] mlx-whisper large-v3-turbo model ...")
    c, s, f = download_whisper_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[11/16] Qwen3-ASR model (primary transcription backend) ...")
    c, s, f = download_qwen_asr_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("CAM++ speaker-ID model (default voice embedder) ...")
    c, s, f = download_campplus_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[12/16] ECAPA-TDNN speaker-ID model (optional legacy embedder) ...")
    c, s, f = download_ecapa_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[13/16] Resemblyzer speaker-ID model (legacy fallback embedder) ...")
    c, s, f = download_resemblyzer_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[14/16] Qwen3-TTS voice-clone model (on-device TTS engine) ...")
    c, s, f = download_qwen_tts_model(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[15/16] Ollama local sidecar model ...")
    c, s, f = install_ollama_model()
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)

    print("[16/16] Database schema and personality defaults ...")
    c, s, f = initialize_database(root)
    all_created += c; all_skipped += s; all_failed += f
    _report(c, s, f)
    c, s, f = initialize_rex_database(root)
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
