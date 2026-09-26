"""Human-labelled reference photographs. No model prediction ever trains the album."""

import json
from pathlib import Path
import threading
import time
import uuid

import config

_lock = threading.RLock()


def root() -> Path:
    return Path(config.PHOTO_MEMORY_DIR).expanduser()


def _read() -> list[dict]:
    path = root() / "album.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Invalid photo album")
    return data


def references(kind: str, label: str) -> list[dict]:
    """Include all pets despite detector dog/cat wobble; objects stay class-specific.

    Fail closed when the gallery exceeds the request budget rather than silently
    omitting a sibling and making an overconfident match against a partial album.
    """
    with _lock:
        rows = [r for r in _read() if r["kind"] == kind
                and (kind == "animal" or r["label"] == label)]
        if len(rows) > int(config.PHOTO_MEMORY_MAX_IDENTITIES):
            raise ValueError("Photo album exceeds comparison budget")
        result = []
        for row in rows:
            photos = []
            for filename in row["photos"]:
                if Path(filename).name != filename or not filename.endswith(".jpg"):
                    raise ValueError("Invalid album photo path")
                photos.append((root() / filename).read_bytes())
            if not photos:
                raise ValueError("Missing reference photographs")
            result.append({**row, "images": photos})
        return result


def save(jpeg: bytes, *, name: str, kind: str, label: str,
         owner_id: int | None) -> dict:
    """Append a confirmed view, scoped to the person doing the teaching.

    Atomic index replacement means interruption can leave an unreferenced JPEG,
    but never a published index pointing at a half-written image.
    """
    with _lock:
        rows = _read()  # A corrupt album must not be silently overwritten.
        row = next((r for r in rows if r["kind"] == kind
                    and r["name"].casefold() == name.casefold()
                    and r.get("owner_id") == owner_id
                    and (kind == "animal" or r["label"] == label)), None)
        if row is None:
            row = {"id": uuid.uuid4().hex, "kind": kind, "label": label,
                   "name": name, "owner_id": owner_id, "photos": []}
            rows.append(row)
        directory = root()
        directory.mkdir(parents=True, exist_ok=True)
        filename = uuid.uuid4().hex + ".jpg"
        photo = directory / filename
        temporary = photo.with_suffix(".tmp")
        temporary.write_bytes(jpeg)
        temporary.replace(photo)
        row["photos"].append(filename)
        cap = max(1, int(config.PHOTO_MEMORY_PHOTOS_PER_IDENTITY))
        retired = row["photos"][:-cap]
        row["photos"] = row["photos"][-cap:]
        row["updated_at"] = time.time()
        index_tmp = directory / "album.tmp"
        index_tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        index_tmp.replace(directory / "album.json")
        for old in retired:
            if Path(old).name == old:
                (directory / old).unlink(missing_ok=True)
        return dict(row)
