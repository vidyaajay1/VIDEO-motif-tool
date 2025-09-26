import os, io, json, uuid, hashlib, shutil
from typing import Dict, List, Tuple
from pathlib import Path
from fastapi import UploadFile, HTTPException

TMP_DIR = os.environ.get("TMP_DIR", "/app_data/tmp_jobs")
TRACKS_DIR = Path(TMP_DIR) / "tracks"
TRACKS_DIR.mkdir(parents=True, exist_ok=True)

def _ns_key_for_regular(data_id: str) -> str:
    # regular mode namespace
    return f"data:{data_id}"

def _ns_key_for_compare(session_id: str, label: str) -> str:
    # compare mode namespace
    return f"cmp:{session_id}:{label}"

def _ns_dir(ns_key: str) -> Path:
    d = TRACKS_DIR / ns_key
    d.mkdir(parents=True, exist_ok=True)
    return d

def _registry_path(ns_key: str) -> Path:
    return _ns_dir(ns_key) / "registry.json"

def _load_registry(ns_key: str) -> Dict[str, Dict]:
    rp = _registry_path(ns_key)
    if not rp.exists():
        return {}
    with open(rp, "r", encoding="utf-8") as fh:
        return json.load(fh)

def _save_registry(ns_key: str, reg: Dict[str, Dict]) -> None:
    rp = _registry_path(ns_key)
    tmp = str(rp) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2)
    os.replace(tmp, rp)

def _sha1_file(fp: Path, buf_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with open(fp, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def persist_tracks(ns_key: str, bigwigs: List[UploadFile], tracknames: List[str]) -> List[str]:
    if len(bigwigs) != len(tracknames):
        raise HTTPException(status_code=400, detail="Mismatch in bigwig and track metadata count")

    reg = _load_registry(ns_key)
    saved_ids: List[str] = []

    for i, bw in enumerate(bigwigs):
        try:
            label, color = tracknames[i].split("|", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid track meta '{tracknames[i]}'. Use 'Label|#RRGGBB'.")

        # Write to namespace dir with a stable filename (uuid to avoid collisions)
        out_fp = _ns_dir(ns_key) / f"{uuid.uuid4().hex}.bw"
        with open(out_fp, "wb") as out:
            shutil.copyfileobj(bw.file, out)
        try:
            bw.file.close()
        except Exception:
            pass

        sha1 = _sha1_file(out_fp)
        # De-dup inside this namespace: if we already have the same sha1+label, reuse id
        for tid, meta in reg.items():
            if meta.get("sha1") == sha1 and meta.get("label") == label and meta.get("color") == color:
                saved_ids.append(tid)
                break
        else:
            track_id = uuid.uuid4().hex
            reg[track_id] = {
                "path": str(out_fp),
                "label": label.strip(),
                "color": color.strip(),
                "sha1": sha1,
            }
            saved_ids.append(track_id)

    _save_registry(ns_key, reg)
    return saved_ids

def list_tracks(ns_key: str) -> Dict[str, Dict]:
    return _load_registry(ns_key)

def remove_track(ns_key: str, track_id: str) -> None:
    reg = _load_registry(ns_key)
    if track_id not in reg:
        raise HTTPException(status_code=404, detail="track_id not found")
    meta = reg.pop(track_id)
    _save_registry(ns_key, reg)
    # Best-effort file delete (don’t break if already gone)
    try:
        Path(meta["path"]).unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass

def resolve_inputs(ns_key: str, track_ids: List[str] | None) -> List[Tuple[str, str, str]]:
    reg = _load_registry(ns_key)
    if not reg:
        raise HTTPException(status_code=400, detail="No tracks registered yet")

    # Default: use all registered tracks
    ids = track_ids or list(reg.keys())

    bw_inputs: List[Tuple[str, str, str]] = []
    for tid in ids:
        meta = reg.get(tid)
        if not meta:
            raise HTTPException(status_code=400, detail=f"Unknown track_id: {tid}")
        if not os.path.exists(meta["path"]):
            raise HTTPException(status_code=410, detail=f"File missing for track_id {tid} (was cleaned?)")
        bw_inputs.append((meta["path"], meta["label"], meta["color"]))
    return bw_inputs
