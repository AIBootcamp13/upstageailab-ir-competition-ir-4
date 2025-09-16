import os
import re
import json
import time
import pickle
import hashlib
from typing import Any, Dict, Optional, List


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)[:200]


def canonicalize(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_key(provider: str, model: str, mode: str, text: str) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "mode": mode,
        "text": text,
    }
    data = canonicalize(payload)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, provider: str, model: str) -> str:
    filename = f"{provider}__{sanitize_model_name(model)}.pkl"
    return os.path.join(cache_dir, filename)


def load_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache_atomic(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def get_cache_entry(cache_dir: str, provider: str, model: str, key: str) -> Optional[List[float]]:
    path = _cache_path(cache_dir, provider, model)
    data = load_cache(path)
    entry = data.get(key)
    if isinstance(entry, dict) and "vector" in entry:
        return entry["vector"]
    if isinstance(entry, list):
        return entry
    return None


def set_cache_entry(cache_dir: str, provider: str, model: str, key: str, vector: Any) -> None:
    if hasattr(vector, 'tolist'):
        vector = vector.tolist()
    path = _cache_path(cache_dir, provider, model)
    data = load_cache(path)
    data[key] = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vector": vector,
    }
    save_cache_atomic(path, data)

