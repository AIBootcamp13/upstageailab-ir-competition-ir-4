import os
import re
import json
import time
import pickle
import hashlib
from typing import Any, Dict, Optional


def sanitize_model_name(model: str) -> str:
    """파일명에 안전하도록 모델명을 정제한다."""
    # 알파넘/점/대시/언더스코어만 허용, 나머지는 언더스코어로 치환
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return safe[:200]


def canonicalize(obj: Any) -> str:
    """캐시 키 생성을 위한 정규화 문자열(JSON)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_key(
    provider: str,
    base_url: str,
    model: str,
    messages: Any,
    tools: Any,
    tool_choice: Any,
    params: Dict[str, Any],
) -> str:
    payload = {
        "provider": provider,
        "base_url": base_url or "",
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "params": params,
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
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
        # 손상/버전 불일치 등은 캐시 미사용 처리
        return {}


def save_cache_atomic(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def get_cache_entry(cache_dir: str, provider: str, model: str, key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(cache_dir, provider, model)
    data = load_cache(path)
    entry = data.get(key)
    if isinstance(entry, dict) and "response" in entry:
        return entry["response"]
    return entry


def set_cache_entry(cache_dir: str, provider: str, model: str, key: str, response: Dict[str, Any]) -> None:
    path = _cache_path(cache_dir, provider, model)
    data = load_cache(path)
    data[key] = {
        "schema": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": response,
    }
    save_cache_atomic(path, data)

