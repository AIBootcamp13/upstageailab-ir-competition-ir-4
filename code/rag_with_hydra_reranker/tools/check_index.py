import os
import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from elasticsearch import Elasticsearch


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def _list_indices(es: Elasticsearch) -> List[str]:
    try:
        items = es.cat.indices(format="json")
        return [it.get("index") for it in items if it.get("index")]
    except Exception:
        try:
            meta = es.indices.get(index="*")
            return list(meta.keys())
        except Exception:
            return []


def _check_index(es: Elasticsearch, index: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"index": index, "exists": False, "count": 0, "fields": []}
    if not es.indices.exists(index=index):
        return out
    out["exists"] = True
    try:
        out["count"] = es.count(index=index).get("count", 0)
    except Exception:
        out["count"] = 0

    try:
        props = es.indices.get_mapping(index=index)[index]["mappings"].get("properties", {})
    except Exception:
        props = {}

    dense_fields = [name for name, p in props.items() if isinstance(p, dict) and p.get("type") == "dense_vector"]
    for preferred in ("embeddings_upstage", "embeddings_sbert", "embeddings_gemini"):
        if preferred in dense_fields:
            dense_fields.remove(preferred)
            dense_fields.insert(0, preferred)

    fields_info: List[Dict[str, Any]] = []
    for field in dense_fields:
        p = props.get(field, {})
        dims = p.get("dims")
        index_flag = p.get("index")
        similarity = p.get("similarity")
        try:
            present_cnt = es.count(index=index, query={"exists": {"field": field}}).get("count", 0)
        except Exception:
            present_cnt = None

        script_ok = False
        script_err = None
        if isinstance(dims, int) and dims > 0:
            q = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.q, '{field}') + 1.0",
                            "params": {"q": [1.0] + [0.0] * (int(dims) - 1)},
                        },
                    }
                },
                "size": 1,
            }
            try:
                es.search(index=index, **q)
                script_ok = True
            except Exception as e:
                script_ok = False
                script_err = str(e)

        fields_info.append({
            "name": field,
            "dims": dims,
            "index": index_flag,
            "similarity": similarity,
            "doc_exists_count": present_cnt,
            "script_score_ok": script_ok,
            "script_score_error": script_err,
        })

    out["fields"] = fields_info
    return out


def main():
    load_dotenv()

    password = os.environ.get("ELASTICSEARCH_PASSWORD")
    if not password:
        print("[ERROR] ELASTICSEARCH_PASSWORD not found in environment/.env")
        return 1

    url = os.environ.get("ELASTICSEARCH_URL", "https://localhost:9200")

    es = Elasticsearch([url], basic_auth=("elastic", password), verify_certs=False)

    try:
        info = es.info()
        info_body = getattr(info, "body", info)
        print("[OK] Connected to Elasticsearch")
        try:
            print(jdump(info_body))
        except Exception:
            print(str(info_body))
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return 2

    env_index = os.environ.get("ES_INDEX")
    indices = [env_index] if env_index else _list_indices(es)
    if not indices:
        print("[WARN] No indices found")
        return 0

    summaries: List[Dict[str, Any]] = []
    for idx in indices:
        print(f"\n===== INDEX: {idx} =====")
        summary = _check_index(es, idx)
        summaries.append(summary)
        print(f"exists={summary['exists']}, docs={summary['count']}")
        if not summary["exists"]:
            continue
        if not summary["fields"]:
            print("(no dense_vector fields)")
        for f in summary["fields"]:
            print(f"- field={f['name']} dims={f['dims']} index={f['index']} similarity={f['similarity']} doc_exists={f['doc_exists_count']} script_ok={f['script_score_ok']}")
            if f["script_score_error"]:
                print(f"  error: {f['script_score_error']}")

    print("\n===== SUMMARY(JSON) =====")
    print(jdump(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
