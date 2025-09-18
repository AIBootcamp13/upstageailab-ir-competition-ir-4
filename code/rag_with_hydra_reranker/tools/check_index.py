import os
import json
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def main():
    load_dotenv()

    index = os.environ.get("ES_INDEX", "test")
    password = os.environ.get("ELASTICSEARCH_PASSWORD")
    if not password:
        print("[ERROR] ELASTICSEARCH_PASSWORD not found in environment/.env")
        return 1

    url = os.environ.get("ELASTICSEARCH_URL", "https://localhost:9200")

    # 인증서 검증은 끄고 확인만 수행
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

    try:
        exists = es.indices.exists(index=index)
        print(f"[INFO] Index exists: {exists} (index='{index}')")
        if not exists:
            return 0

        cnt = es.count(index=index)["count"]
        print(f"[INFO] Total documents: {cnt}")

        mapping = es.indices.get_mapping(index=index)[index]["mappings"].get("properties", {})
        has_upstage = "embeddings_upstage" in mapping
        print(f"[INFO] Has 'embeddings_upstage' field in mapping: {has_upstage}")

        # 해당 필드가 실제 문서에 존재하는지 카운트
        try:
            up_cnt = es.count(index=index, query={"exists": {"field": "embeddings_upstage"}})["count"]
            print(f"[INFO] Documents with 'embeddings_upstage': {up_cnt}")
        except Exception as e:
            print(f"[WARN] Count exists query failed: {e}")

        # 샘플 1건 출력
        try:
            sample = es.search(index=index, size=1, query={"match_all": {}})
            print("[INFO] Sample doc:")
            print(jdump(sample["hits"]["hits"][0]["_source"]))
        except Exception as e:
            print(f"[WARN] Fetch sample failed: {e}")

        return 0
    except Exception as e:
        print(f"[ERROR] Check failed: {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
