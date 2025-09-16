import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch


def main():
    load_dotenv()
    idx = os.environ.get("ES_INDEX", "test")
    password = os.environ.get("ELASTICSEARCH_PASSWORD")
    if not password:
        print("[ERROR] ELASTICSEARCH_PASSWORD not found in environment/.env")
        return 1
    url = os.environ.get("ELASTICSEARCH_URL", "https://localhost:9200")

    es = Elasticsearch([url], basic_auth=("elastic", password), verify_certs=False)
    try:
        if es.indices.exists(index=idx):
            es.indices.delete(index=idx)
            print(f"[OK] Deleted index: {idx}")
        else:
            print(f"[INFO] Index not found: {idx}")
        return 0
    except Exception as e:
        print(f"[ERROR] Delete failed: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

