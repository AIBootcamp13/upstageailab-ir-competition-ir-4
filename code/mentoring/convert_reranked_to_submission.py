from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EVAL_PATH = Path("../../input/data/eval.jsonl")
DOCS_PATH = Path("../../input/data/documents.jsonl")


@dataclass
class RerankItem:
    eval_id: int
    docid: str
    score: Optional[float] = None


def load_eval_queries(eval_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            eval_id = int(obj.get("eval_id"))
            msgs = obj.get("msg") or []
            standalone = ""
            for m in msgs[::-1]:
                if m.get("role") == "user":
                    standalone = m.get("content") or ""
                    break
            mapping[eval_id] = standalone
    return mapping


def load_doc_contents(docs_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docid = str(obj.get("docid"))
            content = obj.get("content") or ""
            mapping[docid] = content
    return mapping


def try_parse_list(value: str) -> Optional[List[str]]:
    s = (value or "").strip()
    if not s:
        return []
    # JSON array
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("\"") and s.endswith("\"")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            if isinstance(parsed, str):
                return [parsed]
        except Exception:
            pass
    # Common separators
    for sep in ["|", ";", ",", " "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if parts:
                return parts
    # Single token
    return [s]


def read_reranked_csv(path: Path, topk: int) -> Dict[int, List[RerankItem]]:
    with path.open("r", encoding="utf-8") as f:
        sniffer = csv.Sniffer()
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = sniffer.sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)

        fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]

        # Column heuristics
        eval_keys = [k for k in fieldnames if k in {"eval_id", "qid", "query_id", "user_id"}]
        doc_keys = [k for k in fieldnames if k in {"docid", "doc_id", "document_id", "id", "candidate_id", "item_id"}]
        score_keys = [k for k in fieldnames if k in {"score", "rerank_score", "retrieval_score", "sim", "similarity"}]
        topk_keys = [k for k in fieldnames if k in {"topk", "docids", "retrieved_docids", "candidates"}]

        # Wide format like top1, top2, ...
        wide_keys = [k for k in fieldnames if k.startswith("top") and k[3:].isdigit()]

        groups: Dict[int, List[RerankItem]] = defaultdict(list)

        for row in reader:
            # Normalize keys to lowercase
            row_l = {k.lower(): v for k, v in row.items()}

            # eval id
            if not eval_keys:
                raise ValueError("CSV에 'eval_id' 또는 동의어(qid, query_id) 컬럼이 필요합니다.")
            raw_eval = row_l.get(eval_keys[0]) or "0"
            # 특수 포맷 처리: '00000078-0000-0000-0000-000000000000' → 78
            if eval_keys[0] == "user_id" and "-" in raw_eval:
                head = raw_eval.split("-", 1)[0]
                try:
                    eval_id = int(head)
                except Exception:
                    eval_id = int(head.lstrip("0") or 0)
            else:
                eval_id = int(raw_eval)

            # Case A: row contains a topk-ish list column
            if topk_keys:
                lst_raw = row_l.get(topk_keys[0]) or ""
                candidates = try_parse_list(lst_raw) or []
                for rank, did in enumerate(candidates[:topk], start=1):
                    groups[eval_id].append(RerankItem(eval_id, str(did), None))
                continue

            # Case B: wide columns top1, top2, ...
            if wide_keys:
                # Sort by number after 'top'
                sorted_keys = sorted(wide_keys, key=lambda k: int(k[3:]))
                for k in sorted_keys[:topk]:
                    did = row_l.get(k)
                    if did:
                        groups[eval_id].append(RerankItem(eval_id, str(did), None))
                continue

            # Case C: row-per-document format
            if doc_keys:
                did = row_l.get(doc_keys[0])
                if did:
                    score_val: Optional[float] = None
                    if score_keys and row_l.get(score_keys[0]) not in (None, ""):
                        try:
                            score_val = float(row_l.get(score_keys[0]))
                        except Exception:
                            score_val = None
                    groups[eval_id].append(RerankItem(eval_id, str(did), score_val))
                continue

            raise ValueError("CSV에서 문서 컬럼(docid/doc_id/id 등)이나 topk 관련 컬럼을 찾을 수 없습니다.")

    # Sort by score(desc) if provided, otherwise preserve insertion order
    sorted_groups: Dict[int, List[RerankItem]] = {}
    for qid, items in groups.items():
        if any(i.score is not None for i in items):
            items = sorted(items, key=lambda x: (x.score is None, -(x.score or 0.0)))
        sorted_groups[qid] = items[:topk]
    return sorted_groups


def build_submission(
    groups: Dict[int, List[RerankItem]],
    eval_queries: Dict[int, str],
    doc_contents: Dict[str, str],
) -> Iterable[str]:
    for eval_id, items in groups.items():
        obj: Dict[str, Any] = {
            "eval_id": eval_id,
            "standalone_query": eval_queries.get(eval_id, ""),
            "topk": [it.docid for it in items],
            "answer": "",
        }
        refs = []
        for it in items:
            refs.append({
                "score": float(it.score) if it.score is not None else 0.0,
                "content": doc_contents.get(it.docid, ""),
            })
        obj["references"] = refs
        yield json.dumps(obj, ensure_ascii=False)


def convert(
    input_csv: Path,
    output_path: Path,
    topk: int = 3,
) -> Tuple[int, int]:
    if not input_csv.exists():
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_csv}")
    if not EVAL_PATH.exists() or not DOCS_PATH.exists():
        raise FileNotFoundError("input/data/eval.jsonl 또는 input/data/documents.jsonl 이 존재해야 합니다.")

    eval_queries = load_eval_queries(EVAL_PATH)
    doc_contents = load_doc_contents(DOCS_PATH)
    groups = read_reranked_csv(input_csv, topk=topk)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for line in build_submission(groups, eval_queries, doc_contents):
            out.write(line + "\n")
            written += 1
    return (len(groups), written)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert reranked CSV to submission JSONL format")
    parser.add_argument("input", type=str, help="입력 CSV 경로 (예: mentoring/reranked_submission.csv)")
    parser.add_argument("--output", type=str, default=str(Path("code/baseline/submission.csv")), help="출력 파일 경로(JSON Lines, .csv 확장자 권장)")
    parser.add_argument("--topk", type=int, default=3, help="상위 K 문서 선택")
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_path = Path(args.output)

    q, w = convert(input_csv, output_path, topk=args.topk)
    print(f"완료: 질의 {q}건, 라인 {w}건 생성 → {output_path}")


if __name__ == "__main__":
    main()
