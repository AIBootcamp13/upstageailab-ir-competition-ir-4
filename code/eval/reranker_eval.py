"""
Qwen Reranker를 사용해 sample_submission(JSON Lines)에서 평가 점수(평균 최대값)를 계산하는 스크립트.

입력 파일은 JSON Lines 형식을 따르는 CSV 확장자를 사용하며, target 폴더 내의 모든 CSV에 대해 평가를 수행할 수 있다.

각 eval_id 별로 standalone_query 와 references[*].content 3개를 비교하여
Qwen3-Reranker의 yes/no 확률(yes 확률)을 계산하고, 그 중 최대값만 선택.
모든 eval_id에 대해 선택된 최대값의 평균을 출력한다.

참고: code/rag_with_hydra_reranker/reranker_usage.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 현재 스크립트 파일의 디렉토리를 작업 디렉토리로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def format_instruction(instruction: str, query: str, doc: str) -> str:
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
    output = (
        "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc
        )
    )
    return output


def process_inputs(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    pairs: List[str],
    max_length: int,
    prefix_tokens: List[int],
    suffix_tokens: List[int],
):
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )
    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


@torch.no_grad()
def compute_yes_scores(
    model: AutoModelForCausalLM,
    inputs,
    token_true_id: int,
    token_false_id: int,
) -> List[float]:
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 비정상 라인은 건너뜀
                continue


def _count_inputs_and_pairs(path: str) -> Tuple[int, int]:
    """전체 처리 예상치를 계산해 tqdm의 total에 전달."""

    total_lines = 0
    total_pairs = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            eval_id = obj.get("eval_id")
            query = obj.get("standalone_query")
            refs = obj.get("references") or []
            if eval_id is None or not query or not isinstance(refs, list):
                continue
            total_lines += 1
            cnt = 0
            for ref in refs[:3]:
                doc = (ref or {}).get("content")
                if doc:
                    cnt += 1
            total_pairs += cnt
    return total_lines, total_pairs


def evaluate_submission(
    input_path: Path,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    token_true_id: int,
    token_false_id: int,
    prefix_tokens: List[int],
    suffix_tokens: List[int],
    task: str,
    batch_size: int,
    max_length: int,
) -> Tuple[Optional[float], int]:
    path_str = str(input_path)
    total_lines, total_pairs = _count_inputs_and_pairs(path_str)

    pending_pairs: List[str] = []
    pending_meta: List[Tuple[int, int]] = []  # (eval_id, doc_index)
    eval_scores: Dict[int, List[float]] = {}

    pbar_score = tqdm(
        total=total_pairs or None,
        desc=f"{input_path.name} 스코어링 진행",
        unit="pair",
        leave=True,
        ascii=True,
    )

    def flush_batch():
        if not pending_pairs:
            return
        inputs = process_inputs(
            tokenizer, model, pending_pairs, max_length, prefix_tokens, suffix_tokens
        )
        scores = compute_yes_scores(model, inputs, token_true_id, token_false_id)
        pbar_score.update(len(scores))
        for (eval_id, _doc_idx), s in zip(pending_meta, scores):
            eval_scores.setdefault(eval_id, []).append(float(s))
        pending_pairs.clear()
        pending_meta.clear()

    for obj in tqdm(
        iter_jsonl(path_str),
        total=total_lines or None,
        desc=f"{input_path.name} 입력 JSON 처리",
        unit="line",
        ascii=True,
    ):
        eval_id = obj.get("eval_id")
        query = obj.get("standalone_query")
        refs = obj.get("references") or []
        if eval_id is None or not query or not isinstance(refs, list):
            continue

        for i, ref in enumerate(refs[:3]):
            doc = (ref or {}).get("content")
            if not doc:
                continue
            pair = format_instruction(task, query, doc)
            pending_pairs.append(pair)
            pending_meta.append((int(eval_id), i))
            if len(pending_pairs) >= batch_size:
                flush_batch()

    flush_batch()
    pbar_score.close()

    selected_max: List[float] = []
    for _eid, scores in tqdm(
        eval_scores.items(),
        total=len(eval_scores) or None,
        desc=f"{input_path.name} 최대값 집계",
        unit="eval",
        ascii=True,
    ):
        if not scores:
            continue
        selected_max.append(max(scores))

    if not selected_max:
        return None, 0

    avg_score = sum(selected_max) / len(selected_max)
    return avg_score, len(selected_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="폴더 단위 reranker 평가")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default="eval_submits",
        help="평가 대상 CSV가 들어있는 폴더 경로 (기본값: eval_submits)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
    # BATCH_SIZE = 20
    # MODEL_ID = "Qwen/Qwen3-Reranker-4B"
    # BATCH_SIZE = 5
    MODEL_ID = "Qwen/Qwen3-Reranker-8B"
    BATCH_SIZE = 4
    DEVICE = "cuda"  # "cpu" 또는 "cuda"
    MAX_LENGTH = 8192
    LOCAL_FILES_ONLY = False #True  # 네트워크 없이 로컬 캐시/경로만 사용
    USE_FP16 = True #False  # DEVICE가 cuda인 경우에만 의미 있음
    ATTENTION_IMPL = "flash_attention_2" #None  # 예: "flash_attention_2" (환경 지원 시)

    target_path = Path(args.target_dir)
    if not target_path.exists() or not target_path.is_dir():
        raise NotADirectoryError(f"평가 대상 폴더를 찾을 수 없습니다: {target_path}")

    csv_paths = sorted(p for p in target_path.glob("*.csv") if p.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"폴더에 CSV 파일이 없습니다: {target_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, padding_side="left", local_files_only=LOCAL_FILES_ONLY
    )

    model_kwargs = {"local_files_only": LOCAL_FILES_ONLY}
    if ATTENTION_IMPL:
        model_kwargs["attn_implementation"] = ATTENTION_IMPL

    if DEVICE == "cuda":
        dtype = torch.float16 if USE_FP16 else None
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=dtype,
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    task = (
        # "Given a web search query, retrieve relevant passages that answer the query"
        "Document 는 Query를 답변하기 위해 참고해야하는 적합한 문서이다."
    )

    results: List[Tuple[str, Optional[float], int]] = []
    for csv_path in csv_paths:
        avg_score, evaluated = evaluate_submission(
            csv_path,
            tokenizer,
            model,
            token_true_id,
            token_false_id,
            prefix_tokens,
            suffix_tokens,
            task,
            BATCH_SIZE,
            MAX_LENGTH,
        )
        if avg_score is None:
            print(f"{csv_path.name}: 평가할 데이터가 없습니다 (모든 레퍼런스가 비어있음).")
        else:
            print(
                f"{csv_path.name}: 평균 점수 {avg_score:.6f} (평가 개수: {evaluated}, 모델: {MODEL_ID}, device: {DEVICE})"
            )
        results.append((csv_path.name, avg_score, evaluated))

    output_path = target_path.parent / f"{target_path.name}.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "avg_score", "eval_count"])
        for name, score, evaluated in results:
            formatted_score = f"{score:.6f}" if score is not None else ""
            writer.writerow([name, formatted_score, evaluated])

    print(f"평가 결과를 저장했습니다: {output_path}")


if __name__ == "__main__":
    main()
