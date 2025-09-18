import os
import glob
import json
from collections import defaultdict
from datetime import datetime

def hard_voting(
    input_dir="input/data/hard_voting",
    output_dir="input/data/hard_voting_result",
    n=3,
    topk=3,
):
    os.makedirs(output_dir, exist_ok=True)

    # JSONL 파일 모으기 (확장자는 csv지만 실제는 jsonl)
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not files:
        print("파일이 없습니다.")
        return

    # 첫 번째 파일을 기준으로 구조 확보
    with open(files[0], "r", encoding="utf-8") as f:
        base_data = [json.loads(line) for line in f]

    # 모든 파일 읽어서 eval_id별 데이터 사전화
    all_data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            all_data.append({entry["eval_id"]: entry for entry in map(json.loads, f)})

    new_data = []

    for entry in base_data:
        eval_id = entry["eval_id"]

        item_votes = defaultdict(int)

        for df in all_data:
            if eval_id not in df:
                continue
            topk_list = df[eval_id].get("topk", [])
            for pos, uuid in enumerate(topk_list[:n]):
                rank = pos + 1
                rank_score = n - rank + 1  # mentoring 스타일 점수 계산
                item_votes[uuid] += rank_score

        if item_votes:
            sorted_items = sorted(item_votes.items(), key=lambda x: x[1], reverse=True)
            top_items = [uuid for uuid, _ in sorted_items[:topk]]
        else:
            top_items = []

        if len(top_items) < topk:
            for uuid in entry.get("topk", []):
                if uuid not in top_items:
                    top_items.append(uuid)
                if len(top_items) == topk:
                    break

        entry["topk"] = top_items[:topk]
        new_data.append(entry)

    # 저장 (JSONL 포맷)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"hard_voting_{timestamp}_{len(files)}.csv")  # 확장자는 csv지만 실제는 jsonl
    with open(save_path, "w", encoding="utf-8") as f:
        for row in new_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"저장 완료: {save_path}")

if __name__ == "__main__":
    hard_voting()
