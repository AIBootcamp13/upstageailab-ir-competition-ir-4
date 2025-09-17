import os
import glob
import json
from collections import Counter
from datetime import datetime

def hard_voting(input_dir="input/data/hard_voting", output_dir="input/data/hard_voting_result"):
    os.makedirs(output_dir, exist_ok=True)

    # JSONL 파일 모으기
    files = glob.glob(os.path.join(input_dir, "*.csv"))  # 확장자는 csv지만 실제는 jsonl
    if not files:
        print("파일이 없습니다.")
        return

    # 첫 번째 파일을 기준으로 구조 확보
    with open(files[0], "r", encoding="utf-8") as f:
        base_data = [json.loads(line) for line in f]

    # 모든 파일 읽어서 eval_id별 topk 수집
    all_data = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            all_data.append({entry["eval_id"]: entry for entry in map(json.loads, f)})

    # 하드보팅
    new_data = []
    for entry in base_data:
        eval_id = entry["eval_id"]

        candidates = []
        for df in all_data:
            if eval_id in df:
                candidates.extend(df[eval_id]["topk"])

        counter = Counter(candidates)
        top3 = [uuid for uuid, _ in counter.most_common(3)]

        while len(top3) < 3 and candidates:
            top3.append(candidates[0])

        # topk 교체
        entry["topk"] = top3
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
