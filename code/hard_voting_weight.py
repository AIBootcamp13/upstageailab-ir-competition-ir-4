import os
import glob
import json
from collections import defaultdict
from datetime import datetime

def hard_voting(input_dir="input/data/hard_voting", output_dir="input/data/hard_voting_result"):
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

    # 하드보팅 (가중치 4,3,2 적용, 패딩 제거)
    new_data = []
    weights = [4, 3, 2]  # 1등, 2등, 3등 가중치

    for entry in base_data:
        eval_id = entry["eval_id"]

        score = defaultdict(int)
        order = []  # 최초 등장 순서 (타이브레이커)

        for df in all_data:
            if eval_id not in df:
                continue
            topk_list = df[eval_id]["topk"]  # 항상 길이 3이라고 가정
            for pos, uuid in enumerate(topk_list[:3]):
                score[uuid] += weights[pos]
                if uuid not in order:
                    order.append(uuid)

        # 점수 내림차순, 동점 시 최초 등장 순서 우선
        sorted_items = sorted(score.items(), key=lambda x: (-x[1], order.index(x[0])))
        top3 = [u for u, _ in sorted_items[:3]]

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
