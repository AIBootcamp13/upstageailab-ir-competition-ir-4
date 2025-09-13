import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
import traceback
import hydra
from omegaconf import DictConfig

# 현재 스크립트 파일의 디렉토리를 작업 디렉토리로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Function calling에 사용할 함수 정의
def get_tools(cfg):
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search relevant documents",
                "parameters": {
                    "properties": {
                        "standalone_query": {
                            "type": "string",
                            "description": cfg.prompts.standalone_query_description
                        }
                    },
                    "required": ["standalone_query"],
                    "type": "object"
                }
            }
        },
    ]

# LLM에게 검색 필요 여부를 판단하게 하는 함수
def check_search_decision(messages, client, cfg):
    """
    주어진 메시지에 대해 LLM이 검색을 호출할지 여부를 판단

    Args:
        messages: 사용자 메시지 리스트
        client: OpenAI 클라이언트
        cfg: Hydra config 객체

    Returns:
        dict: {
            "search_called": bool,
            "standalone_query": str or None
        }
    """
    # 시스템 프롬프트 (config.yaml의 cfg.prompts.function_calling 사용)
    system_prompt = cfg.prompts.function_calling

    msg = [{"role": "system", "content": system_prompt}] + messages

    try:
        result = client.chat.completions.create(
            model=cfg.model.name,  # config에서 모델 이름 사용
            messages=msg,
            tools=get_tools(cfg),
            temperature=cfg.model.temperature,
            seed=cfg.model.seed,
            timeout=cfg.model.timeout,
            reasoning_effort=cfg.model.reasoning_effort
        )

        # LLM 답변 내용 캡처
        assistant_response = result.choices[0].message.content or ""

        # 검색이 필요한 경우 tool_calls가 있음
        if result.choices[0].message.tool_calls:
            tool_call = result.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            standalone_query = function_args.get("standalone_query", "")

            return {
                "search_called": True,
                "standalone_query": standalone_query,
                "assistant_response": assistant_response
            }
        else:
            return {
                "search_called": False,
                "standalone_query": None,
                "assistant_response": assistant_response
            }
            
    except Exception as e:
        log.error(f"Error in check_search_decision: {e}")
        traceback.print_exc()
        return {
            "search_called": False,
            "standalone_query": None,
            "assistant_response": None,
            "error": str(e)
        }

def load_eval_data(eval_filename):
    """eval.jsonl 파일을 로드하여 딕셔너리로 반환"""
    eval_data = {}
    with open(eval_filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            eval_data[data['eval_id']] = data['msg']
    return eval_data

def check_search_decisions_for_eval_ids(cfg: DictConfig, eval_ids, eval_filename="../../input/data/eval.jsonl", output_filename=None):
    """
    지정된 eval_id들에 대해 검색 호출 여부를 확인

    Args:
        cfg: Hydra config 객체
        eval_ids: 확인할 eval_id 리스트
        eval_filename: eval.jsonl 파일 경로
        output_filename: 결과를 저장할 파일명 (None이면 출력만)

    Returns:
        list: 각 eval_id에 대한 결과 리스트
    """
    # OpenAI API 키 환경변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # OpenAI 클라이언트 생성
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    if openai_base_url:
        client = OpenAI(base_url=openai_base_url)
    else:
        client = OpenAI()

    # eval 데이터 로드
    eval_data = load_eval_data(eval_filename)

    results = []

    for eval_id in eval_ids:
        if eval_id not in eval_data:
            log.warning(f"eval_id {eval_id} not found in eval data")
            continue

        messages = eval_data[eval_id]

        # 사용자 질문 추출 (마지막 user 메시지)
        user_question = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_question = msg["content"]
                break

        log.info(f"Processing eval_id {eval_id}: {user_question[:100]}...")

        # 검색 호출 여부 확인 (config 전달)
        decision = check_search_decision(messages, client, cfg)

        result = {
            "eval_id": eval_id,
            "question": user_question,
            "search_called": decision["search_called"],
            "standalone_query": decision["standalone_query"],
            "assistant_response": decision.get("assistant_response")
        }

        if "error" in decision:
            result["error"] = decision["error"]

        results.append(result)

        log.info(f"eval_id {eval_id}: search_called={decision['search_called']}, query='{decision.get('standalone_query', 'N/A')}'")

    # 결과 출력
    print("\n=== 검색 호출 여부 결과 ===")
    search_called_count = 0
    for result in results:
        status = "🔍 검색 호출" if result["search_called"] else "💬 직접 답변"
        print(f"\neval_id {result['eval_id']}: {status}")
        print(f"질문: {result['question']}")
        if result["search_called"]:
            search_called_count += 1
            print(f"검색 쿼리: {result['standalone_query']}")
        if result.get("assistant_response"):
            print(f"LLM 답변: {result['assistant_response']}")

    print(f"\n총 {len(results)}개 질문 중 {search_called_count}개가 검색 호출, {len(results) - search_called_count}개가 직접 답변")

    # 파일로 저장
    if output_filename:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        log.info(f"Results saved to {output_filename}")

    return results

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """메인 함수 - 예시 사용법"""
    # 예시: 몇 개의 eval_id로 테스트
    test_eval_ids = [66, 17]

    print("=== Search Decision Checker ===")
    print(f"테스트할 eval_id: {test_eval_ids}")

    try:
        results = check_search_decisions_for_eval_ids(
            cfg=cfg,
            eval_ids=test_eval_ids,
            # output_filename="search_decision_results.jsonl"
        )

        print(f"\n처리 완료! 총 {len(results)}개 결과")

    except Exception as e:
        log.error(f"Error in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
