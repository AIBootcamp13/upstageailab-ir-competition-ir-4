import os
import json
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import hydra
from omegaconf import DictConfig

# 현재 스크립트 파일의 디렉토리를 작업 디렉토리로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# .env 파일에서 환경 변수 로드
load_dotenv()

# Sentence Transformer 모델 - main 함수에서 초기화


# SetntenceTransformer를 이용하여 임베딩 생성
def get_embedding(sentences, model):
    return model.encode(sentences)


# 주어진 문서의 리스트에서 배치 단위로 임베딩 생성
def get_embeddings_in_batches(docs, model, batch_size=100):
    log = logging.getLogger(__name__)
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents, model)
        batch_embeddings.extend(embeddings)
        log.info(f'Processing batch {i}')
    return batch_embeddings


# 새로운 index 생성
def create_es_index(es, index, settings, mappings):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
        es.indices.delete(index=index)
    # 지정된 설정으로 새로운 인덱스 생성
    es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(es, index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(es, index, docs):
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]
    return helpers.bulk(es, actions)


# 역색인을 이용한 검색
def sparse_retrieve(es, index_name, query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index=index_name, query=query, size=size, sort="_score")


# Vector 유사도를 이용한 검색
def dense_retrieve(es, model, index_name, query_str, size, num_candidates=100):
    # 벡터 유사도 검색에 사용할 쿼리 임베딩 가져오기
    query_embedding = get_embedding([query_str], model)[0]

    # KNN을 사용한 벡터 유사성 검색을 위한 매개변수 설정
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": num_candidates
    }

    # 지정된 인덱스에서 벡터 유사도 검색 수행
    return es.search(index=index_name, knn=knn)


# Elasticsearch 설정은 main 함수에서 초기화

# Elasticsearch 설정들은 main 함수에서 정의

# 초기화 코드들은 main 함수로 이동

# RAG를 구현하는 코드
from openai import OpenAI
import traceback

# OpenAI 설정들은 main 함수에서 초기화

# 프롬프트들은 config.yaml에서 관리

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


# LLM과 검색엔진을 활용한 RAG 구현
def answer_question(messages, client, cfg, es, index_name):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": cfg.prompts.function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=cfg.model.name,
            messages=msg,
            tools=get_tools(cfg),
            #tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=cfg.model.temperature,
            seed=cfg.model.seed,
            timeout=cfg.model.timeout,
            reasoning_effort=cfg.model.reasoning_effort
        )
    except Exception as e:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # Baseline으로는 sparse_retrieve만 사용하여 검색 결과 추출
        search_result = sparse_retrieve(es, index_name, standalone_query, cfg.search.top_k)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        for i,rst in enumerate(search_result['hits']['hits']):
            retrieved_context.append(rst["_source"]["content"])
            response["topk"].append(rst["_source"]["docid"])
            response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})

        if cfg.qa.use_final_answer:
            # 검색된 컨텍스트로 별도 QA 수행
            content = json.dumps(retrieved_context)
            messages.append({"role": "assistant", "content": content})
            msg = [{"role": "system", "content": cfg.prompts.qa}] + messages
            try:
                qaresult = client.chat.completions.create(
                        model=cfg.model.name,
                        messages=msg,
                        temperature=cfg.model.temperature,
                        seed=cfg.model.seed,
                        timeout=cfg.model.qa_timeout
                    )
            except Exception as e:
                traceback.print_exc()
                return response
            response["answer"] = qaresult.choices[0].message.content
        else:
            # 현재 방식: 검색 결과만 반환
            response["answer"] = result.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename, client, cfg, es, index_name):
    log = logging.getLogger(__name__)
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            log.info(f'Test {idx} - Question: {j["msg"]}')
            response = answer_question(j["msg"], client, cfg, es, index_name)
            log.info(f'Answer: {response["answer"]}')
            log.info(f'Retrieved {len(response["topk"])} documents: {response["topk"]}')
            log.debug(f'References: {len(response["references"])} items')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

            if cfg.eval.max_iterations > 0 and idx >= cfg.eval.max_iterations:
                break

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info("Starting RAG evaluation process")
    
    # OpenAI API 키 환경변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # OpenAI 클라이언트 생성
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    if openai_base_url:
        client = OpenAI(base_url=openai_base_url)
    else:
        client = OpenAI()

    # Elasticsearch 설정
    es_password = os.getenv("ELASTICSEARCH_PASSWORD")
    if not es_password:
        raise ValueError("ELASTICSEARCH_PASSWORD environment variable is required")

    # Elasticsearch client 생성
    es = Elasticsearch(
        list(cfg.elasticsearch.hosts), 
        basic_auth=(cfg.elasticsearch.username, es_password), 
        ca_certs=cfg.elasticsearch.ca_certs
    )
    log.info(f'Elasticsearch connection established: {es.info()}')

    # Sentence Transformer 모델 초기화
    model = SentenceTransformer(cfg.embedding.model_name)

    # Elasticsearch 인덱스 설정
    settings = {
        "analysis": {
            "analyzer": {
                "nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["nori_posfilter"]
                }
            },
            "filter": {
                "nori_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
                }
            }
        }
    }

    mappings = {
        "properties": {
            "content": {"type": "text", "analyzer": "nori"},
            "embeddings": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "l2_norm"
            }
        }
    }

    # 인덱스 생성
    create_es_index(es, cfg.index.name, settings, mappings)

    # 문서 임베딩 생성 및 인덱싱
    index_docs = []
    with open(cfg.paths.documents) as f:
        docs = [json.loads(line) for line in f]
    
    embeddings = get_embeddings_in_batches(docs, model, cfg.embedding.batch_size)
    
    for doc, embedding in zip(docs, embeddings):
        doc["embeddings"] = embedding.tolist()
        index_docs.append(doc)

    # 대량 문서 추가
    ret = bulk_add(es, cfg.index.name, index_docs)
    log.info(f'Bulk indexing completed: {ret}')

    # 테스트 쿼리 실행
    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    log.info(f'Running test query: {test_query}')
    
    # 역색인을 사용하는 검색 예제
    search_result_retrieve = sparse_retrieve(es, cfg.index.name, test_query, cfg.search.top_k)
    log.info('Sparse retrieval results:')
    for rst in search_result_retrieve['hits']['hits']:
        log.info(f'Score: {rst["_score"]:.4f}, Content: {rst["_source"]["content"][:100]}...')

    # Vector 유사도 사용한 검색 예제
    search_result_retrieve = dense_retrieve(es, model, cfg.index.name, test_query, cfg.search.top_k, cfg.search.num_candidates)
    log.info('Dense retrieval results:')
    for rst in search_result_retrieve['hits']['hits']:
        log.info(f'Score: {rst["_score"]:.4f}, Content: {rst["_source"]["content"][:100]}...')

    # 평가 데이터에 대해서 결과 생성
    # CSV 파일을 Hydra outputs 디렉토리에 저장
    from hydra.core.hydra_config import HydraConfig
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    output_path = os.path.join(hydra_output_dir, cfg.paths.output)
    
    log.info(f'Current working directory: {os.getcwd()}')
    log.info(f'Hydra output directory: {hydra_output_dir}')
    log.info(f'Starting evaluation with output file: {output_path}')
    eval_rag(cfg.paths.eval_data, output_path, client, cfg, es, cfg.index.name)
    log.info('RAG evaluation process completed')


if __name__ == "__main__":
    main()

