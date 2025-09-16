import os
import json
import logging
import psutil
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from langchain_upstage import UpstageEmbeddings
import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 현재 스크립트 파일의 디렉토리를 작업 디렉토리로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# .env 파일에서 환경 변수 로드
load_dotenv()

# 메모리 사용량 측정 유틸리티 함수
def log_memory_usage(log, message=""):
    """CPU와 GPU 메모리 사용량을 로그에 출력"""
    try:
        # CPU 메모리 (RAM)
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_memory_mb = memory_info.rss / 1024 / 1024  # MB 단위

        # 시스템 전체 메모리
        system_memory = psutil.virtual_memory()
        total_memory_gb = system_memory.total / 1024 / 1024 / 1024  # GB 단위
        available_memory_gb = system_memory.available / 1024 / 1024 / 1024  # GB 단위
        used_percent = system_memory.percent

        log_msg = f"{message} - CPU 메모리: {cpu_memory_mb:.1f}MB, 시스템 메모리: {used_percent:.1f}% 사용 ({available_memory_gb:.1f}GB/{total_memory_gb:.1f}GB 가용)"

        # GPU 메모리 (PyTorch CUDA 사용 시)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB 단위
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB 단위
            gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # GB 단위
            log_msg += f", GPU 메모리: {gpu_memory_allocated:.1f}GB 할당/{gpu_memory_reserved:.1f}GB 예약/{gpu_total_memory:.1f}GB 총용량"
        else:
            log_msg += ", GPU: 사용 불가"

        log.info(log_msg)
    except Exception as e:
        log.warning(f"메모리 사용량 측정 실패: {e}")

# Sentence Transformer 모델 - main 함수에서 초기화


# Upstage / SBERT 임베딩 유틸
def upstage_get_embedding(sentences, model, is_query=False):
    if is_query:
        return model.embed_query(sentences[0]) if len(sentences) == 1 else [model.embed_query(q) for q in sentences]
    else:
        return model.embed_documents(sentences)

def sbert_get_embedding(sentences, model):
    return model.encode(sentences)

# Gemini 임베딩 유틸 (단순화됨)
def gemini_get_embedding(sentences, model, is_query=False):
    """
    Gemini 임베딩 생성 (단순화된 버전)

    Args:
        sentences: 임베딩할 문장 리스트
        model: GoogleGenerativeAIEmbeddings 모델
        is_query: 질의용 임베딩인지 여부
    """
    if is_query:
        # 질의는 보통 단일 문장
        if len(sentences) == 1:
            return model.embed_query(sentences[0])
        else:
            return [model.embed_query(q) for q in sentences]
    else:
        # 문서 임베딩 (배치 처리는 상위 함수에서 수행)
        return model.embed_documents(sentences)


# 문서 임베딩 배치 생성 (passage embedding)
def get_embeddings_in_batches(docs, model, batch_size=100):
    log = logging.getLogger(__name__)
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = upstage_get_embedding(contents, model, is_query=False)
        batch_embeddings.extend(embeddings)
        log.info(f'Processing batch {i}')
    return batch_embeddings



# 새로운 index 생성
def create_es_index(es, index, settings, mappings, force_recreate=True):
    # 인덱스가 이미 존재하는지 확인
    if es.indices.exists(index=index):
        if force_recreate:
            # 인덱스가 이미 존재하면 설정을 새로운 것으로 갱신하기 위해 삭제
            es.indices.delete(index=index)
            # 지정된 설정으로 새로운 인덱스 생성
            es.indices.create(index=index, settings=settings, mappings=mappings)
        # force_recreate가 False면 기존 인덱스를 그대로 사용
    else:
        # 인덱스가 없으면 새로 생성
        es.indices.create(index=index, settings=settings, mappings=mappings)


# 지정된 인덱스 삭제
def delete_es_index(es, index):
    es.indices.delete(index=index)


# Elasticsearch 헬퍼 함수를 사용하여 대량 인덱싱 수행
def bulk_add(es, index, docs):
    """
    대량 인덱싱을 수행 (타임아웃 설정 포함)

    Args:
        es: Elasticsearch 클라이언트
        index: 인덱스 이름
        docs: 인덱싱할 문서 리스트

    Returns:
        (성공한 문서 수, 실패한 문서 수)
    """
    # 대량 인덱싱 작업을 준비
    actions = [
        {
            '_index': index,
            '_source': doc
        }
        for doc in docs
    ]

    return helpers.bulk(
        es,
        actions,
        request_timeout=120,  # 120초 타임아웃
        max_retries=3,        # 최대 3번 재시도
        initial_backoff=2,    # 2초 초기 백오프
        max_backoff=60        # 최대 60초 백오프
    )


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


# Vector 유사도를 이용한 검색 (backend별)
def dense_retrieve_upstage(es, model, index_name, query_str, size, num_candidates=100):
    log = logging.getLogger(__name__)
    # 쿼리 임베딩 (4096차원)
    query_embedding = upstage_get_embedding([query_str], model, is_query=True)
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()

    # num_candidates 보정 (k <= num_candidates 유지)
    orig_num_cand = num_candidates
    if num_candidates < size:
        num_candidates = size
        log.warning(f"[Upstage] num_candidates({orig_num_cand}) < k/size({size}) → num_candidates={num_candidates}로 보정")

    knn = {
        "field": "embeddings_upstage",
        "query_vector": query_embedding,
        "k": size,
        "num_candidates": num_candidates
    }
    # top-level size로 최종 반환 개수 지정
    return es.search(index=index_name, knn=knn, size=size)

def dense_retrieve_sbert(es, model, index_name, query_str, size, num_candidates=100):
    log = logging.getLogger(__name__)
    query_embedding = sbert_get_embedding([query_str], model)[0]
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()

    # num_candidates 보정
    orig_num_cand = num_candidates
    if num_candidates < size:
        num_candidates = size
        log.warning(f"[SBERT] num_candidates({orig_num_cand}) < k/size({size}) → num_candidates={num_candidates}로 보정")

    knn = {
        "field": "embeddings_sbert",
        "query_vector": query_embedding,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=index_name, knn=knn, size=size)

# HyDE 캐시 (간단한 메모리 캐시)
_hyde_cache = {}

# HyDE 기법: 가상 문서 생성 함수 (범용)
def generate_hypothetical_document(query, client, cfg):
    """질의에 대해 LLM을 사용하여 가상의 답변 문서를 생성 (OpenAI/Gemini 호환)"""
    log = logging.getLogger(__name__)

    # 캐시 확인
    cache_key = f"{query}_{getattr(cfg.prompts, 'hyde', '')}"
    if cache_key in _hyde_cache:
        log.debug(f"HyDE 캐시에서 문서 반환: {query[:50]}...")
        return _hyde_cache[cache_key]
    try:
        # 통합된 hyde 프롬프트 사용
        hyde_prompt = getattr(cfg.prompts, 'hyde', '다음 질문에 대해 전문적이고 상세한 설명 문서를 작성해주세요.')

        # 통합 LLM 호출 함수 사용
        messages = [
            {"role": "system", "content": hyde_prompt},
            {"role": "user", "content": query}
        ]

        result = call_llm_unified(
            client=client,
            messages=messages,
            cfg=cfg
        )

        # result 타입에 따라 적절히 처리
        if isinstance(result, dict):
            hypothetical_doc = result["choices"][0]["message"]["content"]
        else:
            hypothetical_doc = result.choices[0].message.content
        log.debug(f"Generated hypothetical document for query: {query[:50]}...")

        # 설정에 따라 생성된 문서 출력
        if getattr(cfg.logging, 'show_hyde_generated_document', False):
            log.info(f"Dense Retrieval HyDE 생성 문서 (질의: {query[:30]}...):\n{hypothetical_doc}")

        # 캐시에 저장
        _hyde_cache[cache_key] = hypothetical_doc

        return hypothetical_doc

    except Exception as e:
        log.warning(f"Failed to generate hypothetical document: {e}")
        # 실패 시 원본 질의 반환
        return query

# HyDE 기법을 활용한 Upstage Dense Retrieve
def dense_retrieve_upstage_hyde(es, model, index_name, query_str, size, num_candidates, client, cfg):
    """HyDE 기법: 질의 -> 가상문서 생성 -> 임베딩 -> 검색"""
    # 1단계: 가상 문서 생성
    hypothetical_doc = generate_hypothetical_document(query_str, client, cfg)

    # 2단계: 가상 문서를 임베딩하여 검색 (기존 dense_retrieve_upstage와 동일)
    log = logging.getLogger(__name__)
    query_embedding = upstage_get_embedding([hypothetical_doc], model, is_query=True)
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()

    # num_candidates 보정
    orig_num_cand = num_candidates
    if num_candidates < size:
        num_candidates = size
        log.warning(f"[Upstage-HyDE] num_candidates({orig_num_cand}) < k/size({size}) → num_candidates={num_candidates}로 보정")

    knn = {
        "field": "embeddings_upstage",
        "query_vector": query_embedding,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=index_name, knn=knn, size=size)

# Gemini Vector 유사도를 이용한 검색
def dense_retrieve_gemini(es, model, index_name, query_str, size, num_candidates=100):
    """Gemini 임베딩을 사용한 dense retrieve"""
    # 쿼리 임베딩 (3072차원)
    log = logging.getLogger(__name__)
    query_embedding = gemini_get_embedding([query_str], model, is_query=True)
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()

    # num_candidates 보정
    orig_num_cand = num_candidates
    if num_candidates < size:
        num_candidates = size
        log.warning(f"[Gemini] num_candidates({orig_num_cand}) < k/size({size}) → num_candidates={num_candidates}로 보정")

    knn = {
        "field": "embeddings_gemini",
        "query_vector": query_embedding,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=index_name, knn=knn, size=size)

# HyDE 기법을 활용한 Gemini Dense Retrieve
def dense_retrieve_gemini_hyde(es, model, index_name, query_str, size, num_candidates, client, cfg):
    """HyDE 기법: 질의 -> 가상문서 생성 -> Gemini 임베딩 -> 검색"""
    # 1단계: 가상 문서 생성
    hypothetical_doc = generate_hypothetical_document(query_str, client, cfg)

    # 2단계: 가상 문서를 Gemini 임베딩하여 검색
    log = logging.getLogger(__name__)
    query_embedding = gemini_get_embedding([hypothetical_doc], model, is_query=True)
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()

    # num_candidates 보정
    orig_num_cand = num_candidates
    if num_candidates < size:
        num_candidates = size
        log.warning(f"[Gemini-HyDE] num_candidates({orig_num_cand}) < k/size({size}) → num_candidates={num_candidates}로 보정")

    knn = {
        "field": "embeddings_gemini",
        "query_vector": query_embedding,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=index_name, knn=knn, size=size)

def retrieve_all(es, index_name):
    """모든 문서를 조회하여 리랭킹 후보로 사용하기 위한 리스트를 반환한다.
    score는 0.0으로 설정한다.
    """
    documents = []
    # helpers.scan은 전체 문서를 스트리밍으로 순회
    for hit in helpers.scan(es, index=index_name, query={"query": {"match_all": {}}}):
        src = hit.get("_source", {})
        docid = src.get("docid", hit.get("_id"))
        content = src.get("content", "")
        documents.append({
            "content": content,
            "docid": docid,
            "score": 0.0
        })
    return documents

# 공식 사용법 기반 Qwen3-Reranker-8B 초기화 (CausalLM + yes/no)
def initialize_reranker(cfg):
    log = logging.getLogger(__name__)
    if not cfg.reranker.use_reranker:
        return None, None, None

    try:
        log.info(f"Loading reranker model (CausalLM): {cfg.reranker.model_name}")

        kwargs = {}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            kwargs.update({
                'dtype': torch.float16,
            })

        tokenizer = AutoTokenizer.from_pretrained(cfg.reranker.model_name, padding_side='left')

        # pad 토큰 보장 (일부 토크나이저는 pad_token이 없을 수 있음)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.sep_token is not None:
                tokenizer.pad_token = tokenizer.sep_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model = AutoModelForCausalLM.from_pretrained(cfg.reranker.model_name, **kwargs).to(device).eval()

        # yes/no 토큰 id
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        max_length = 8192

        # 공식 프리픽스/서픽스 템플릿
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

        aux = {
            'device': device,
            'token_true_id': token_true_id,
            'token_false_id': token_false_id,
            'max_length': max_length,
            'prefix_tokens': prefix_tokens,
            'suffix_tokens': suffix_tokens,
        }
        log.info("Reranker(CausalLM) initialized")
        return tokenizer, model, aux
    except Exception as e:
        log.error(f"Failed to load reranker model: {e}")
        return None, None, None


def _format_instruction(instruction, query, doc):
    if instruction is None or len(str(instruction).strip()) == 0:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


# 공식 사용법 기반 reranking (배치 처리 및 메모리 관리 기능 추가)
def rerank_documents(query, documents, reranker_tokenizer, reranker_model, reranker_aux, cfg, client=None):
    log = logging.getLogger(__name__)

    if reranker_tokenizer is None or reranker_model is None or reranker_aux is None:
        log.warning("Reranker not initialized, returning original order")
        return documents[:cfg.reranker.top_k]

    try:
        instruction = getattr(cfg.reranker, 'instruction', None)
        # config에서 batch_size를 가져오거나, 없으면 기본값(예: 2)으로 설정
        batch_size = getattr(cfg.reranker, 'batch_size', 2)

        # HyDE 기법 사용 여부 확인
        use_hyde = getattr(cfg.reranker, 'use_hyde', False)
        rerank_query = query

        if use_hyde and client is not None:
            log.info("Reranker HyDE 기법 활성화 - 가상 문서 생성 중...")
            rerank_query = generate_hypothetical_document(query, client, cfg)
        elif use_hyde and client is None:
            log.warning("Reranker HyDE 활성화되었으나 client가 없어 원본 쿼리 사용")

        log.info(f"Reranking with {'HyDE query' if use_hyde and client else 'original query'}")

        # 실제 쿼리 (원본 또는 HyDE 생성)를 사용하여 리랭킹 수행
        actual_query = rerank_query
        
        all_scores = []
        device = reranker_aux['device']
        
        # 문서를 배치 단위로 나누어 처리
        for i in range(0, len(documents), batch_size):
            log.info(f"Reranking batch {i // batch_size + 1}...")
            batch_docs = documents[i:i + batch_size]
            
            # 입력 문자열 생성 (HyDE 쿼리 또는 원본 쿼리 사용)
            pairs = [_format_instruction(instruction, actual_query, doc["content"]) for doc in batch_docs]

            # 토크나이즈
            max_length = reranker_aux['max_length']
            prefix_tokens = reranker_aux['prefix_tokens']
            suffix_tokens = reranker_aux['suffix_tokens']

            inputs = reranker_tokenizer(
                pairs,
                padding=False,
                truncation='longest_first',
                return_attention_mask=False,
                max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
            )
            for j, ids in enumerate(inputs['input_ids']):
                inputs['input_ids'][j] = prefix_tokens + ids + suffix_tokens

            inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
            
            # 데이터를 GPU로 이동
            for key in inputs:
                inputs[key] = inputs[key].to(device)

            # 배치 추론
            with torch.no_grad():
                batch_scores_logits = reranker_model(**inputs).logits[:, -1, :]
                true_vector = batch_scores_logits[:, reranker_aux['token_true_id']]
                false_vector = batch_scores_logits[:, reranker_aux['token_false_id']]
                stacked_scores = torch.stack([false_vector, true_vector], dim=1)
                log_softmax_scores = torch.nn.functional.log_softmax(stacked_scores, dim=1)
                scores = log_softmax_scores[:, 1].exp().tolist()
            
            all_scores.extend(scores)

            # === 메모리 관리 코드 추가 ===
            # 현재 배치에서 사용한 텐서들을 GPU 메모리에서 명시적으로 삭제
            del inputs, batch_scores_logits, true_vector, false_vector, stacked_scores, log_softmax_scores
            # PyTorch가 캐싱하고 있는 사용되지 않는 메모리를 GPU에서 해제
            torch.cuda.empty_cache()
            # =========================

        # 전체 점수를 기준으로 정렬
        doc_score_pairs = list(zip(documents, all_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, _ in doc_score_pairs[:cfg.reranker.top_k]]

        log.info(f"Reranked {len(documents)} documents to top {len(reranked_docs)} using batch size {batch_size} with memory clearing")
        return reranked_docs

    except Exception as e:
        log.error(f"Error during reranking: {e}", exc_info=True) # 에러 발생 시 상세 정보 출력
        return documents[:cfg.reranker.top_k]


# Elasticsearch 설정은 main 함수에서 초기화

# Elasticsearch 설정들은 main 함수에서 정의

# 초기화 코드들은 main 함수로 이동

# RAG를 구현하는 코드
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # embedding만 LangChain 사용
from google import genai
from google.genai import types
import traceback
import time

# OpenAI 및 Gemini 클라이언트 생성 함수
def create_llm_client(cfg):
    """설정에 따라 LLM 클라이언트 생성 (OpenAI 또는 Gemini)"""
    log = logging.getLogger(__name__)
    model_name = cfg.llm.model

    # Gemini 모델인지 확인
    if "gemini" in model_name.lower():
        # GOOGLE_API_KEY 또는 GEMINI_API_KEY 지원
        google_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Gemini 모델을 사용하려면 GEMINI_API_KEY 또는 GOOGLE_API_KEY 환경변수가 필요합니다.")

        log.info(f"Gemini LLM 클라이언트 생성: {model_name}")
        return genai.Client(api_key=google_api_key)
    else:
        # OpenAI 호환 클라이언트 (solar-pro2 등)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI 호환 모델을 사용하려면 OPENAI_API_KEY가 필요합니다.")

        log.info(f"OpenAI 호환 클라이언트 생성: {model_name}")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_base_url:
            return OpenAI(base_url=openai_base_url)
        else:
            return OpenAI()

def apply_llm_delay(cfg):
    """설정된 시간만큼 대기 (rate limit 회피)"""
    delay_seconds = getattr(cfg.llm, 'delay_seconds', 0)
    if delay_seconds > 0:
        log = logging.getLogger(__name__)
        log.debug(f"LLM 호출 대기: {delay_seconds}초")
        time.sleep(delay_seconds)

def call_llm_unified(client, messages, cfg, tools=None, tool_choice=None):
    """OpenAI/Gemini 통합 LLM 호출 함수"""
    log = logging.getLogger(__name__)
    model_name = cfg.llm.model

    # rate limit 회피를 위한 대기
    apply_llm_delay(cfg)

    # Gemini 클라이언트인지 확인 (더 정확한 타입 체크)
    if isinstance(client, genai.Client):  # genai.Client
        # 재시도 설정값 로드 (기본값: 최대 5회, 30초 대기)
        retry_max = int(getattr(cfg.llm, 'retry_max', 5) or 5)
        retry_delay = int(getattr(cfg.llm, 'retry_delay_seconds', 30) or 30)

        # OpenAI 메시지 형식을 Gemini types.Content 형식으로 변환
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"
            # system 메시지는 user로 포함 (Gemini는 system role 없음)
            content_text = msg["content"]
            if msg["role"] == "system":
                content_text = f"System: {content_text}"

            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=content_text)]
            ))

        # Gemini tool calling 지원
        gemini_tools = None
        if tools:
            gemini_tools = []
            for tool in tools:
                if tool["type"] == "function":
                    func_def = tool["function"]
                    # Gemini FunctionDeclaration 생성
                    gemini_func = types.FunctionDeclaration(
                        name=func_def["name"],
                        description=func_def["description"],
                        parameters=func_def["parameters"]
                    )
                    gemini_tools.append(types.Tool(function_declarations=[gemini_func]))

        config = types.GenerateContentConfig(
            temperature=cfg.llm.temperature,
            tools=gemini_tools if gemini_tools else None,
        )

        # 2xx가 아닌 응답/예외 발생 시 재시도
        last_exc = None
        response = None
        for attempt in range(1, retry_max + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                # google-genai 클라이언트는 2xx 외에는 예외를 던지는 것이 일반적이므로
                # 성공적으로 객체가 반환되면 그대로 진행
                break
            except Exception as e:
                # 상태 코드 추출 시도
                status_code = None
                try:
                    status_code = getattr(e, 'status_code', None)
                except Exception:
                    pass
                try:
                    if status_code is None and hasattr(e, 'response') and getattr(e, 'response') is not None:
                        status_code = getattr(e.response, 'status_code', None)
                except Exception:
                    pass

                # 예외 메시지에서 503 등 키워드 추출 (fallback)
                status_hint = None
                msg_text = str(e)
                if 'HTTP/1.1' in msg_text or 'Service Unavailable' in msg_text or '503' in msg_text:
                    status_hint = 'non-2xx (likely 503)'

                # 재시도 여부 판단: 2xx가 아니거나 상태 코드를 알 수 없어도 예외가 발생한 경우 재시도
                should_retry = True
                if status_code is not None:
                    should_retry = not (200 <= int(status_code) < 300)

                last_exc = e
                if attempt < retry_max and should_retry:
                    sc_str = f"status={status_code}" if status_code is not None else (f"hint={status_hint}" if status_hint else "status=unknown")
                    log.warning(f"Gemini 호출 실패(시도 {attempt}/{retry_max}, {sc_str}). {retry_delay}초 후 재시도합니다...")
                    time.sleep(retry_delay)
                    continue
                # 마지막 시도이거나, 재시도 대상이 아니면 예외 재전파
                raise

        # OpenAI 형식으로 응답 변환
        result = {
            "choices": [{
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": None
                }
            }]
        }

        # 응답에서 function call 또는 텍스트 추출
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                tool_calls = []
                text_parts = []

                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Function call 처리
                        func_call = part.function_call
                        tool_calls.append({
                            "id": f"call_{hash(func_call.name)}",
                            "type": "function",
                            "function": {
                                "name": func_call.name,
                                "arguments": json.dumps(dict(func_call.args))
                            }
                        })
                    elif hasattr(part, 'text') and part.text:
                        # 텍스트 응답 처리
                        text_parts.append(part.text)

                if tool_calls:
                    result["choices"][0]["message"]["tool_calls"] = tool_calls
                if text_parts:
                    result["choices"][0]["message"]["content"] = "".join(text_parts)

        return result
    else:  # OpenAI 클라이언트
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": cfg.llm.temperature,
            "seed": cfg.llm.seed,
            "timeout": cfg.llm.timeout
        }

        # reasoning_effort 지원 (o3 계열 등)
        if hasattr(cfg.llm, 'reasoning_effort') and getattr(cfg.llm, 'reasoning_effort'):
            params["reasoning_effort"] = cfg.llm.reasoning_effort

        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        return client.chat.completions.create(**params)

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
def answer_question(messages, client, cfg, es, index_name, dense_ctx=None, reranker_tokenizer=None, reranker_model=None, reranker_aux=None):
    # 함수 출력 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    # 로거 초기화
    log = logging.getLogger(__name__)

    # 질의 분석 및 검색 이외의 질의 대응을 위한 LLM 활용
    msg = [{"role": "system", "content": cfg.prompts.function_calling}] + messages
    try:
        result = call_llm_unified(
            client=client,
            messages=msg,
            cfg=cfg,
            tools=get_tools(cfg),
            #tool_choice={"type": "function", "function": {"name": "search"}}
        )
    except Exception:
        traceback.print_exc()
        return response

    # 검색이 필요한 경우 검색 호출후 결과를 활용하여 답변 생성
    # result가 딕셔너리인지 객체인지 확인하여 처리
    if isinstance(result, dict):
        # 딕셔너리 형태 (Gemini)
        message = result["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
    else:
        # 객체 형태 (OpenAI)
        message = result.choices[0].message
        tool_calls = getattr(message, 'tool_calls', None)

    if tool_calls:
        tool_call = tool_calls[0]
        if isinstance(tool_call, dict):
            function_args = json.loads(tool_call["function"]["arguments"])
        else:
            function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # Standalone query 로깅
        log.info(f"검색 쿼리 생성 완료: '{standalone_query}'")

        # 설정 토글에 따른 검색 동작 분기
        response["standalone_query"] = standalone_query

        sparse_enabled = getattr(cfg.retrieve.sparse, 'enabled', True)
        upstage_enabled = getattr(cfg.retrieve.dense_upstage, 'enabled', False)
        sbert_enabled = getattr(cfg.retrieve.dense_sbert, 'enabled', False)
        upstage_hyde_enabled = getattr(cfg.retrieve.dense_upstage_hyde, 'enabled', False)
        gemini_enabled = getattr(cfg.retrieve.dense_gemini, 'enabled', False)
        gemini_hyde_enabled = getattr(cfg.retrieve.dense_gemini_hyde, 'enabled', False)

        documents = []
        if not sparse_enabled and not upstage_enabled and not sbert_enabled and not upstage_hyde_enabled and not gemini_enabled and not gemini_hyde_enabled:
            # 리트리브 비활성화: 전체 문서를 리랭킹 대상으로 사용
            documents = retrieve_all(es, index_name)
        else:
            docids = set()
            # 각 retrieve 방식별 문서 수집 현황 추적을 위한 카운터
            sparse_count = 0
            upstage_count = 0
            sbert_count = 0
            hyde_count = 0
            gemini_count = 0
            gemini_hyde_count = 0

            # 각 retrieve 방식별 DocID 수집을 위한 리스트
            sparse_docids = []
            upstage_docids = []
            sbert_docids = []
            hyde_docids = []
            gemini_docids = []
            gemini_hyde_docids = []
            if sparse_enabled:
                sparse_result = sparse_retrieve(es, index_name, standalone_query, cfg.retrieve.sparse.top_k)
                sparse_retrieved = len(sparse_result['hits']['hits'])
                for rst in sparse_result['hits']['hits']:
                    src = rst.get("_source", {})
                    docid = src.get("docid")
                    if docid:
                        sparse_docids.append(docid)  # 가져온 모든 docid 수집
                        if docid not in docids:
                            docids.add(docid)
                            documents.append({
                                "content": src.get("content", ""),
                                "docid": docid,
                                "score": rst.get("_score", 0.0)
                            })
                            sparse_count += 1
                docid_info = f" - DocIDs: {sparse_docids}" if getattr(cfg.logging, 'show_retrieved_docids', False) else ""
                log.info(f"Sparse retrieve: {sparse_retrieved}개 검색, {sparse_count}개 추가 (중복 {sparse_retrieved - sparse_count}개){docid_info}")
            # 하이브리드 dense: upstage → sbert 순서로 덧붙임
            if upstage_enabled and dense_ctx and dense_ctx.get('upstage'):
                du = dense_ctx['upstage']
                dense_result = dense_retrieve_upstage(
                    es, du.get('model'), index_name, standalone_query,
                    cfg.retrieve.dense_upstage.top_k, cfg.retrieve.dense_upstage.num_candidates
                )
                upstage_retrieved = len(dense_result['hits']['hits'])
                for rst in dense_result['hits']['hits']:
                    src = rst.get("_source", {})
                    docid = src.get("docid")
                    if docid:
                        upstage_docids.append(docid)  # 가져온 모든 docid 수집
                        if docid not in docids:
                            docids.add(docid)
                            documents.append({
                                "content": src.get("content", ""),
                                "docid": docid,
                                "score": rst.get("_score", 0.0)
                            })
                            upstage_count += 1
                docid_info = f" - DocIDs: {upstage_docids}" if getattr(cfg.logging, 'show_retrieved_docids', False) else ""
                log.info(f"Dense Upstage retrieve: {upstage_retrieved}개 검색, {upstage_count}개 추가 (중복 {upstage_retrieved - upstage_count}개){docid_info}")
            if sbert_enabled and dense_ctx and dense_ctx.get('sbert'):
                ds = dense_ctx['sbert']
                dense_result = dense_retrieve_sbert(
                    es, ds.get('model'), index_name, standalone_query,
                    cfg.retrieve.dense_sbert.top_k, cfg.retrieve.dense_sbert.num_candidates
                )
                sbert_retrieved = len(dense_result['hits']['hits'])
                for rst in dense_result['hits']['hits']:
                    src = rst.get("_source", {})
                    docid = src.get("docid")
                    if docid:
                        sbert_docids.append(docid)  # 가져온 모든 docid 수집
                        if docid not in docids:
                            docids.add(docid)
                            documents.append({
                                "content": src.get("content", ""),
                                "docid": docid,
                                "score": rst.get("_score", 0.0)
                            })
                            sbert_count += 1
                docid_info = f" - DocIDs: {sbert_docids}" if getattr(cfg.logging, 'show_retrieved_docids', False) else ""
                log.info(f"Dense SBERT retrieve: {sbert_retrieved}개 검색, {sbert_count}개 추가 (중복 {sbert_retrieved - sbert_count}개){docid_info}")
            # HyDE 기법을 활용한 Upstage Dense Retrieve
            if upstage_hyde_enabled and dense_ctx and dense_ctx.get('upstage_hyde'):
                duh = dense_ctx['upstage_hyde']
                dense_result = dense_retrieve_upstage_hyde(
                    es, duh.get('model'), index_name, standalone_query,
                    cfg.retrieve.dense_upstage_hyde.top_k, cfg.retrieve.dense_upstage_hyde.num_candidates,
                    client, cfg
                )
                hyde_retrieved = len(dense_result['hits']['hits'])
                for rst in dense_result['hits']['hits']:
                    src = rst.get("_source", {})
                    docid = src.get("docid")
                    if docid:
                        hyde_docids.append(docid)  # 가져온 모든 docid 수집
                        if docid not in docids:
                            docids.add(docid)
                            documents.append({
                                "content": src.get("content", ""),
                                "docid": docid,
                                "score": rst.get("_score", 0.0)
                            })
                            hyde_count += 1
                docid_info = f" - DocIDs: {hyde_docids}" if getattr(cfg.logging, 'show_retrieved_docids', False) else ""
                log.info(f"Dense Upstage HyDE retrieve: {hyde_retrieved}개 검색, {hyde_count}개 추가 (중복 {hyde_retrieved - hyde_count}개){docid_info}")

            # Gemini Dense Retrieve
            if gemini_enabled and dense_ctx and dense_ctx.get('gemini'):
                dg = dense_ctx['gemini']
                dense_result = dense_retrieve_gemini(
                    es, dg.get('model'), index_name, standalone_query,
                    cfg.retrieve.dense_gemini.top_k, cfg.retrieve.dense_gemini.num_candidates
                )
                gemini_retrieved = len(dense_result['hits']['hits'])
                for rst in dense_result['hits']['hits']:
                    src = rst.get("_source", {})
                    docid = src.get("docid")
                    if docid:
                        gemini_docids.append(docid)  # 가져온 모든 docid 수집
                        if docid not in docids:
                            docids.add(docid)
                            documents.append({
                                "content": src.get("content", ""),
                                "docid": docid,
                                "score": rst.get("_score", 0.0)
                            })
                            gemini_count += 1
                docid_info = f" - DocIDs: {gemini_docids}" if getattr(cfg.logging, 'show_retrieved_docids', False) else ""
                log.info(f"Dense Gemini retrieve: {gemini_retrieved}개 검색, {gemini_count}개 추가 (중복 {gemini_retrieved - gemini_count}개){docid_info}")

            # HyDE 기법을 활용한 Gemini Dense Retrieve
            if gemini_hyde_enabled and dense_ctx and dense_ctx.get('gemini_hyde'):
                dgh = dense_ctx['gemini_hyde']
                dense_result = dense_retrieve_gemini_hyde(
                    es, dgh.get('model'), index_name, standalone_query,
                    cfg.retrieve.dense_gemini_hyde.top_k, cfg.retrieve.dense_gemini_hyde.num_candidates,
                    client, cfg
                )
                gemini_hyde_retrieved = len(dense_result['hits']['hits'])
                for rst in dense_result['hits']['hits']:
                    src = rst.get("_source", {})
                    docid = src.get("docid")
                    if docid:
                        gemini_hyde_docids.append(docid)  # 가져온 모든 docid 수집
                        if docid not in docids:
                            docids.add(docid)
                            documents.append({
                                "content": src.get("content", ""),
                                "docid": docid,
                                "score": rst.get("_score", 0.0)
                            })
                            gemini_hyde_count += 1
                docid_info = f" - DocIDs: {gemini_hyde_docids}" if getattr(cfg.logging, 'show_retrieved_docids', False) else ""
                log.info(f"Dense Gemini HyDE retrieve: {gemini_hyde_retrieved}개 검색, {gemini_hyde_count}개 추가 (중복 {gemini_hyde_retrieved - gemini_hyde_count}개){docid_info}")

            # 전체 retrieve 요약 로그 출력
            active_methods = []
            if sparse_enabled and sparse_count > 0:
                active_methods.append(f"Sparse({sparse_count})")
            if upstage_enabled and upstage_count > 0:
                active_methods.append(f"Upstage({upstage_count})")
            if sbert_enabled and sbert_count > 0:
                active_methods.append(f"SBERT({sbert_count})")
            if upstage_hyde_enabled and hyde_count > 0:
                active_methods.append(f"UpstageHyDE({hyde_count})")
            if gemini_enabled and gemini_count > 0:
                active_methods.append(f"Gemini({gemini_count})")
            if gemini_hyde_enabled and gemini_hyde_count > 0:
                active_methods.append(f"GeminiHyDE({gemini_hyde_count})")

            total_docs = len(documents)
            summary = " + ".join(active_methods) if active_methods else "없음"
            log.info(f"📊 Retrieve 요약: {summary} = 총 {total_docs}개 문서")

        # Reranker가 활성화된 경우 reranking 수행
        if cfg.reranker.use_reranker and reranker_tokenizer is not None and reranker_model is not None:
            reranked_documents = rerank_documents(standalone_query, documents, reranker_tokenizer, reranker_model, reranker_aux, cfg, client)
        else:
            # Reranker가 비활성화된 경우 상위 top_k개만 선택
            reranked_documents = documents[:cfg.reranker.top_k]
        
        # 최종 결과를 response에 저장
        retrieved_context = []
        for doc in reranked_documents:
            retrieved_context.append(doc["content"])
            response["topk"].append(doc["docid"])
            response["references"].append({"score": doc["score"], "content": doc["content"]})

        if cfg.qa.use_final_answer:
            # 검색된 컨텍스트로 별도 QA 수행
            content = json.dumps(retrieved_context)
            messages.append({"role": "assistant", "content": content})
            msg = [{"role": "system", "content": cfg.prompts.qa}] + messages
            try:
                qaresult = call_llm_unified(
                    client=client,
                    messages=msg,
                    cfg=cfg
                )
            except Exception:
                traceback.print_exc()
                return response
            # qaresult도 딕셔너리/객체 구분하여 처리
            if isinstance(qaresult, dict):
                response["answer"] = qaresult["choices"][0]["message"]["content"]
            else:
                response["answer"] = qaresult.choices[0].message.content
        else:
            # 현재 방식: 검색 결과만 반환
            if isinstance(result, dict):
                response["answer"] = result["choices"][0]["message"]["content"]
            else:
                response["answer"] = result.choices[0].message.content

    # 검색이 필요하지 않은 경우 바로 답변 생성
    else:
        if isinstance(result, dict):
            response["answer"] = result["choices"][0]["message"]["content"]
        else:
            response["answer"] = result.choices[0].message.content

    return response


# 평가를 위한 파일을 읽어서 각 평가 데이터에 대해서 결과 추출후 파일에 저장
def eval_rag(eval_filename, output_filename, client, cfg, es, index_name, dense_ctx=None):
    log = logging.getLogger(__name__)

    # 리랭킹 모델 로드 전 메모리 사용량 확인
    log_memory_usage(log, "리랭킹 모델 로드 전")

    # 리랭킹 모델을 평가 시작 직전에 초기화 (메모리 효율성을 위해)
    reranker_tokenizer, reranker_model, reranker_aux = initialize_reranker(cfg)

    # 리랭킹 모델 로드 후 메모리 사용량 확인
    log_memory_usage(log, "리랭킹 모델 로드 후")

    general_questions = []  # 일반질문 eval_id, answer 저장 리스트
    general_eval_ids = []   # eval_id만 저장 리스트
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            log.info(f'🚩Test {idx + 1} - Question: {j["msg"]}')
            response = answer_question(j["msg"], client, cfg, es, index_name, dense_ctx, reranker_tokenizer, reranker_model, reranker_aux)
            log.info(f'Answer: {response["answer"]}')
            log.info(f'Retrieved {"👆일반질문👆" if len(response["topk"]) == 0 else len(response["topk"])} documents: {response["topk"]}')
            log.debug(f'References: {len(response["references"])} items')

            # 일반질문일 경우 리스트에 저장
            if len(response["topk"]) == 0:
                general_questions.append({"eval_id": j["eval_id"], "answer": response["answer"]})
                general_eval_ids.append(j["eval_id"])

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1

            if cfg.eval.max_iterations > 0 and idx >= cfg.eval.max_iterations:
                break

    # 일반질문 eval_id 리스트와 갯수 로그에 출력
    log.info(f'일반질문 eval_id 리스트 ({len(general_eval_ids)}개): {general_eval_ids}')

    # chit_chat_ids를 제외한 최종 리스트 로그에 출력
    chit_chat_ids = {2, 32, 57, 64, 67, 83, 90, 94, 103, 218, 220, 222, 227, 229, 245, 247, 261, 276, 283, 301}
    filtered_eval_ids = [eid for eid in general_eval_ids if eid not in chit_chat_ids]
    log.info(f'chit_chat_ids 제외 최종 eval_id 리스트 ({len(filtered_eval_ids)}개): {filtered_eval_ids}')

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info("Starting RAG evaluation process")
    
    # LLM API 키 환경변수 확인 (모델에 따라)
    model_name = cfg.llm.model.lower()
    if "gemini" in model_name:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise ValueError("Gemini 모델 사용시 GEMINI_API_KEY 또는 GOOGLE_API_KEY environment variable is required")
    else:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")

    # LLM 클라이언트 생성 (OpenAI 또는 Gemini)
    client = create_llm_client(cfg)

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

    # Retrieve 백엔드 활성화 여부
    upstage_enabled = getattr(cfg.retrieve.dense_upstage, 'enabled', False)
    sbert_enabled = getattr(cfg.retrieve.dense_sbert, 'enabled', False)
    upstage_hyde_enabled = getattr(cfg.retrieve.dense_upstage_hyde, 'enabled', False)
    gemini_enabled = getattr(cfg.retrieve.dense_gemini, 'enabled', False)
    gemini_hyde_enabled = getattr(cfg.retrieve.dense_gemini_hyde, 'enabled', False)

    # 백엔드별 모델 초기화 (활성화된 경우에만)
    # upstage 또는 upstage_hyde가 활성화된 경우 solar_model 필요
    solar_model = UpstageEmbeddings(model=cfg.retrieve.dense_upstage.model_name) if (upstage_enabled or upstage_hyde_enabled) else None
    sbert_model = SentenceTransformer(cfg.retrieve.dense_sbert.model_name) if sbert_enabled else None

    # Gemini 임베딩 모델 초기화 (gemini 또는 gemini_hyde가 활성화된 경우)
    gemini_model = None
    if gemini_enabled or gemini_hyde_enabled:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Gemini 임베딩 사용시 GOOGLE_API_KEY가 필요합니다.")
        gemini_model = GoogleGenerativeAIEmbeddings(
            model=cfg.retrieve.dense_gemini.model_name,
            google_api_key=google_api_key
        )
        log.info(f"Gemini 임베딩 모델 초기화: {cfg.retrieve.dense_gemini.model_name}")

    # Elasticsearch 인덱스 설정
    # Elasticsearch 9.x에서는 Nori의 품사 태그가 세분화되어
    # 기존의 "E", "J" 등의 대분류 태그가 허용되지 않습니다.
    # 아래는 동일 의미의 세분화된 태그로 교체한 목록입니다.
    # - E(어미): EC, EF, EP, ETN, ETM
    # - J(조사): JKS, JKC, JKG, JKO, JKB, JKV, JKQ, JX, JC
    # - 기호/구두점: SC, SE, SF, SP, SSO, SSC, SY
    # - 보조 용언/계사: VCP, VCN, VX
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
                    "stoptags": [
                        # E (어미)
                        "EC", "EF", "EP", "ETN", "ETM",
                        # J (조사)
                        "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
                        # 기호/구두점
                        "SC", "SE", "SF", "SP", "SSO", "SSC", "SY",
                        # 보조 용언/계사
                        "VCP", "VCN", "VX"
                    ]
                }
            }
        }
    }

    # 활성화된 dense retrieve 방식에 따라 매핑 동적 생성
    mappings = {
        "properties": {
            "content": {"type": "text", "analyzer": "nori"}
        }
    }

    # Upstage 임베딩이 활성화된 경우에만 필드 추가 (일반 upstage 또는 hyde 방식 모두 동일한 필드 사용)
    if upstage_enabled or upstage_hyde_enabled:
        mappings["properties"]["embeddings_upstage"] = {
            "type": "dense_vector",
            "dims": 4096,
            "index": True,
            "similarity": "l2_norm"
        }

    # SBERT 임베딩이 활성화된 경우에만 필드 추가
    if sbert_enabled:
        mappings["properties"]["embeddings_sbert"] = {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "l2_norm"
        }

    # Gemini 임베딩이 활성화된 경우에만 필드 추가 (3072차원)
    if gemini_enabled or gemini_hyde_enabled:
        mappings["properties"]["embeddings_gemini"] = {
            "type": "dense_vector",
            "dims": 3072,
            "index": True,
            "similarity": "l2_norm"
        }

    # 인덱스 생성 또는 재사용
    force_recreate = getattr(cfg.index, 'force_recreate', True)
    index_exists = es.indices.exists(index=cfg.index.name)
    create_es_index(es, cfg.index.name, settings, mappings, force_recreate)

    # 인덱스가 이미 존재하고 force_recreate가 False면 임베딩 및 색인 건너뛰기
    if index_exists and not force_recreate:
        log.info(f'기존 인덱스 "{cfg.index.name}" 재사용 - 임베딩 및 색인 과정 건너뜀')
    else:
        # 문서 임베딩 생성 및 인덱싱
        index_docs = []
        with open(cfg.paths.documents) as f:
            docs = [json.loads(line) for line in f]

        # Upstage(4096 차원 그대로 사용) - 일반 upstage 또는 hyde 방식 모두 동일한 임베딩 사용
        solar_embeds = None
        if (upstage_enabled or upstage_hyde_enabled) and solar_model is not None:
            solar_embeds = get_embeddings_in_batches(docs, solar_model, cfg.embedding.batch_size)

        # SBERT(768)
        sbert_embeds = None
        if sbert_enabled and sbert_model is not None:
            sbert_embeds = []
            bs = cfg.embedding.batch_size
            for i in range(0, len(docs), bs):
                batch = docs[i:i+bs]
                contents = [d["content"] for d in batch]
                sbert_embeds.extend(sbert_get_embedding(contents, sbert_model))

        # Gemini(3072) - 별도 스크립트로 생성된 임베딩 로드
        gemini_embeds = None
        if (gemini_enabled or gemini_hyde_enabled) and gemini_model is not None:
            try:
                from gemini_embedding_generator import GeminiEmbeddingGenerator
                generator = GeminiEmbeddingGenerator(cfg)
                gemini_embeds = generator.get_all_embeddings()

                if gemini_embeds is None:
                    log.warning("Gemini 임베딩이 없습니다. 다음 명령을 먼저 실행하세요:")
                    log.warning("python gemini_embedding_generator.py")
                    log.warning("Gemini 임베딩 없이 계속 진행합니다...")
                    # Gemini 기능 비활성화
                    gemini_enabled = False
                    gemini_hyde_enabled = False
                else:
                    log.info(f"Gemini 임베딩 로드 완료: {len(gemini_embeds)}개")
            except Exception as e:
                log.error(f"Gemini 임베딩 로드 실패: {e}")
                log.warning("Gemini 임베딩 없이 계속 진행합니다...")
                gemini_enabled = False
                gemini_hyde_enabled = False
                gemini_embeds = None

        # 문서에 필요한 필드만 추가하여 색인
        for idx, doc in enumerate(docs):
            if solar_embeds is not None:
                vec = solar_embeds[idx]
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                doc["embeddings_upstage"] = vec
            if sbert_embeds is not None:
                vec = sbert_embeds[idx]
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                doc["embeddings_sbert"] = vec
            if gemini_embeds is not None:
                vec = gemini_embeds[idx]
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                doc["embeddings_gemini"] = vec
            index_docs.append(doc)

        ret = bulk_add(es, cfg.index.name, index_docs)
        log.info(f'Bulk indexing completed: {ret}')

        # 색인 완료 후 불필요한 대용량 메모리 해제 (리랭킹 모델 로드 전 OOM 방지)
        import gc
        log.info('색인 완료 - 불필요한 메모리 해제 시작')

        # 대용량 변수들 메모리 해제
        if 'solar_embeds' in locals() and solar_embeds is not None:
            del solar_embeds
        if 'sbert_embeds' in locals() and sbert_embeds is not None:
            del sbert_embeds
        if 'gemini_embeds' in locals() and gemini_embeds is not None:
            del gemini_embeds
        if 'index_docs' in locals():
            del index_docs
        if 'docs' in locals():
            del docs

        # 가비지 컬렉션 실행
        gc.collect()

        # GPU 메모리 캐시 정리 (PyTorch 사용 시)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info('메모리 해제 완료 - 리랭킹 모델 로드 준비')

    # 평가 데이터에 대해서 결과 생성
    # CSV 파일을 Hydra outputs 디렉토리에 저장
    from hydra.core.hydra_config import HydraConfig
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    output_path = os.path.join(hydra_output_dir, cfg.paths.output)
    
    log.info(f'Current working directory: {os.getcwd()}')
    log.info(f'Hydra output directory: {hydra_output_dir}')
    log.info(f'Starting evaluation with output file: {output_path}')
    dense_ctx = {
        'upstage': {'model': solar_model} if upstage_enabled and solar_model is not None else None,
        'sbert': {'model': sbert_model} if sbert_enabled and sbert_model is not None else None,
        'upstage_hyde': {'model': solar_model} if upstage_hyde_enabled and solar_model is not None else None,
        'gemini': {'model': gemini_model} if gemini_enabled and gemini_model is not None else None,
        'gemini_hyde': {'model': gemini_model} if gemini_hyde_enabled and gemini_model is not None else None,
    }
    eval_rag(cfg.paths.eval_data, output_path, client, cfg, es, cfg.index.name, dense_ctx)
    log.info('RAG evaluation process completed')


if __name__ == "__main__":
    main()
