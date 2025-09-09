# Information Retrieval 경진대회 - 과학 지식 질의 응답 시스템 구축
## Team 4조

| ![류지헌](https://avatars.githubusercontent.com/u/10584296?v=4) | ![김태현](https://avatars.githubusercontent.com/u/7031901?v=4) | ![박진섭](https://avatars.githubusercontent.com/u/208775216?v=4) | ![문진숙](https://avatars.githubusercontent.com/u/204665219?v=4) | ![김재덕](https://avatars.githubusercontent.com/u/33456585?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [류지헌](https://github.com/mahomi)             |            [김태현](https://github.com/huefilm)             |            [박진섭](https://github.com/seob1504)             |            [문진숙](https://github.com/June3723)             |            [김재덕](https://github.com/ttcoaster)             |
|                   팀장, RAG 아키텍처 설계<br/>RAG 파이프라인 구현                   |                   문서 전처리 및 분할<br/>검색 최적화                   |                   임베딩 및 벡터 저장소<br/>성능 튜닝                   |                   프롬프트 엔지니어링<br/>답변 품질 개선                   |                   API 통합 및 배포<br/>환경 설정 관리                   |

## 0. Overview
### Environment
- OS: Linux (테스트: Ubuntu 20.04/22.04 계열)
- Python: 3.10+

### Requirements
- Python 패키지: `uv`(권장)
- Elasticsearch 8.x (로컬 설치 스크립트 제공)
- 모델/키: OpenAI API 또는 Upstage 호환 OpenAI API, Sentence-Transformers

## 1. Competiton Info

### Overview

- 과학 지식 질의 응답 시스템 구축: 질문과 이전 대화 히스토리를 바탕으로 검색엔진에서 관련 문서를 추출하고, 이를 활용해 적합한 답변을 생성하는 RAG 태스크입니다.

### Timeline

- 2025-09-08 10:00 대회 시작
- 2025-09-18 19:00 최종 마감

## 2. Components

### Directory

```
├── code
│   ├── baseline
│   │   ├── rag_with_elasticsearch.py      # 베이스라인 단일 스크립트 (RAG + ES)
│   │   ├── README.md                      # 베이스라인 실행 가이드
│   │   ├── install_elasticsearch.sh       # 로컬 Elasticsearch 8.x 설치 스크립트
│   │   └── env_template.txt               # .env 템플릿 (ES/LLM 키)
│   └── rag_with_hydra
│       ├── rag_with_hydra.py              # Hydra 기반 RAG 실행 스크립트
│       ├── env_template.txt               # .env 템플릿 (ES/LLM 키)
│       └── conf
│           └── config.yaml                # Hydra 구성 기반 고급 설정 예시
├── input
│   └── data
│       ├── documents.jsonl                # 문서 코퍼스
│       └── eval.jsonl                     # 평가 데이터
└── README.md
```

### baseline 코드 실행
```bash
cd code/baseline

# .env 생성하고 LLM API 키 입력
cp env_template.txt .env
# OPENAI_API_KEY에 upstage api key입력

# Elasticsearch 설치 (아직 설치 안한 경우만 설치)
./install_elasticsearch.sh
# "Please confirm that you would like to continue"에서 y 입력하고, 출력되는 비빌번호를 .env에 입력할것.

uv run rag_with_elasticsearch.py
```

실행 시 동작 개요
- Elasticsearch 인덱스 `test` 생성(한국어 `nori` 분석기 + `dense_vector` 필드)
- `input/data/documents.jsonl` 임베딩 생성 후 대량 색인
- 검색 데모: 질의에 대해 역색인(sparse)·벡터(dense) 검색 예시 출력
- RAG 파이프라인: 함수 호출형 질의분석 → 검색 → QA 모델로 최종 답변 생성
- 평가 결과 파일 생성: `code/baseline/sample_submission.csv`


## 3. Data descrption

### Dataset overview
 - `documents.jsonl`: 검색 대상 과학 지식 문서 코퍼스 (총 4,272행, 필드 예시: `docid`, `content`)
 - `eval.jsonl`: 평가용 질의 세트 (총 220행, 필드 예시: `eval_id`, `msg`)

### EDA

- 간단 점검: 문서 길이 분포, 중복 여부, 섹션/문단 단위 분할 필요성 검토
- 한국어 토크나이저(`nori`)와 SBERT 임베딩 적합성 확인

### Data Processing

- 텍스트 클리닝(공백/특수문자 정리), 문서 분할(필요 시), 메타데이터 정규화
- 임베딩: `snunlp/KR-SBERT-V40K-klueNLI-augSTS`로 768차원 벡터 생성(batch 처리)
- 인덱스: 역색인(`content`), 벡터(`embeddings`) 동시 구축

## 4. Modeling

### Model descrition
- Retrieval: Elasticsearch 8.x
  - Sparse: `match` 쿼리(`nori` 분석기)
  - Dense: KNN(`dense_vector` with `l2_norm`)
- Embedding: Sentence-Transformers 한국어 SBERT(`snunlp/KR-SBERT-V40K-klueNLI-augSTS`)
- LLM: OpenAI 호환(Chat Completions)
  - 기본값 모델: `gpt-4o-mini`(환경변수로 변경 가능)
  - 프롬프트: QA용/Function-Calling용 분리 설계

### Modeling Process
- 질의 분석(Function Calling) → `standalone_query` 생성
- 검색: baseline은 기본적으로 sparse 검색 사용(확장: dense 재랭킹/혼합)
- QA: 검색된 컨텍스트를 시스템 프롬프트와 함께 LLM에 전달하여 최종 답변 생성
- 평가: `eval.jsonl`을 순회하며 `standalone_query`, `topk`, `answer`, `references` 기록

## 5. Result

### Leader Board

- 제출 파일: `sample_submission.csv` 또는 `submission.csv` 형식(JSON Lines → CSV 확장자)
- 점수 산출: `topk` 기반 평가(답변 텍스트는 자동평가 시 보조 용도)

### Presentation

- 발표 자료 업로드 예정

## etc

### Meeting Log

- Issues : https://github.com/AIBootcamp13/upstageailab-ir-competition-ir-4/issues

### Reference

- Elasticsearch 8.x KNN 검색
- Sentence-Transformers 한국어 SBERT
- OpenAI(Chat Completions) 호환 API

---

## 부록: 설정과 트러블슈팅

### 환경변수 요약(.env)
```
ELASTICSEARCH_PASSWORD=필수
OPENAI_API_KEY=필수
OPENAI_BASE_URL=선택
OPENAI_MODEL=선택(기본: gpt-4o-mini)
```

### 경로/권한 이슈
- `ca_certs` 경로는 설치 스크립트 출력 경로(`/data/ephemeral/home/elasticsearch-8.8.0/config/certs/http_ca.crt`)를 사용합니다.
- 로컬 환경에 따라 경로가 다를 수 있으니, 필요 시 `code/baseline/rag_with_elasticsearch.py` 내 Elasticsearch 클라이언트 생성부를 수정하세요.

### 확장 포인트
- Dense 검색 가중치/재랭킹 혼합, 하이브리드 검색 성능 개선
- Hydra 구성(`code/rag_with_hydra/conf/config.yaml`) 기반의 파이프라인 파라미터화 및 실험 관리

<br>

---

<br>

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AIBootcamp13/upstageailab-ir-competition-ir-4)
