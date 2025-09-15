# Gemini 임베딩 배치별 생성 가이드

Gemini API의 일일 제한량으로 인해 전체 문서 임베딩을 한 번에 처리하기 어려울 때 사용하는 시스템입니다.

## 🎯 주요 기능

- **배치별 저장**: 각 배치 완료 시마다 결과를 파일에 저장
- **이어받기**: 중단된 지점부터 다시 시작
- **오류 처리**: API 제한 도달 시 우아한 종료
- **진행상황 추적**: 실시간 진행상황 및 예상 시간 표시

## 📁 저장 구조

```
gemini_embeddings/
├── metadata.json          # 진행상황, 설정 정보
├── batch_0000.npy        # 첫 번째 배치 임베딩
├── batch_0001.npy        # 두 번째 배치 임베딩
├── ...
└── all_embeddings.npy    # 모든 배치 병합된 최종 파일 (완료 후)
```

## 🚀 사용 방법

### 1. Gemini 임베딩 생성 시작

```bash
cd /root/dev/upstageailab-ir-competition-ir-4/code/rag_with_hydra_reranker
python gemini_embedding_generator.py
```

### 2. 제한 도달 시 자동 중단

API 제한에 도달하면 자동으로 중단되고 현재까지의 진행상황이 저장됩니다:

```
⚠️  Rate limit 또는 quota 도달로 중단됨: ...
📊 현재까지 15개 배치 완료
💡 제한 해제 후 동일한 명령으로 다시 실행하시면 이어서 진행됩니다.
```

### 3. 다음날 이어서 실행

동일한 명령을 다시 실행하면 자동으로 중단된 지점부터 계속됩니다:

```bash
python gemini_embedding_generator.py
```

출력 예시:
```
🔄 임베딩 생성 재개: 배치 16부터 시작 (총 50개 배치)
```

### 4. 완료 후 RAG 시스템 실행

모든 임베딩 생성이 완료되면 기존 RAG 시스템을 실행합니다:

```bash
python rag_with_hybrid_reranker_es9.py
```

## ⚙️ 설정

`conf/config.yaml`에서 배치 설정을 조정할 수 있습니다:

```yaml
retrieve:
  dense_gemini:
    batch_size: 100              # 한 번에 처리할 문서 수
    batch_delay_seconds: 65      # 배치 간 대기 시간 (초)
```

## 📊 진행상황 확인

진행상황은 실시간으로 표시됩니다:

```
[16/50] 완료 (32.0%) - 배치 처리시간: 45.2초, 예상 남은시간: 25.7분
```

`metadata.json` 파일에서도 현재 진행상황을 확인할 수 있습니다:

```json
{
  "last_completed_batch": 15,
  "total_documents": 5000,
  "total_batches": 50,
  "completed": false,
  "start_time": 1703123456.789
}
```

## 🔧 문제 해결

### Q: 임베딩 파일이 손상되었나요?

각 배치가 개별 파일로 저장되므로 일부 배치만 재생성하면 됩니다. 손상된 배치 파일을 삭제하고 다시 실행하세요.

### Q: 처음부터 다시 시작하고 싶어요

`gemini_embeddings/` 폴더를 삭제하고 다시 실행하세요:

```bash
rm -rf gemini_embeddings/
python gemini_embedding_generator.py
```

### Q: 진행상황이 이상해요

`metadata.json`을 삭제하면 처음부터 다시 시작됩니다:

```bash
rm gemini_embeddings/metadata.json
python gemini_embedding_generator.py
```

## 💡 팁

1. **배치 크기 조정**: API 제한에 자주 걸리면 `batch_size`를 줄여보세요
2. **대기 시간 증가**: `batch_delay_seconds`를 늘리면 제한에 덜 걸립니다
3. **백그라운드 실행**: `nohup python gemini_embedding_generator.py &`로 백그라운드에서 실행 가능

## 🎉 완료 확인

모든 작업이 완료되면 다음과 같은 메시지를 볼 수 있습니다:

```
✅ 모든 임베딩 생성 완료!
📊 총 5000개 문서, 50개 배치, 총 소요시간: 120.5분
📁 병합 완료: 5000개 임베딩 -> gemini_embeddings/all_embeddings.npy
🎉 모든 작업이 성공적으로 완료되었습니다!
```