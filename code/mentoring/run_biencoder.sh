# 모델이 너무 커서 CUDA_OUT_OF_MEMORY 에러가 발생한다면, 모델을 아래 중에 더 작은 걸로 변경해보세요!
# Qwen/Qwen3-Embedding-0.6B
# Qwen/Qwen3-Embedding-4B
# Qwen/Qwen3-Embedding-8B
# 혹은 batch_size를 1로 줄여보세요!

# flash attention 관련 에러가 생기면, --use_flash_attention을 지우고 실행해보세요!

# 현재 코드는 20개의 일상대화를 모두 포함한 상태에서 수행되었고, "멀티턴 대화의 모든 사용자 발화를 단순히 공백으로 연결하여 하나의 긴 쿼리로 변환하는 전략"을 사용하고 있습니다.

# Bi-encoder 실행
python biencoder.py \
    --model_name Qwen/Qwen3-Embedding-8B \
    --documents_path ../data/documents.jsonl \
    --eval_path ../data/eval.jsonl \
    --scores_path ./similarity_scores.csv \
    --output_path ./reranked_submission.csv \
    --doc_embeddings_path ./doc_embeddings.pkl \
    --top_k 3 \
    --batch_size 16 \
    --use_flash_attention \
    --recompute_embeddings # 해당 인자를 포함하면 "기존 임베딩 파일이 있어도 문서 임베딩을 재계산합니다." 만약, 임베딩 파일이 저장되어 있다면, 없애주세요!

echo "Bi-Encoder 검색 완료!"
echo "결과 파일: reranked_submission.csv"
echo "점수 파일: similarity_scores.csv"