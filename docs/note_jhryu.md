retrieve method | llm | MAP | MRR
-- | -- | -- | --
dense_gemini_embedding_hyde(org) | solar-pro2 | - | -
dense_gemini_embedding_hyde | solar-pro2 | - | -
dense_gemini_embedding | solar-pro2 | 0.8985 | 0.9030
dense_upstage_embedding_hyde(org) | solar-pro2 | 0.7667 | 0.7682
dense_upstage_embedding_hyde | solar-pro2 | 0.8197 | 0.8227
dense_upstage_embedding(PCA768) | solar-pro2 | 0.8939 | 0.8970
dense_upstage_embedding | solar-pro2 | 0.8970 | 0.8985
dense_sbert_embedding | solar-pro2 | 0.5053 | 0.5061
sparse (bm25) | solar-pro2 | 0.7742 | 0.7788
dense_gemini_embedding_hyde(org) | gemini-2.5-flash | - | -
dense_gemini_embedding_hyde | gemini-2.5-flash | - | -
dense_gemini_embedding | gemini-2.5-flash | - | -
dense_upstage_embedding_hyde(org) | gemini-2.5-flash | - | -
dense_upstage_embedding_hyde | gemini-2.5-flash | - | -
dense_upstage_embedding(PCA768) | gemini-2.5-flash | - | -
dense_upstage_embedding | gemini-2.5-flash | - | -
dense_sbert_embedding | gemini-2.5-flash | - | -
sparse (bm25) | gemini-2.5-flash | - | -


*일반상식분류건중 chit-chat 20건을 빼면 잘못분류한건수가 됨
| Parameters | 일반상식분류건 | MAP | MRR |
|---|---|---|---|
| model: solarpro2, reasoning_effort: high, sparse, function_calling프롬프트 변경4, rerank(topk30/btch4) | 21 | 0.8720 | 0.8742 |
| model: solarpro2, reasoning_effort: high, sparse, function_calling프롬프트 변경4, rerank(topk30/btch2) | 22 | 0.8682 | 0.8727 |
| model: solarpro2, reasoning_effort: high, sparse, function_calling프롬프트 변경3 | 21 | 0.7576 | 0.7636 |
| model: solarpro2, reasoning_effort: high, sparse, function_calling프롬프트 변경3 | 24 | 0.7409 | 0.7455 |
| model: solarpro2, reasoning_effort: high, sparse, function_calling프롬프트 변경2, standalone_query_description 변경 | 33 | 0.6985 | 0.7030 |
| model: solarpro2, reasoning_effort: high, sparse, function_calling프롬프트 변경1, standalone_query_description 변경 | 40 | 0.7015 | 0.7091 |
| model: solarpro2, reasoning_effort: high, sparse, function_calling프롬프트 변경1 | 57 | 0.4682 | 0.4712 |
| model: solarpro2, reasoning_effort: low, sparse, function_calling프롬프트 변경1 | 45 | 0.6523 | 0.6561 |
| model: solarpro2 | 127 | 0.3402 | 0.3394 |
| model: gpt-4o-mini | 18 | 0.4242 | 0.4242 |
- function_calling프롬프트 변경4 :
    - 먼저 질문이 chit-chat 즉, 지극히 개인적질문이나 안부, 인사, 감정에 대한 질문 일 경우에는 절대로 search api를 호출하지 말고 적절한 대답을 생성한다.
    - chit-chat 즉, 지극히 개인적질문이나 안부, 인사, 감정에 대한 질문 이외에는 무조건 반드시 search api를 호출해야 하고 스스로 판단해서 대답하면 안된다.
    - 과학 상식 범위 내에서 설명 가능하더라도 반드시 search api를 호출해야 한다.
    - 추가 검색 없이도 과학 상식으로 설명 가능한 범위이더라도 반드시 search api를 호출해야 한다.
    - 어떤 사건에 관한것이거나 코드 프로그램적인 질문을 하더라도 반드시 search api를 호출해야 한다.
    - search api를 호출할 때는 반드시 질문에 대한 최종 korean query를 생성해서 호출해야 한다.
- function_calling프롬프트 변경3 : (과학질문을 구분해내는것이 아니라 chit-chat만제외하도록)
    - 먼저 질문이 chit-chat 즉, 지극히 개인적질문이나 안부, 인사, 감정에 대한 질문 일 경우에는 절대로 search api를 호출하지 말아야함. 적절한 대답을 생성한다.
    - chit-chat 즉, 지극히 개인적질문이나 안부, 인사, 감정에 대한 질문 이외는 무조건 반드시 search api를 호출해야 하고 스스로 판단해서 대답하면 안된다.
    - 해당 내용이 과학 상식 범위 내에서 설명 가능하며, 추가 검색 없이 답변할 수 있더라도 반드시 search api를 호출해야 한다.
- function_calling프롬프트 변경2 :
    - 지식에 대한 질문이 아닌 chit-chat 혹은 잡담 대화 메시지에는 search api를 호출하지 말고 적절한 대답을 생성한다.
    - 그외 사용자가 대화를 통해 상식 혹은 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다. 일반 상식선에서 답변이 가능하더라도 무조건 search api를 호출해야 한다.
- function_calling프롬프트 변경1 :  "일반 상식선에서 답변이 가능하더라도 무조건 search api를 호출" 추가
- standalone_query_description 변경 : query -> korean query 변경


## 일상 질문 20개 구분하기 (RAG대상 아님)
```
{"eval_id": 2, "msg": [{"role": "user", "content": "이제 그만 얘기하자."}]}
{"eval_id": 32, "msg": [{"role": "user", "content": "오늘 너무 즐거웠다!"}]}
{"eval_id": 57, "msg": [{"role": "user", "content": "우울한데 신나는 얘기 좀 해줄래?"}]}
...
```

## 멀티 턴 대화문 20개 (RAG대상에 포함됨)
```
{"eval_id": 3, "msg": [{"role": "user", "content": "동물들이 종종 집단으로 이주하는 경우가 발생하더라구?"}, {"role": "assistant", "content": "네 맞습니다."}, {"role": "user", "content": "이렇게 집단으로 이주하게 되는 계기는 어떤 것들이 있어?"}]}
{"eval_id": 33, "msg": [{"role": "user", "content": "Python 공부중이야."}, {"role": "assistant", "content": "네 python은 배워 두시면 유용합니다."}, {"role": "user", "content": "list의 값중 가장 작은 값을 알려주는 함수가 뭐야?"}]}
{"eval_id": 39, "msg": [{"role": "user", "content": "새로 만든 항생제가 드디어 나왔어."}, {"role": "assistant", "content": "네 무엇을 도와 드릴까요?"}, {"role": "user", "content": "그 효과를 확실히 얘기하기 위해서 해야할 일은?"}]}
...
```

k10_all_hard_voting_rank5
0.9121/0.9121, 0.9152/0.9152

dense_gemini_embeddding
0.8985/0.9182, 0.9030/0.9182


solar_k10_all_rerankHyde_prompt2
0.9167/0.8417, 0.9197/0.8424

solar_k10_all_rerankHyde_prompt1
0.8811/0.8303, 0.8833/0.8348

prompt1
  hyde: "다음 질문에 대해 전문적이고 사실과 정확한 정보를 바탕으로 핵심 정보가 포함된 요약 답변을 작성해줘."
promtpt2
  hyde: |
    질문에 대해서 아래 예시와 같은 문장 스타일과 길이로 답변해줘.
    
    - 예시 답변 1 :
    건강한 사람이 에너지 균형을 평형 상태로 유지하는 것은 중요합니다. 에너지 균형은 에너지 섭취와 에너지 소비의 수학적 동등성을 의미합니다. 일반적으로 건강한 사람은 1-2주의 기간 동안 에너지 균형을 달성합니다. 이 기간 동안에는 올바른 식단과 적절한 운동을 통해 에너지 섭취와 에너지 소비를 조절해야 합니다. 식단은 영양가 있는 식품을 포함하고, 적절한 칼로리를 섭취해야 합니다. 또한, 운동은 에너지 소비를 촉진시키고 근육을 강화시킵니다. 이렇게 에너지 균형을 유지하면 건강을 유지하고 비만이나 영양 실조와 같은 문제를 예방할 수 있습니다. 따라서 건강한 사람은 에너지 균형을 평형 상태로 유지하는 것이 중요하며, 이를 위해 1-2주의 기간 동안 식단과 운동을 조절해야 합니다.
    - 예시 답변 2 :
    수소, 산소, 질소 가스의 혼합물에서 평균 속도가 가장 빠른 분자는 수소입니다. 수소 분자는 가장 가볍고 작은 원자로 구성되어 있기 때문에 다른 분자들보다 더 빠르게 움직입니다. 이러한 이유로 수소 분자는 주어진 온도에서 가장 빠른 평균 속도를 가지고 있습니다. 수소 분자는 화학 반응에서도 활발하게 참여하며, 수소 연료로도 널리 사용됩니다. 따라서 수소 분자는 주어진 온도에서 평균 속도가 가장 빠른 분자입니다.



hard_voting_10(531)
0.9424/0.8985, 0.9439/0.9000
