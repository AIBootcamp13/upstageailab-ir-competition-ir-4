#!/usr/bin/env bash

# uv run rag_with_hybrid_reranker_es9.py retrieve.sparse.enabled=true retrieve.dense_upstage.enabled=false retrieve.dense_sbert.enabled=false reranker.use_reranker=false 
# uv run rag_with_hybrid_reranker_es9.py retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=true retrieve.dense_sbert.enabled=false reranker.use_reranker=false 
# uv run rag_with_hybrid_reranker_es9.py retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_sbert.enabled=true reranker.use_reranker=false 

# uv run rag_with_hybrid_reranker_es9_pca768.py retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=true retrieve.dense_sbert.enabled=false reranker.use_reranker=false 

# uv run rag_with_hybrid_reranker_es9.py retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_upstage_hyde.enabled=true retrieve.dense_sbert.enabled=false reranker.use_reranker=false

# uv run rag_with_hybrid_reranker_es9.py retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_upstage_hyde.enabled=false retrieve.dense_sbert.enabled=false retrieve.dense_gemini.enabled=true retrieve.dense_gemini_hyde.enabled=false reranker.use_reranker=false index.force_recreate=true
# uv run rag_with_hybrid_reranker_es9.py retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_upstage_hyde.enabled=false retrieve.dense_sbert.enabled=false retrieve.dense_gemini.enabled=false retrieve.dense_gemini_hyde.enabled=true reranker.use_reranker=false

# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 reranker.use_hyde=true eval.max_iterations=2

# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 reranker.use_hyde=false
# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 reranker.use_hyde=true
# uv run rag_with_hybrid_reranker_es9.py llm.model="gemini-2.5-flash" llm.delay_seconds=7 reranker.use_hyde=false
# uv run rag_with_hybrid_reranker_es9.py llm.model="gemini-2.5-flash" llm.delay_seconds=7 reranker.use_hyde=true

# uv run rag_with_hybrid_reranker_es9.py retrieve.sparse.enabled=true retrieve.dense_upstage.enabled=false reranker.use_reranker=false



# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_upstage_hyde.enabled=true reranker.use_reranker=false hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_upstage_hyde.enabled=true reranker.use_reranker=false hyde.use_original_query=true


# uv run rag_with_hybrid_reranker_es9.py llm.model="gemini-2.5-flash" llm.delay_seconds=7 retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_gemini_hyde.enabled=true reranker.use_reranker=false hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py llm.model="gemini-2.5-flash" llm.delay_seconds=7 retrieve.sparse.enabled=false retrieve.dense_upstage.enabled=false retrieve.dense_gemini_hyde.enabled=true reranker.use_reranker=false hyde.use_original_query=true


# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 retrieve.dense_gemini.enabled=true retrieve.dense_sbert.enabled=true reranker.use_hyde=true hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 retrieve.dense_gemini.enabled=true retrieve.dense_sbert.enabled=true reranker.use_hyde=true hyde.use_original_query=true

# uv run rag_with_hybrid_reranker_es9.py --config-name=config_hyde_prompt2 llm.model="solar-pro2" llm.delay_seconds=0 retrieve.dense_gemini.enabled=true retrieve.dense_sbert.enabled=true reranker.use_hyde=true hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py --config-name=config_hyde_prompt2 llm.model="solar-pro2" llm.delay_seconds=0 retrieve.dense_gemini.enabled=true retrieve.dense_sbert.enabled=true reranker.use_hyde=true hyde.use_original_query=true

# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 retrieve.dense_gemini.enabled=true reranker.use_hyde=true hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 retrieve.dense_gemini.enabled=true reranker.use_hyde=true hyde.use_original_query=true

# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 reranker.use_hyde=true hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py llm.model="solar-pro2" llm.delay_seconds=0 reranker.use_hyde=true hyde.use_original_query=true

# nohup uv run rag_with_hybrid_reranker_es9.py llm.model="gemini-2.5-flash" llm.delay_seconds=7 reranker.use_hyde=true > run.log 2>&1 &

# pkill -f rag_with_hybrid_reranker_es9.py



# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.model="solar-pro2" llm.delay_seconds=0 hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.model="solar-pro2" llm.delay_seconds=0 hyde.use_original_query=true
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt1)" llm.model="solar-pro2" llm.delay_seconds=0 hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt1)" llm.model="solar-pro2" llm.delay_seconds=0 hyde.use_original_query=true

# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt3)" llm.model="gemini-2.5-flash" llm.delay_seconds=7 hyde.use_original_query=false
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt3)" llm.model="gemini-2.5-flash" llm.delay_seconds=7 hyde.use_original_query=false reranker.use_reranker=false


# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.model="gpt-5" llm.temperature=0.1 llm.delay_seconds=0 hyde.use_original_query=false reranker.use_reranker=false eval.max_iterations=0 llm.temperature=1

# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.model="solar-1-mini-chat" llm.reasoning_effort=
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" reranker.model_name="Qwen/Qwen3-Reranker-0.6B"
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.model="solar-1-mini-chat" llm.reasoning_effort= reranker.model_name="Qwen/Qwen3-Reranker-0.6B"

# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.model="solar-1-mini-chat" llm.reasoning_effort=
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" reranker.model_name="Qwen/Qwen3-Reranker-0.6B"
# uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.model="solar-1-mini-chat" llm.reasoning_effort= reranker.model_name="Qwen/Qwen3-Reranker-0.6B"

uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.reasoning_effort=low reranker.use_hyde=true
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=true
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.reasoning_effort=low reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=true

uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.reasoning_effort=low reranker.use_hyde=true
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=true
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.reasoning_effort=low reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=true


uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.reasoning_effort=low reranker.use_hyde=false
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=false
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.reasoning_effort=low reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=false

uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.reasoning_effort=low reranker.use_hyde=false
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=false
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.reasoning_effort=low reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=false

uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.reasoning_effort=high reranker.use_hyde=false
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.reasoning_effort=high reranker.model_name="Qwen/Qwen3-Reranker-0.6B" reranker.use_hyde=false
uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt5)" llm.reasoning_effort=high reranker.model_name="Qwen/Qwen3-Reranker-8B" reranker.use_hyde=false

uv run rag_with_hybrid_reranker_es9.py --config-name="config_retrieve_(k10_all)_rerank(hyde_prompt2)" llm.reasoning_effort=high reranker.model_name="Qwen/Qwen3-Reranker-8B" reranker.use_hyde=false
