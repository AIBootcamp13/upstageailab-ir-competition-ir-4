#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_reranker_loading():
    """리랭커 모델만 단독으로 로드하여 테스트"""
    model_name = "Qwen/Qwen3-Reranker-8B"
    
    try:
        log.info(f"GPU 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            # PyTorch 메모리 캐시 정리
            log.info("PyTorch GPU 메모리 캐시 정리 중...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            log.info(f"GPU 메모리 정보: {torch.cuda.get_device_properties(0)}")
            log.info(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            log.info(f"GPU 메모리 캐시 사용량: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        log.info(f"Loading reranker model: {model_name}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        log.info("Tokenizer loaded successfully")
        
        # 직접 GPU에서 float16으로 모델 로드
        if torch.cuda.is_available():
            log.info("직접 GPU에서 float16으로 모델 로드 시도...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                dtype=torch.float16,
                device_map="auto"
            ).eval()
            log.info("Model loaded on GPU with float16 precision successfully")
            log.info(f"GPU 메모리 사용량 (모델 로드 후): {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            # CPU fallback
            model = AutoModelForCausalLM.from_pretrained(model_name).eval()
            log.info("Model loaded on CPU successfully")
        
        # 간단한 테스트 입력
        test_text = "<Instruct>: Given a web search query, retrieve relevant passages\n<Query>: 테스트 쿼리\n<Document>: 테스트 문서"
        
        log.info("Testing model inference...")
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # device_map="auto"로 로드했으므로 입력을 모델과 같은 디바이스로 이동
        if hasattr(model, 'device') and model.device.type == 'cuda':
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            log.info("Model inference completed successfully")
            log.info(f"Output shape: {outputs.logits.shape}")
            
            # 메모리 정보 추가 출력
            if torch.cuda.is_available():
                log.info(f"최종 GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                log.info(f"최종 GPU 메모리 캐시: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        log.info("리랭커 모델 로드 및 테스트 완료!")
        return True
        
    except Exception as e:
        log.error(f"리랭커 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reranker_loading()
    if success:
        print("✅ 리랭커 모델이 정상적으로 로드되었습니다!")
    else:
        print("❌ 리랭커 모델 로드에 실패했습니다.")