import os
import json
import logging
import numpy as np
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import hydra
from omegaconf import DictConfig

# 현재 스크립트 파일의 디렉토리를 작업 디렉토리로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# .env 파일에서 환경 변수 로드
load_dotenv()

class GeminiEmbeddingGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

        # 저장 디렉토리 설정
        self.output_dir = Path("gemini_embeddings")
        self.output_dir.mkdir(exist_ok=True)

        # 메타데이터 파일 경로
        self.metadata_file = self.output_dir / "metadata.json"

        # Gemini 모델 초기화
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Gemini 임베딩 사용시 GOOGLE_API_KEY가 필요합니다.")

        self.model = GoogleGenerativeAIEmbeddings(
            model=cfg.retrieve.dense_gemini.model_name,
            google_api_key=google_api_key
        )

        # 설정값
        self.batch_size = getattr(cfg.retrieve.dense_gemini, 'batch_size', 100)
        self.batch_delay = getattr(cfg.retrieve.dense_gemini, 'batch_delay_seconds', 65)

        self.log.info(f"Gemini 임베딩 생성기 초기화 완료")
        self.log.info(f"모델: {cfg.retrieve.dense_gemini.model_name}")
        self.log.info(f"배치 크기: {self.batch_size}")
        self.log.info(f"배치 대기 시간: {self.batch_delay}초")

    def load_metadata(self):
        """메타데이터 로드 (진행상황, 설정 등)"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "last_completed_batch": -1,
                "total_documents": 0,
                "total_batches": 0,
                "batch_size": self.batch_size,
                "completed": False,
                "start_time": None,
                "last_update_time": None
            }

    def save_metadata(self, metadata):
        """메타데이터 저장"""
        metadata["last_update_time"] = time.time()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def load_documents(self):
        """문서 로드"""
        docs = []
        with open(self.cfg.paths.documents, 'r', encoding='utf-8') as f:
            for line in f:
                docs.append(json.loads(line))
        return docs

    def save_batch_embeddings(self, batch_idx, embeddings):
        """배치 임베딩 저장"""
        batch_file = self.output_dir / f"batch_{batch_idx:04d}.npy"
        np.save(batch_file, embeddings)
        self.log.info(f"배치 {batch_idx} 임베딩 저장 완료: {batch_file}")

    def load_batch_embeddings(self, batch_idx):
        """배치 임베딩 로드"""
        batch_file = self.output_dir / f"batch_{batch_idx:04d}.npy"
        if batch_file.exists():
            return np.load(batch_file)
        return None

    def is_rate_limit_error(self, error):
        """Rate limit 또는 quota 오류인지 확인"""
        error_str = str(error).lower()
        rate_limit_keywords = [
            "rate limit", "quota", "too many requests",
            "resource_exhausted", "429", "quota exceeded"
        ]
        return any(keyword in error_str for keyword in rate_limit_keywords)

    def generate_embeddings(self):
        """임베딩 생성 (이어받기 지원)"""
        # 메타데이터 로드
        metadata = self.load_metadata()

        # 문서 로드
        docs = self.load_documents()
        total_docs = len(docs)
        total_batches = (total_docs + self.batch_size - 1) // self.batch_size

        # 메타데이터 업데이트
        if metadata["start_time"] is None:
            metadata["start_time"] = time.time()
        metadata["total_documents"] = total_docs
        metadata["total_batches"] = total_batches
        metadata["batch_size"] = self.batch_size

        start_batch = metadata["last_completed_batch"] + 1

        if start_batch == 0:
            self.log.info(f"새로운 임베딩 생성 시작: 총 {total_docs}개 문서, {total_batches}개 배치")
        else:
            self.log.info(f"임베딩 생성 재개: 배치 {start_batch}부터 시작 (총 {total_batches}개 배치)")

        try:
            for batch_idx in range(start_batch, total_batches):
                batch_start_time = time.time()

                # 배치 데이터 준비
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, total_docs)
                batch_docs = docs[start_idx:end_idx]
                contents = [doc["content"] for doc in batch_docs]

                self.log.info(f"[{batch_idx+1}/{total_batches}] 배치 처리 중... (문서 {start_idx+1}-{end_idx}/{total_docs})")

                # 임베딩 생성
                embeddings = self.model.embed_documents(contents)

                # 배치 저장
                self.save_batch_embeddings(batch_idx, embeddings)

                # 메타데이터 업데이트
                metadata["last_completed_batch"] = batch_idx
                self.save_metadata(metadata)

                batch_time = time.time() - batch_start_time
                progress = (batch_idx + 1) / total_batches * 100

                # 진행상황 로그
                if batch_idx > start_batch:
                    elapsed_time = time.time() - metadata["start_time"]
                    avg_batch_time = elapsed_time / (batch_idx - start_batch + 1)
                    remaining_batches = total_batches - (batch_idx + 1)
                    eta_seconds = remaining_batches * avg_batch_time
                    eta_minutes = eta_seconds / 60
                    self.log.info(f"[{batch_idx+1}/{total_batches}] 완료 ({progress:.1f}%) - 배치 처리시간: {batch_time:.1f}초, 예상 남은시간: {eta_minutes:.1f}분")
                else:
                    self.log.info(f"[{batch_idx+1}/{total_batches}] 완료 ({progress:.1f}%) - 배치 처리시간: {batch_time:.1f}초")

                # 마지막 배치가 아니면 대기 (rate limit 회피)
                if batch_idx + 1 < total_batches:
                    self.log.info(f"Rate limit 회피를 위해 {self.batch_delay}초 대기...")
                    time.sleep(self.batch_delay)

        except Exception as e:
            if self.is_rate_limit_error(e):
                self.log.warning(f"Rate limit 또는 quota 도달로 중단됨: {e}")
                self.log.info(f"현재까지 {metadata['last_completed_batch'] + 1}개 배치 완료")
                self.log.info("제한 해제 후 동일한 명령으로 다시 실행하시면 이어서 진행됩니다.")
            else:
                self.log.error(f"예상치 못한 오류 발생: {e}", exc_info=True)
            return False

        # 모든 배치 완료
        metadata["completed"] = True
        metadata["completion_time"] = time.time()
        self.save_metadata(metadata)

        total_time = time.time() - metadata["start_time"]
        self.log.info(f"✅ 모든 임베딩 생성 완료!")
        self.log.info(f"총 {total_docs}개 문서, {total_batches}개 배치, 총 소요시간: {total_time/60:.1f}분")

        # 병합된 파일 생성
        self.merge_all_batches()
        return True

    def merge_all_batches(self):
        """모든 배치를 하나의 파일로 병합"""
        metadata = self.load_metadata()
        if not metadata["completed"]:
            self.log.warning("아직 모든 배치가 완료되지 않았습니다.")
            return

        self.log.info("모든 배치를 병합하는 중...")
        all_embeddings = []

        for batch_idx in range(metadata["total_batches"]):
            embeddings = self.load_batch_embeddings(batch_idx)
            if embeddings is not None:
                all_embeddings.extend(embeddings)
            else:
                self.log.error(f"배치 {batch_idx} 로드 실패")
                return

        # 병합된 파일 저장
        merged_file = self.output_dir / "all_embeddings.npy"
        np.save(merged_file, all_embeddings)

        self.log.info(f"병합 완료: {len(all_embeddings)}개 임베딩 -> {merged_file}")

    def get_all_embeddings(self):
        """모든 임베딩 반환 (병합된 파일이 있으면 사용, 없으면 배치별로 로드)"""
        merged_file = self.output_dir / "all_embeddings.npy"

        if merged_file.exists():
            self.log.info(f"병합된 임베딩 파일 로드: {merged_file}")
            return np.load(merged_file)

        # 배치별로 로드
        metadata = self.load_metadata()
        if not metadata["completed"]:
            self.log.error("임베딩 생성이 완료되지 않았습니다. gemini_embedding_generator.py를 먼저 실행하세요.")
            return None

        self.log.info("배치별 임베딩 파일들을 로드하는 중...")
        all_embeddings = []

        for batch_idx in range(metadata["total_batches"]):
            embeddings = self.load_batch_embeddings(batch_idx)
            if embeddings is not None:
                all_embeddings.extend(embeddings)
            else:
                self.log.error(f"배치 {batch_idx} 로드 실패")
                return None

        return np.array(all_embeddings)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    log = logging.getLogger(__name__)
    log.info("Gemini 임베딩 생성기 시작")

    try:
        generator = GeminiEmbeddingGenerator(cfg)
        success = generator.generate_embeddings()

        if success:
            log.info("🎉 모든 작업이 성공적으로 완료되었습니다!")
        else:
            log.info("⏸️  작업이 중단되었습니다. 나중에 다시 실행하시면 이어서 진행됩니다.")

    except Exception as e:
        log.error(f"치명적 오류: {e}", exc_info=True)

if __name__ == "__main__":
    main()