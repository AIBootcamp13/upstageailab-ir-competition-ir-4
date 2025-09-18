### Flash Attention 환경 설정

#### Flash Attention의 장점
Flash Attention은 Transformer 모델의 attention 메커니즘을 최적화한 기술로, 다음과 같은 주요 장점을 제공합니다:

- **메모리 효율성**: 기존 attention 계산의 O(N²) 메모리 복잡도를 O(N)으로 줄여 대용량 시퀀스 처리 가능
- **계산 속도 향상**: GPU 메모리 대역폭을 효율적으로 활용하여 2-4배 빠른 attention 계산
- **확장성**: 긴 시퀀스 길이에서도 안정적인 성능 유지
- **정확성 보장**: 수치적으로 동일한 결과를 보장하면서도 메모리와 속도 최적화
- **GPU 메모리 절약**: 대용량 모델(Qwen3-Reranker-8B 등)에서 메모리 부족 문제 해결

#### Flash Attention 설치 시 발생하는 주요 문제점

**1. GLIBC 버전 호환성 문제**
- **문제**: Flash Attention 2가 컴파일된 바이너리가 `GLIBC_2.32` 심볼을 요구하지만, 대부분의 Ubuntu 시스템은 더 낮은 버전의 glibc를 사용
- **증상**: `ImportError: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.32' not found` 오류 발생
- **해결**: 별도 위치에 glibc 2.32를 빌드하여 설치하고 로더로 사용

**2. CUDA 버전 호환성 문제**
- **문제**: Flash Attention 빌드 시 CUDA 12.4 개발 도구가 필요하지만 시스템에 설치되지 않음
- **증상**: 컴파일 오류 또는 런타임 CUDA 라이브러리 오류
- **해결**: CUDA Toolkit 12.4 및 빌드 도구 설치

**3. GPU 아키텍처 호환성 문제**
- **문제**: Flash Attention이 특정 GPU 컴퓨트 캐파빌리티를 요구
- **증상**: 빌드 실패 또는 런타임 오류
- **해결**: `TORCH_CUDA_ARCH_LIST` 환경변수로 GPU 아키텍처 지정 (예: RTX 3090 → "8.6")

#### 해결 방법 및 설치 절차

**1. 환경 점검**
```bash
# GPU/드라이버 확인
nvidia-smi

# PyTorch와 CUDA 버전 확인
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

**2. CUDA 12.4 개발 도구 설치**
```bash
# NVIDIA CUDA 레포지토리 키 추가
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update

# CUDA Toolkit 및 빌드 도구 설치
sudo apt-get install -y cuda-toolkit-12-4
sudo apt-get install -y ninja-build gawk bison m4
```

**3. glibc 2.32 빌드 및 설치**
```bash
cd /tmp
wget https://ftp.gnu.org/gnu/libc/glibc-2.32.tar.xz
mkdir glibc-2.32-src glibc-build
tar -xf glibc-2.32.tar.xz -C glibc-2.32-src --strip-components=1
cd glibc-build
../glibc-2.32-src/configure --prefix=/opt/glibc-2.32
make -j$(nproc)
sudo make install
```

**4. flash-attn 소스에서 빌드 설치**
```bash
# 기존 설치 제거
uv pip uninstall flash-attn

# 소스에서 빌드하여 설치
FLASH_ATTENTION_FORCE_BUILD=1 \
TORCH_CUDA_ARCH_LIST="8.6" \
CUDA_HOME=/usr/local/cuda-12.4 \
PATH=/usr/local/cuda-12.4/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
uv add --no-build-isolation --no-cache --no-binary-package flash-attn \
  flash-attn@git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3
```

**5. 설치 검증**
```bash
# glibc 2.32 로더를 사용하여 Flash Attention import 테스트
/opt/glibc-2.32/lib/ld-linux-x86-64.so.2 \
  --library-path /opt/glibc-2.32/lib:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  .venv/bin/python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
```

#### enable_flash_attn_env.sh 스크립트
이 프로젝트에서는 `scripts/enable_flash_attn_env.sh` 스크립트를 통해 Flash Attention 환경을 설정합니다:

**주요 기능:**
- **glibc 2.32 환경 구성**: Flash Attention이 요구하는 glibc 2.32 로더 설정
- **CUDA 라이브러리 경로 설정**: CUDA 12.4 라이브러리 경로 자동 구성
- **uv 명령어 래핑**: `uv run python` 명령어가 자동으로 glibc 2.32 래퍼를 사용하도록 설정
- **타임존 설정**: 한국 표준시(KST) 자동 적용

**사용법:**
```bash
# Flash Attention 환경 활성화 (uv run 명령어 실행 전에 필수)
source scripts/enable_flash_attn_env.sh

# 이후 모든 uv run python 명령어가 Flash Attention 환경에서 실행됨
uv run python rag_with_hybrid_reranker_es9_voting.py
```

**환경 변수:**
- `GLIBC_ROOT`: glibc 2.32 설치 경로 (기본값: `/opt/glibc-2.32`)
- `CUDA_ROOT`: CUDA 설치 경로 (기본값: `/usr/local/cuda-12.4`)
- `FLASH_ATTN_LD_LIBRARY_PATH`: 라이브러리 검색 경로 자동 설정
- `TZ`: 타임존 설정 (기본값: `Asia/Seoul`)

#### 트러블 슈팅 팁
- Flash Attention 관련 오류가 발생하면 `--use_flash_attention` 옵션을 제거한 뒤 재실행하여 GPU 메모리나 환경 문제를 우선 확인
- CUDA Toolkit과 glibc는 시스템 전역 변경이므로, 다른 서비스와 충돌이 없는지 점검 후 작업
- GPU 아키텍처는 `nvidia-smi`로 확인 후 `TORCH_CUDA_ARCH_LIST`에 적절한 값 설정

이 스크립트를 통해 Flash Attention을 사용하는 모델들(Qwen3-Reranker-8B 등)이 최적의 성능으로 실행됩니다.