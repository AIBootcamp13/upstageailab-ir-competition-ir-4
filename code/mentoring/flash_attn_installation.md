# flash-attn 설치 가이드

이 문서는 `code/mentoring/run_reranker2.sh`가 Flash Attention 2를 사용해 정상 실행되도록 준비하는 절차를 정리한 것입니다. 아래 순서를 다른 서버에서도 그대로 따라 하면 됩니다.

## 1. 환경 점검
- GPU/드라이버 확인
  ```bash
  nvidia-smi
  ```
- 현재 PyTorch와 CUDA 빌드 확인 (프로젝트 루트에서 실행)
  ```bash
  uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
  ```
  *본 가이드에서는 `torch 2.8.0+cu128` 기준으로 작업했습니다.*

## 2. CUDA 12.4 개발 도구 설치
1. NVIDIA CUDA 레포지토리 키 추가
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
   sudo dpkg -i /tmp/cuda-keyring.deb
   sudo apt-get update
   ```
2. CUDA Toolkit 및 빌드 도구 설치
   ```bash
   sudo apt-get install -y cuda-toolkit-12-4
   sudo apt-get install -y ninja-build gawk bison m4
   ```
3. 확인
   ```bash
   /usr/local/cuda-12.4/bin/nvcc --version
   ```

## 3. glibc 2.32 빌드 및 설치
Flash Attention 2가 생성한 바이너리가 `GLIBC_2.32` 심볼을 요구하므로, 별도 위치에 glibc 2.32를 설치해 로더로 사용합니다.

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

설치 후 `/opt/glibc-2.32/lib/ld-linux-x86-64.so.2` 로더가 생성되어 있어야 합니다.

## 4. flash-attn 설치
1. 기존 설치 제거(필요 시)
   ```bash
   uv pip uninstall flash-attn
   ```
2. 소스에서 다시 빌드하여 설치 (프로젝트 루트에서 실행)
   ```bash
   FLASH_ATTENTION_FORCE_BUILD=1 \
   TORCH_CUDA_ARCH_LIST="8.6" \
   CUDA_HOME=/usr/local/cuda-12.4 \
   PATH=/usr/local/cuda-12.4/bin:$PATH \
   LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
   uv add --no-build-isolation --no-cache --no-binary-package flash-attn \
     flash-attn@git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3
   ```
   - `TORCH_CUDA_ARCH_LIST`는 사용 중인 GPU 컴퓨트 캡에 맞춰 변경합니다. (RTX 3090 → `8.6`)

## 5. 설치 검증
프로젝트 루트에서 아래 명령으로 Flash Attention 2가 정상 import 되는지 확인합니다. (새 glibc 로더 사용)

```bash
/opt/glibc-2.32/lib/ld-linux-x86-64.so.2 \
  --library-path /opt/glibc-2.32/lib:/usr/local/cuda-12.4/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
  .venv/bin/python -c "import flash_attn; print('flash-attn', flash_attn.__version__)"
```

`flash-attn 2.8.3` 와 같이 버전이 출력되면 성공입니다.

## 6. reranker 실행 방법
- 프로젝트 루트에서 `code/mentoring/run_reranker2.sh` 실행
  ```bash
  cd code/mentoring
  ./run_reranker2.sh
  ```
- 스크립트 내부에서 `uv sync --frozen`으로 환경을 고정한 뒤, glibc 2.32 로더를 이용해 `.venv/bin/python`을 실행합니다. 별도 설정 없이 Flash Attention 2가 활성화된 상태로 추론이 진행됩니다.

## 참고 사항
- Flash Attention 관련 오류가 다시 발생하면 `run_reranker2.sh`의 `--use_flash_attention` 옵션을 제거한 뒤 재실행해 GPU 메모리나 환경 문제를 우선 확인하세요.
- CUDA Toolkit과 glibc는 시스템 전역 변경이므로, 다른 서비스와 충돌이 없는지 점검 후 작업하세요.
