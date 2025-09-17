#!/usr/bin/env python3
"""
LLM 캐시 관리 도구

특정 LLM 캐시 키를 삭제하거나 전체 캐시를 정리할 수 있습니다.
"""

import os
import sys
import argparse
from typing import Any, Dict

# 부모 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from utils.llm_cache import _cache_path, load_cache, save_cache_atomic


def list_cache_keys(cache_dir: str, provider: str, model: str) -> None:
    """캐시 파일의 모든 키를 출력"""
    path = _cache_path(cache_dir, provider, model)

    if not os.path.exists(path):
        print(f"[INFO] 캐시 파일이 존재하지 않습니다: {path}")
        return

    data = load_cache(path)
    if not data:
        print(f"[INFO] 캐시 파일이 비어있습니다: {path}")
        return

    print(f"[INFO] 캐시 파일: {path}")
    print(f"[INFO] 총 {len(data)}개의 캐시 항목:")

    for i, (key, entry) in enumerate(data.items(), 1):
        created_at = ""
        if isinstance(entry, dict) and "created_at" in entry:
            created_at = f" (생성시간: {entry['created_at']})"
        print(f"  {i:3d}. {key[:20]}...{created_at}")


def delete_cache_key(cache_dir: str, provider: str, model: str, key: str) -> bool:
    """특정 캐시 키를 삭제"""
    path = _cache_path(cache_dir, provider, model)

    if not os.path.exists(path):
        print(f"[ERROR] 캐시 파일이 존재하지 않습니다: {path}")
        return False

    data = load_cache(path)
    if not data:
        print(f"[ERROR] 캐시 파일이 비어있습니다: {path}")
        return False

    if key not in data:
        print(f"[ERROR] 키 '{key}'를 찾을 수 없습니다.")
        return False

    # 키 삭제
    del data[key]

    # 파일 저장
    try:
        save_cache_atomic(path, data)
        print(f"[OK] 키 '{key}'가 삭제되었습니다.")
        print(f"[INFO] 남은 캐시 항목 수: {len(data)}")
        return True
    except Exception as e:
        print(f"[ERROR] 캐시 저장 실패: {e}")
        return False


def clear_all_cache(cache_dir: str, provider: str, model: str) -> bool:
    """전체 캐시 파일 삭제"""
    path = _cache_path(cache_dir, provider, model)

    if not os.path.exists(path):
        print(f"[INFO] 캐시 파일이 존재하지 않습니다: {path}")
        return True

    try:
        os.remove(path)
        print(f"[OK] 캐시 파일이 삭제되었습니다: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] 캐시 파일 삭제 실패: {e}")
        return False


def find_key_by_prefix(cache_dir: str, provider: str, model: str, key_prefix: str) -> str:
    """키 접두사로 전체 키를 찾기"""
    path = _cache_path(cache_dir, provider, model)

    if not os.path.exists(path):
        return ""

    data = load_cache(path)
    if not data:
        return ""

    matches = [k for k in data.keys() if k.startswith(key_prefix)]

    if len(matches) == 0:
        print(f"[ERROR] '{key_prefix}'로 시작하는 키를 찾을 수 없습니다.")
        return ""
    elif len(matches) == 1:
        full_key = matches[0]
        print(f"[INFO] 매칭된 키: {full_key}")
        return full_key
    else:
        print(f"[ERROR] '{key_prefix}'로 시작하는 키가 {len(matches)}개 있습니다:")
        for i, key in enumerate(matches, 1):
            print(f"  {i}. {key}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="LLM 캐시 관리 도구")
    parser.add_argument("--provider", required=True, help="LLM provider (예: gemini, openai)")
    parser.add_argument("--model", required=True, help="모델명 (예: gemini-2.5-flash)")
    parser.add_argument("--cache-dir", default="cache/llm", help="캐시 디렉토리 경로")
    parser.add_argument("--yes", "-y", action="store_true", help="확인 프롬프트 없이 실행")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--key", help="삭제할 캐시 키 (전체 키 또는 접두사)")
    group.add_argument("--clear-all", action="store_true", help="전체 캐시 삭제")
    group.add_argument("--list", action="store_true", help="캐시 키 목록 출력")

    args = parser.parse_args()

    # .env 파일 로드
    load_dotenv()

    print(f"[INFO] Provider: {args.provider}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Cache directory: {args.cache_dir}")

    if args.list:
        list_cache_keys(args.cache_dir, args.provider, args.model)
        return 0

    elif args.clear_all:
        print("[WARNING] 전체 캐시를 삭제합니다.")
        if args.yes:
            confirm = 'y'
        else:
            confirm = input("계속하시겠습니까? (y/N): ")
        if confirm.lower() in ('y', 'yes'):
            success = clear_all_cache(args.cache_dir, args.provider, args.model)
            return 0 if success else 1
        else:
            print("[INFO] 취소되었습니다.")
            return 0

    elif args.key:
        key_to_delete = args.key

        # 키가 짧으면 접두사로 간주하여 전체 키 찾기
        if len(key_to_delete) < 20:
            full_key = find_key_by_prefix(args.cache_dir, args.provider, args.model, key_to_delete)
            if not full_key:
                return 1
            key_to_delete = full_key

        print(f"[WARNING] 키를 삭제합니다: {key_to_delete}")
        if args.yes:
            confirm = 'y'
        else:
            confirm = input("계속하시겠습니까? (y/N): ")
        if confirm.lower() in ('y', 'yes'):
            success = delete_cache_key(args.cache_dir, args.provider, args.model, key_to_delete)
            return 0 if success else 1
        else:
            print("[INFO] 취소되었습니다.")
            return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())