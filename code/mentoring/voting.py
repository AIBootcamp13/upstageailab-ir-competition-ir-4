import json
import pandas as pd
import argparse
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import os
import csv

class VotingSystem:
    def __init__(self, model_results: List[Dict], weights: List[float] = None):
        """
        Voting 시스템 초기화
        
        Args:
            model_results: 각 모델의 결과 리스트 [{'name': 'model1', 'data': {...}}, ...]
            weights: 각 모델의 가중치 리스트 (Soft Voting용)
        """
        self.model_results = model_results
        self.num_models = len(model_results)
        
        if weights is None:
            # 기본값: 모든 모델에 동일한 가중치 부여
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            # 가중치 정규화
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        print(f"Voting 시스템 초기화 완료:")
        print(f"- 모델 개수: {self.num_models}")
        print(f"- 모델 이름: {[result['name'] for result in model_results]}")
        print(f"- 가중치: {self.weights}")
    
    def load_voting_data(self, result_dict: Dict) -> Dict:
        """Voting 데이터를 로딩하고 정리"""
        voting_data = {}
        
        for eval_id, items in result_dict.items():
            voting_data[eval_id] = {}
            for item in items:
                item_id = item['item_id']
                voting_data[eval_id][item_id] = {
                    'rank': item['rank'],
                    'cosine_similarity': item['cosine_similarity']
                }
        
        return voting_data
    
    def hard_voting(self, n: int, topk: int) -> List[Dict]:
        """
        Hard Voting 구현
        - 각 모델의 상위 n개 결과에서 공통으로 포함된 아이템들에 대해 순위 기반 투표
        - 순위가 높을수록 높은 점수 부여 (예: 1위=n점, 2위=n-1점, ...)
        """
        print(f"\n=== Hard Voting 수행 (n={n}, topk={topk}) ===")
        
        # 각 모델의 데이터를 로딩
        all_voting_data = []
        for result in self.model_results:
            voting_data = self.load_voting_data(result['data'])
            all_voting_data.append(voting_data)
        
        # 모든 eval_id에 대해 처리
        eval_ids = set()
        for voting_data in all_voting_data:
            eval_ids.update(voting_data.keys())
        
        final_results = []
        
        for eval_id in sorted(eval_ids):
            print(f"처리 중: eval_id {eval_id}")
            
            # 각 모델에서 해당 eval_id의 상위 n개 아이템 수집
            item_votes = defaultdict(int)
            item_info = {}
            
            for model_idx, voting_data in enumerate(all_voting_data):
                if eval_id in voting_data:
                    model_items = voting_data[eval_id]
                    
                    # 상위 n개만 선택 (rank 기준)
                    sorted_items = sorted(model_items.items(), key=lambda x: x[1]['rank'])[:n]
                    
                    for item_id, item_data in sorted_items:
                        # 순위 기반 점수 부여 (1위=n점, 2위=n-1점, ...)
                        rank_score = n - item_data['rank'] + 1
                        item_votes[item_id] += rank_score
                        
                        # 아이템 정보 저장 (첫 번째로 나타나는 모델의 정보 사용)
                        if item_id not in item_info:
                            item_info[item_id] = {
                                'cosine_similarity': item_data['cosine_similarity'],
                                'original_ranks': {}
                            }
                        item_info[item_id]['original_ranks'][self.model_results[model_idx]['name']] = item_data['rank']
            
            # 투표 결과를 기준으로 정렬
            sorted_items = sorted(item_votes.items(), key=lambda x: x[1], reverse=True)
            
            # 상위 topk개 선택
            top_items = sorted_items[:topk]
            
            for rank, (item_id, vote_score) in enumerate(top_items, 1):
                final_results.append({
                    'eval_id': eval_id,
                    'item_id': item_id,
                    'rank': rank,
                    'vote_score': vote_score,
                    'method': 'hard_voting',
                    'original_ranks': item_info[item_id]['original_ranks'],
                    'cosine_similarity': item_info[item_id]['cosine_similarity']
                })
        
        print(f"Hard Voting 완료: {len(final_results)}개 결과 생성")
        return final_results
    
    def soft_voting(self, n: int, topk: int) -> List[Dict]:
        """
        Soft Voting 구현
        - 각 모델의 cosine similarity를 가중치를 적용하여 합산
        - 최종 점수 = Σ(weight_i * similarity_i)
        """
        print(f"\n=== Soft Voting 수행 (n={n}, topk={topk}) ===")
        print(f"가중치: {self.weights}")
        
        # 각 모델의 데이터를 로딩
        all_voting_data = []
        for result in self.model_results:
            voting_data = self.load_voting_data(result['data'])
            all_voting_data.append(voting_data)
        
        # 모든 eval_id에 대해 처리
        eval_ids = set()
        for voting_data in all_voting_data:
            eval_ids.update(voting_data.keys())
        
        final_results = []
        
        for eval_id in sorted(eval_ids):
            print(f"처리 중: eval_id {eval_id}")
            
            # 각 아이템의 가중 평균 점수 계산
            item_scores = defaultdict(float)
            item_info = defaultdict(dict)
            item_count = defaultdict(int)
            
            for model_idx, voting_data in enumerate(all_voting_data):
                if eval_id in voting_data:
                    model_items = voting_data[eval_id]
                    model_weight = self.weights[model_idx]
                    model_name = self.model_results[model_idx]['name']
                    
                    # 상위 n개만 선택 (rank 기준)
                    sorted_items = sorted(model_items.items(), key=lambda x: x[1]['rank'])[:n]
                    
                    for item_id, item_data in sorted_items:
                        # 가중치를 적용한 유사도 점수 합산
                        weighted_score = item_data['cosine_similarity'] * model_weight
                        item_scores[item_id] += weighted_score
                        item_count[item_id] += 1
                        
                        # 아이템 정보 저장
                        if 'original_ranks' not in item_info[item_id]:
                            item_info[item_id]['original_ranks'] = {}
                            item_info[item_id]['individual_similarities'] = {}
                        
                        item_info[item_id]['original_ranks'][model_name] = item_data['rank']
                        item_info[item_id]['individual_similarities'][model_name] = item_data['cosine_similarity']
            
            # 점수를 기준으로 정렬
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 상위 topk개 선택
            top_items = sorted_items[:topk]
            
            for rank, (item_id, final_score) in enumerate(top_items, 1):
                final_results.append({
                    'eval_id': eval_id,
                    'item_id': item_id,
                    'rank': rank,
                    'weighted_similarity': final_score,
                    'method': 'soft_voting',
                    'original_ranks': item_info[item_id]['original_ranks'],
                    'individual_similarities': item_info[item_id]['individual_similarities'],
                    'model_count': item_count[item_id]  # 몇 개 모델에서 선택되었는지
                })
        
        print(f"Soft Voting 완료: {len(final_results)}개 결과 생성")
        return final_results

def load_model_results(file_paths: List[str], model_names: List[str] = None) -> List[Dict]:
    """
    여러 모델의 결과 파일을 로딩
    
    Args:
        file_paths: 모델 결과 파일 경로들
        model_names: 모델 이름들 (없으면 파일명 사용)
    
    Returns:
        모델 결과 리스트
    """
    model_results = []
    
    for i, file_path in enumerate(file_paths):
        if not os.path.exists(file_path):
            print(f"경고: 파일이 존재하지 않습니다: {file_path}")
            continue
        
        model_name = model_names[i] if model_names and i < len(model_names) else f"model_{i+1}"
        
        print(f"로딩 중: {model_name} ({file_path})")
        
        # 결과 데이터 로딩
        data = defaultdict(list)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line.strip())
                eval_id = result['eval_id']
                data[eval_id].append(result)
        
        model_results.append({
            'name': model_name,
            'data': dict(data),
            'file_path': file_path
        })
        
        print(f"  - {len(data)}개 쿼리, 총 {sum(len(items) for items in data.values())}개 아이템")
    
    return model_results

def save_results(results: List[Dict], output_path: str, method: str):
    """결과를 JSONL 형식으로 저장"""
    print(f"\n{method} 결과를 {output_path}에 저장 중...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f'{json.dumps(result, ensure_ascii=False)}\n')
    
    print(f"저장 완료: {len(results)}개 결과")

def save_results_as_csv(results: List[Dict], output_path: str, method: str):
    """결과를 CSV 형식으로 저장 (JSONL 내용을 CSV에 저장)"""
    print(f"\n{method} 결과를 {output_path}에 저장 중...")
    
    # reranked_submission_final.csv 형식에 맞게 변환
    csv_results = []
    for result in results:
        # eval_id별로 topk 리스트 생성
        eval_id = result['eval_id']
        item_id = result['item_id']
        
        # 같은 eval_id의 기존 결과 찾기
        existing = None
        for csv_result in csv_results:
            if csv_result['eval_id'] == eval_id:
                existing = csv_result
                break
        
        if existing is None:
            # 새로운 eval_id인 경우
            csv_result = {
                'eval_id': eval_id,
                'standalone_query': '',  # 이 정보는 원본 데이터에서 가져와야 함
                'topk': [item_id],
                'answer': '',
                'references': []
            }
            csv_results.append(csv_result)
        else:
            # 기존 eval_id에 item_id 추가
            if len(existing['topk']) < 3:  # topk는 최대 3개
                existing['topk'].append(item_id)
    
    # CSV 파일로 저장 (실제로는 JSONL 형식이지만 .csv 확장자)
    with open(output_path, 'w', encoding='utf-8') as f:
        for csv_result in csv_results:
            f.write(f'{json.dumps(csv_result, ensure_ascii=False)}\n')
    
    print(f"저장 완료: {len(csv_results)}개 결과")

def convert_model_to_csv_format(model_result: Dict, output_path: str, model_name: str):
    """개별 모델 결과를 CSV 형식으로 변환하여 저장"""
    print(f"\n{model_name} 모델 결과를 {output_path}에 저장 중...")
    
    # eval_id별로 그룹화
    grouped_results = defaultdict(list)
    for result in model_result['data'].values():
        for item in result:
            grouped_results[item['eval_id']].append(item)
    
    csv_results = []
    for eval_id, items in grouped_results.items():
        # rank 순으로 정렬하여 상위 3개 선택
        sorted_items = sorted(items, key=lambda x: x['rank'])[:3]
        topk_items = [item['item_id'] for item in sorted_items]
        
        csv_result = {
            'eval_id': eval_id,
            'standalone_query': '',  # 이 정보는 원본 데이터에서 가져와야 함
            'topk': topk_items,
            'answer': '',
            'references': []
        }
        csv_results.append(csv_result)
    
    # CSV 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for csv_result in csv_results:
            f.write(f'{json.dumps(csv_result, ensure_ascii=False)}\n')
    
    print(f"저장 완료: {len(csv_results)}개 결과")

def main():
    parser = argparse.ArgumentParser(description='다중 모델 Voting 시스템')
    parser.add_argument('--model_files', nargs='+', required=True, 
                       help='모델 결과 파일들 (예: reranker_voting.jsonl biencoder_voting.jsonl)')
    parser.add_argument('--model_names', nargs='+', default=None,
                       help='모델 이름들 (기본값: 파일명 기반)')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                       help='Soft Voting용 모델 가중치 (기본값: 균등 가중치)')
    parser.add_argument('--n', type=int, default=50,
                       help='각 모델에서 상위 n개 결과 사용 (기본값: 50)')
    parser.add_argument('--topk', type=int, default=3,
                       help='최종 상위 k개 결과 선택 (기본값: 3)')
    parser.add_argument('--output_dir', type=str, default='./voting_results',
                       help='결과 저장 디렉토리 (기본값: ./voting_results)')
    parser.add_argument('--method', type=str, choices=['hard', 'soft', 'both'], default='both',
                       help='Voting 방법 선택 (기본값: both)')
    parser.add_argument('--save_individual_models', action='store_true',
                       help='개별 모델 결과도 함께 저장 (기본값: False)')
    
    args = parser.parse_args()
    
    print("=== 다중 모델 Voting 시스템 ===")
    print(f"입력 파일들: {args.model_files}")
    print(f"모델 이름들: {args.model_names}")
    print(f"가중치: {args.weights}")
    print(f"n={args.n}, topk={args.topk}")
    print(f"방법: {args.method}")
    print(f"개별 모델 저장: {args.save_individual_models}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 결과 로딩
    model_results = load_model_results(args.model_files, args.model_names)
    
    if len(model_results) == 0:
        print("오류: 로딩된 모델 결과가 없습니다.")
        return
    
    # Voting 시스템 초기화
    voting_system = VotingSystem(model_results, args.weights)
    
    # Hard Voting 수행
    if args.method in ['hard', 'both']:
        hard_results = voting_system.hard_voting(args.n, args.topk)
        hard_output_path = os.path.join(args.output_dir, 'hard_voting_results.jsonl')
        hard_csv_path = os.path.join(args.output_dir, 'hard_voting_results.csv')
        
        save_results(hard_results, hard_output_path, 'Hard Voting')
        save_results_as_csv(hard_results, hard_csv_path, 'Hard Voting CSV')
        
        # 통계 출력
        print(f"\n=== Hard Voting 통계 ===")
        eval_ids = set(r['eval_id'] for r in hard_results)
        print(f"처리된 쿼리 수: {len(eval_ids)}")
        avg_vote_score = np.mean([r['vote_score'] for r in hard_results])
        print(f"평균 투표 점수: {avg_vote_score:.2f}")
    
    # Soft Voting 수행
    if args.method in ['soft', 'both']:
        soft_results = voting_system.soft_voting(args.n, args.topk)
        soft_output_path = os.path.join(args.output_dir, 'soft_voting_results.jsonl')
        soft_csv_path = os.path.join(args.output_dir, 'soft_voting_results.csv')
        
        save_results(soft_results, soft_output_path, 'Soft Voting')
        save_results_as_csv(soft_results, soft_csv_path, 'Soft Voting CSV')
        
        # 통계 출력
        print(f"\n=== Soft Voting 통계 ===")
        eval_ids = set(r['eval_id'] for r in soft_results)
        print(f"처리된 쿼리 수: {len(eval_ids)}")
        avg_similarity = np.mean([r['weighted_similarity'] for r in soft_results])
        print(f"평균 가중 유사도: {avg_similarity:.4f}")
        
        # 모델별 기여도 분석
        model_contributions = defaultdict(int)
        for result in soft_results:
            for model_name in result['original_ranks'].keys():
                model_contributions[model_name] += 1
        
        print(f"모델별 기여도 (상위 {args.topk}개 결과에 포함된 횟수):")
        for model_name, count in model_contributions.items():
            percentage = (count / len(soft_results)) * 100
            print(f"  - {model_name}: {count}회 ({percentage:.1f}%)")
    
    # 개별 모델 결과 저장
    if args.save_individual_models:
        print(f"\n=== 개별 모델 결과 저장 ===")
        for model_result in model_results:
            model_name = model_result['name']
            model_csv_path = os.path.join(args.output_dir, f'{model_name}_results.csv')
            convert_model_to_csv_format(model_result, model_csv_path, model_name)
    
    print(f"\n=== Voting 완료 ===")
    print(f"결과가 {args.output_dir}에 저장되었습니다.")


if __name__ == "__main__":
    main()
