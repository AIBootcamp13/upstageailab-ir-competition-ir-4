

- 과학질문이란? 
	- 우리 경진대회에서는 chit-chat 20개,와 멀티턴 대화20가 명확히 존재 하는것으로  파악됨. (깃이슈참고)
	  
- function_calling 프롬프트로 일반질문/과학질문 구분하기
	- 과학질문인 경우로 하지않고 chit-chat인지 아닌지로 구분하기로 함. 
	- 일반질문이 아닌데 빠트리는 경우가 있어 애를 먹음.
	  
- Qwen-embedding적용
	- 0.6B, 4B, 8B가 있는데 높은 모델일경우 memory Error가 발생하여 모델을 낮추고 batch_size조절등을 하였으나 
	- os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 소스 적용 및 메모리 cleaning적용 후 메모리 문제가 해결됨.
	  
- sparse(bm25)+dense(solar2) = hybrid retrieval 방식 적용
	- 각 retrival topk실험시 40개부근에서 가장 점수가 잘나옴.
		- 100일때0.8894
		- 50일때 0.8955
		- 45일때 0.8970
		- 40일때 0.9061 <= 같은조건 실험 최고
		- 30일때 0.8992 
		  
- 차원에러 (=> 차원축소)
	- elasticsearch와 solar의 차원이 달라 처음에 solar의 차원(4096)을 엘라스틱서치 768에서 잘라서 실험함.
	- 이후 elastic search 버전을 9으로 변경해서 차원축소를 없앰.


- 질문 증강
	-  가상질문(pseudo question) 생성 (3개)

- hard-voting 
	- 결과csv들을 여러개 취합하여 해당 topk 3개를 첫번째 두번째 세번째 순서대로 가중치를 주어 다시 topk 3개를 뽑는다.
	- 1:1:1 의 점수 가중치보다 5:3:1로 가중치를 준것이 점수가 더 좋았다)