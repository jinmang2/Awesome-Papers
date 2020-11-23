# Improving Language Understanding by Generative Pre-Training

- Alec Radford, OpenAI
- Karthik Narasimhan, OpenAI
- Tim Salimans, OpenAI
- Ilya Sutkever, OpenAI

## Abstract
- NLU은 TE, QA, SSA, DC 등에서 광범위하게 사용됨
- 방대한 unlabeled text 말뭉치들은 풍부하지만 labeled data는 희박함
- GPT는 unlabeled text 말뭉치 셋의 LM을 generative pre-training하고 각 sub-task에 discriminative fine-tuning!
- 이전 연구들과의 차별점: task-aware input transformations
- gpt는 task-agnostic(태스크에 대한 지식 없이)하지만 SOTA 달성!

## 1. Introduction
NLP에서 지도학습에 대한 부담을 덜기 위해 raw text를 효과적으로 학습하는 것이 트렌드. 대부분의 딥러닝 방법은 labeling된 데이터를 많이 필요로 함. 이런 상황에서 labeling되지 않은 데이터에서 언어정보 이점을 가지는 모델은 시간적 계산적으로 훨씬 효율적(labeling을 굳이?!). 게다가 supervision으로 가능한 경우라도 이를 뛰어넘는 성능을 보여줄 수 있음.(Word2Vec같이!)

그러나 unlabeled text에서 단어수준의 정보의 이점은 아래 두 문제를 가짐
1. 어떤 최적화 목적함수가 전이를 위해 사용할만한 text 표현을 학습하기에 최상인지 불분명
2. 위와 같이 학습한 표현을 목적 태스크에 어떻게 효과적으로 전이할지 합의점이 없음

위와 같은 불확실성은 NLP의 효과적인 semi-supervised 접근법 개발을 어렵게 만듦

본 논문에서 비지도 사전 학습과 지도 미세 조정의 조합을 사용하여 자연어 이해를 위한 semi-supervised 기법을 탐색. 최종 목표는 `universal representation`을 학습하는 것! 이에 앞서, unlabeled large corpora와 target task의 annotated training examples가 준비돼있다고 가정(비지도 데이터셋과 지도 데이터셋). target task가 unlabeled corpus와 동일한 domain이 아니어도 가능. fine-tuning based, pre-trained weight를 활용하여 적용.

`Attention Is All You Needs`에서 제안된 Transformer를 모델의 구조로 활용. 이를 활용함으로써 text의 장기의존성 문제를 조절할 수 있음. 