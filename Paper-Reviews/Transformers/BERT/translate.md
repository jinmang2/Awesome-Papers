# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- Google AI Language
  - Jacob Devlin
  - Ming-Wei Chang
  - Kenton Lee
  - Kristina Toutanova

## Abstract

#### BERT를 소개합니다! 뭐 별거 수정안해도 SOTA랍니다~
BERT: Bidirectional Encoder Representations from Transformers 소개
AllenAI의 ELMo, OpenAI의 GPT와 다르게 BERT는 모든 layer에서 left-right context를 동시에 학습할 수 있는 deep bidirectional representations를 unlabeled text에 대해 pre-training하도록 설계됨
pre-trained BERT 모데은 QA, language inference 등에서 task 특화된 구조로 수정할 필요없이 광범위한 nlp task에서 고작 output layer 하나를 추가하여 fine-tuning한 것만으로 SOTA를 달성

#### BERT는 개쩜
BERT의 concept은 간단하고 실험적으로 강력함.
다양한 task에서 괄목할 성능 개선을 달성 및 새로운 SOTA를 달성!


## 1. Introduction
#### Pre-training LM은 NLP task에서 매우 뛰어났음!
Language Model pre-training은 자연어 처리 task에서 매우 효과적이었음
- Semi-supervised sequence learning, Dai and Le, 2015
- Deep contextualized word representations, Peters et al., 2018a
- Improving language understanding with unsupervised learning, Radford et al, 2018
- Universal language model fine-tuning for text classification, Howard and Ruder, 2018

token-level task에서도 뛰어났고
- token-level task란, model이 output을 token 단위로 fine-grained(잘게 쪼개진)하게 생산할 것을 요구되는 task
- named entity recognition과 question answering 등이 존재
- introduction to th conll-2003 shared task: language-independent named entity recognition, Tjong Kim Sang and De Meulder, 2003
- SQuad: 100,000+ questions for machine comprehension of text, Rajpurkar et al., 2016

sentence-level task에서도 우수한 성능을 보임
- natural language inference
  - A large annotated corpus for learning natural language inference, Bowman et al., 2015 (EMNLP)
  - A board-coverage challenge corpus for sentence understanding through inference, Williams et al., 2018
- Paraphrasing
  - Automatically constructing a corpus of sentential paraphrases, Dolan and Brockett, 2005

Note that: coarse-grained, fine-grained
- Vision algorithm에서 자주 사용되는 개념으로 coarse-to-fine strategy가 있음
- 간략화된 이미지(lowered-resolution)에서 세밀화된 이미지(the finest resolution)으로 변화시키며 KLT Tracker 등 optimzation 기반의 알고리즘의 convergence basin을 넓혀주는 역할을 수행
- fine and coarse는 표면 작업 등을 할 때 얼마나 `곱게` 혹은 `듬성듬성 우둘투둘`하게 하느냐를 말할 때 쓰는 단어
- SW Engineering에서는 문제를 `성글게(coarse, chunk)` 혹은 `세밀하고 작은 단위로(fine)` 나눌 것인지에 대한 얘기
- coarse-grained programming은 프로그램을 큰 덩어리로 나누는 것
- fine-grained programming은 더 세밀한 요소로 프로그래밍을 나누어서 고려

#### Pre-training을 downstream-task에 적용시킬 두 가지 전략!
_feature-based_ vs _fine-tuning_

ELMo는 _feature-based_ 접근 방법
- pre-trained representations를 feature로 추가하는 task-specific architectures를 사용

GPT(Generative Pre-trained Transformer)는 _fine-tuning_ 접근 방법
- minimal task-specific parameters를 소개
- pre-trained parameters를 간단히 fine-tuning하여 downstream task에 적용

이 구 접근 방법은 pre-training동안에는 objective function을 공유함
그런데 이 objective function은 `unidirectional LM`을 사용했다는 것!

#### ELMo, GPT? 문제있어 너희들~ 왜 unidirectional LM 모델을 사용하는거야~
위 `unidirectional LM`이 문제! fine-tuning에 사용되는 pre-trained representation의 표현력(power)을 제한함
OpenAI GPT에서보면, 저자는 `left-to-right` 구조를 사용
이 구조는 Transformer의 self-attention layer에서 이전 token에만 집중할 수밖에 없음
이러한 제약은 양방향으로 문맥을 파악해야하는 token 단위의 QA task같은 경우에 fine-tuning을 적용시 매우 불리하게 작용할 수 있고 sentence 단위의 task에서 sub-optimal(underfitting?과 유사?)함

#### BERT의 B? Bidirectional!
본 논문에서는 BERT를 제안, fine-tuning 접근 방식을 개선하고자 함

BERT는 위에서 언급한 unidirectional 한 제약을 `masked language model(MLM)` pre-training objective를 사용하여 문제를 해결하고자 함!
- MLM pre-training objective는 Cloze task(Cloze procedure: A new tool for measuring readability, Taylor, 1953)에서 영감을 얻었다고 함!


masked langauge moddel은 input token의 일부분을 random하게 masking!
그리고 objective는 masking된 token id를 오직 문맥으로 original vocab id을 예측하는 것이 목표!

`left-to-right` LM pre-training, 그러니까 GPT랑 다.르.게! MLM objective는 deep-bidirectional transformer를 pre-training할 수 있게 left와 right context를 연결해주는(fuse) 표현을 가능케해줌!
- 응..? 근데 뭐가 다른거지..? GPT를 읽어야 하나...
- fuse한다는게 bi-direct하게 연결해준 다는 표현으로 이해해야 하나?
- 아 보니까 left context와 right context를 둘 다 고려한다는 말이었네
- `left-to-right`와 `right-to-left`를 간결하게 쓴 말!

MLM에 더하여 `next sentence prediction` task를 사용!
- text-pair 표현을 동시에 pre-training하는 task

#### 본 논문의 contribution
1. Bidirectional Pre-training (fine-tuning based)
- GPT와 다르게 Language representation에 있어 `bidirectional`한 pre-training 방식의 중요성을 부각시킴
- ELMo와도 다름! ELMo는 `left-to-right`와 `right-to-left` LM을 각각 독립적으로 학습한 후에 shallow concat을 사용한다구!

2. NLP task에 specific하지 않고 universal하게 적용 가능! SOTA신기록 까지!
- BERT는 sentence-level과 token-level task에서 SOTA를 당성
- 많은 task-specific한 구조에서도 출중!
- 이는 task-specific하게 구조를 완전 수정하지 않고도 달성!!(맞나:?)

3. 11개의 NLP task에 대해 SOTA를 개선
- 코드는 https://github.com/google-research/bert 에서 확인 가능

## 2. Related Work

### 2.1 Unsupervised Feature-based Approaches
수십년간 광범위하게 적용이 가능한 단어 표현을 학습하는 일은 굉장히 많은 연구가 진행됐음
- non-neural case
  - Class-based n-gram models of natural language, Brown et al., 1992
  - A framework for learning predictive structures from multiple tasks and unlabeld data, Ando and Zhang, 2005
  - Domain adaptation with structual correspondence learning, Blitzer et al., 2006
- neural case
  - Distributed representations of words and phrases and their compositionality, Mikolov et al., 2013
  - Glove: Global vectors for word representation, Pennington et al., 2014

Pre-trained word embedding은 현대 NLP system에서 중요
- Word representations: A simple and general method for semi-supervised learning, Turian et al., 2010
  - Bengio 교수도 참가했네... ㄷㄷ

word embedding vector를 pre-train시키기 위해 `left-to-right` LM objective가 사용되옴
- A scalable hierarchical distributed language model, Mnih and Hinton, 2009

혹은 좌우 문맥에서 틀린 단어와 정확하게 구별하는 objective를 사용
- Distributed representations of words and phrases and their compositionality, Mikolov et al., 2013 (Word2Vec)

위와 같은 접근 방식은 `coarser granularities`로 일반화 가능
- 큰 범위로 세분화, 예를 들어
- Sentence embeddings
  - Skip-thought vectors, Kiros et al., 2015
  - An efficient framework for learning sentence representations, Logeswaran and Lee, 2018
- paragraph embeddings
  - Distributed representations of sentences and documents, Le and Mikolov, 2014

sentence representation을 학습시키기 위해 이전 연구는 다음 문잘의 후보의 rank를 objective로 사용하기도 함
- Discourse-based objectives for fast unsupervised sentence representation learning, Jernite et al., 2017
- An efficient framework for learning sentence representations, Logeswaran and Lee, 2018

주어진 이전 문장에 대한 표현으로 다음 문장 단어를 left-to-right로 생성하는 objective도 사용
- Skip-thought vectors, Kiros et al., 2015

혹은 denoising autoencoder에서 파생된 objective도 사용
- Learning distributed representations of sentences from unlabelled data, Hill et al., 2016
  - 조경현 교수님 여기도 계시네... ㄷㄷ

ELMo와 peter의 이전 논문은 기존 word embedding 연구를 다른 차원으로 일반화 시킴
- Semi-supervised sequence tagging with bidirectional language models, Peters e al., 2017
- Deep contextualied word representations, Peters et al., 2018a

biLM(left-to-right and right-to-left)로 `context-sensitive` feature를 추출
각 토큰의 contextual 표현은 독립적인 양 방향을 concat한 것
ELMo도 대단함... task-specific 구조에 ELMo contextual word embedding을 결합하여 다양한 NLP task에서 SOTA를 달성함

Melamud는 아래 논문에서 LSTM을 사용하여 left context, right context 둘 다에서 단일 단어를 예측하는 task를 통해 contextual representation을 학습하는 방법을 제안
- context2vec: Learning generic context embedding with bidirectional LSTM, Melamud et al., 2016

ELMo와 유사하게 `context2vec`도 feature-based 모델이며 deeply bidirectional이 아님

Fedus는 Maskgan에서 cloze task가 text 생성 모델에서 robustness를 개선하는데 사용될 수 있음을 보임
- Maskgan: Better text generation ia fillign in the ..., Fedus et al., 2018

### 2.2 Unsupervised Fine-tuning Approaches
feature-based 접근방식과 마찬가지로, 첫 번째 방법은 레이블이 없는 텍스트에서 매개 변수를 포함시키는 사전 훈련된 단어만 이 방향으로 작동
- Deep neural networks with multitask learning, Collobert and Weston, 2008

최근엔 contextual token 표현을 생산하는 sentence/document encoders가 unlabeled text에서 pre-trained되고 supervised downstream task에서 fine-tune시키는 연구가 다수 진행됐다.
- Semi-supervised sequence learning, Dai and Le, 2015
- Universal language model fine-tuning for text classification, Howard and Ruder, 2018
- Improving language understanding with unsupervised learning, Radford et al, 2018 (OpenAI GPT)

이러한 접근방법의 장점은 처음부터 학습할 parameter가 적은 것
이러한 장점 때문인지는 몰라도 OpenAI의 GPT는 GLUE benchmark에서 많은 문장 단위 task에서 SOTA를 갱신!
- GLUE benchmark
  - Glue: A multi-task benchmark and analysis platform for natural language understanind, Wang et al., 2018a

left-to-right와 auto-encoder objective가 아래 모델들의 pre-training에 사용돼옴
- Universal language model fine-tuning for text classification, Howard and Ruder, 2018
- Improving language understanding with unsupervised learning, Radford et al, 2018 (OpenAI GPT)
- Semi-supervised sequence learning, Dai and Le, 2015
- 위 세 논문들 겁나 언급하네...

### 2.3 Transfer Learning from Supervised Data
또 아래와 같이 natural language inference, machine translation 태스크에서 많은 데이터셋의 supervised task에 효과적으로 transfer learning을 적용한 사례가 있다.
- natural language inference
  - Supervised learning of universal sentence representations from natural language inference data, Conneau et al., 2017
- machine translation
  - Learning generic context embedding with bidirectional, McCann et al., 2017 (CoVE?)

Computer Vision 연구는 이미 transfer learning의 중요성을 강조했었다.
- ImageNet
  - ImageNet: A Large-Scale Hierarchical Image Database, Deng et al., 2009
  - How transferable are features in deep neural networks?, Tosinski et al., 2014

## 3. BERT
이제 드디어 BERT와 구현 세부사항을 소개하자
pre-training과 fine-tuning 두 단계가 BERT framework에 존재한다
pre-training 도중 model은 다른 pre-training task의 unlabeled data에서 학습된다.
fine-tuning에서 BERT 모델은 pre-trained parameter로 초기화되며 모든 parameter들은 downstream task의 labeled data를 사용하여 fine-tune된다.
각 downstream task는 같은 pre-trained parameter로 초기화되며 분리된 fine-tuned model은 가진다.

BERT의 독특한(distinctive) 특성은 각 다른 task의 구조를 단일화한다는 것이다. 이는 pre-trained 구조와 최종 downstream 구조 사이엔 아주 조금의 차이만이 존재한다.

#### Model Architecture
BERT 모델은 `Attention Is All You Needs`에서 구현된 Transformer의 Encoder 부분을 활용한 multi-layer bidirectional Transformer 구조로 돼있다.
이를 `tensor2tensor` 라이브러리로 제공 중이다 ^^
Transformer의 구조를 이용하기 때문에 BERT 구현체는 transformer와 거의 같다! 때문에 상세한 설명은 `Attention Is All You Needs`를 읽어라 본 논문에서는 생략하겠다.
http://nlp.seas.harvard.edu/2018/04/03/attention.html 여기서 설명이 잘 나와 있다규

**Notation**
본 연구에서 Layer(i.e., Transformer Block)의 수를 $L$로, hidden size를 $H$, self-attention heads의 수를 $A$로 표기하도록 한다.
  - 모든 경우에서 feed-forward/filter size를 4H로 설정했다.
  - 즉, $H=768$인 경우엔 $3072$, $H=1024$인 경우엔 $4096$이 된다.

본 연구에서 두 모델을 제시한다;
- $\text{BERT}_{\text{BASE}}$ (L=12, H=768, A=12, Total Parameters=110M)
- $\text{BERT}_{\text{LARGE}}$ (L=24, H=1024, A=16, Total Parameters=340M)

$\text{BERT}_{\text{BASE}}$는 OpenAI GPT와의 비교를 위해 같은 모델 size를 가지도록 setting함. 그.러.나! BERT Transformer는 `bidirectional self-attention`을 사용하고 GPT Transformer는 몯느 token이 왼쪽의 context에만 집중할 수 밖에 없는 제한된 self-attention을 사용한다!!
- GPT는 `Transformer decoder`로 언급된다네요? left-context-only versino이래요
- 왜냐? text 생성을 위해 사용되므로!
- BERT는 `Transformer encoder`를 사용, bidirectional하게!!

#### Input/Output Representations
BERT가 다양한 다운스트림 작업을 처리하도록 하기 위해, 우리의 입력 표현은 하나의 토큰 시퀀스로 한 문장과 한 쌍의 문장 모두를 명확하게 나타낼 수 있다.
위 작업을 하는 동안 "Sentence"는 언어학적인 실제 문장이라기보다 연속된 text의 임의의 span이 될 수 있다. (오... 수학적 표현인데?)
"Sequence"는 BERT의 입력 token sequence를 의미하며 이는 단일 문장 혹은 두 문장을 묶은 입력일 수 있다.

본 연구를 위해 30,000 token vocabularry에 `WordPiece embeddings`을 수행했다.
- Google's neural machine translation system: Bridging the gap between human and machine translation, Wu et al., 2016

모든 sequence의 첫 token은 항상 special classification token ([CLS])으로 준비한다. 이 token의 최종 hidden state는 분류 문제에서 sequence 표현을 묶는데(aggregate) 사용된다. sentence pair는 single seqeuence로 pack됨(구현 코드에서 sentence ix 0, 1 넣는 부분!).

자, 우리는 문장을 두 방법으로 구분 짓는다.
첫 째, special token ([SEP])을 넣어 구분짓는다.
둘 째, 모든 token이 sentence A에 포함돼있는지 B에 포함돼있는지 학습된 embedding을 추가한다.
input embedding을 $E$로 표기, special [CLS] token의 final hidden vector를 $C\in\mathbb{R}^H$로, $i^{th}$번째 input token의 최종 hidden vector를 $T_i\in\mathbb{R}^H$로 표기한다.

주어진 token에 대해 input 표현은 token, segment, position embedding을 더하여 구성된다.

+++ 추가
- bert v1판에서 보면 `We use learned positional embeddings with supported sequence lengths up to 512 tokens` 라는 표현이 등장
- 보니까 Appendix로 뺌...

### 3.1 Pre-training BERT
다시금 말하지만 ELMo와 GPT와는 다르다!! 단순하게 left-to-right, right-to-left로 학습시키지 않는단 말이다!
대신, 두 unsupervised task를 사용하여 pre-training하는데 이를 설명코자 한다.

#### Task #1: Masked LM
직관적으로, `deep bidirectional model`이 단순 `left-to-right` 모델이나 `left-to-right`와 `right-to-left`를 얕게 cnocat한 모델보다 훨씬 강력하다는 것은 자명하다.
불행하게도, `bidirectional conditioning`은 간접적으로 각 단어가 "자신을 보는 것"을 가능케하고 이에 따라 모델이 너무 자명하게 multi-layered context하에서 타켓 단어를 예측할 수 있기 때문에 기존 `conditional LM`은 `left-to-right` 또는 `right-to-left`로만 훈련될 수 있었다.

deep bidirectional representation을 학습하기 위해 본 연구에선 input token을 무작위로 일부 `masking`하고 그 `masking`된 token을 예측하도록 했다. 이를 우리는 `masked LM(MLM)`이라 부르고 이는 앞서 설명한 바와 같이 1953 Cloze procedure에서 영감을 받았다. 이 경우, mask token의 최종 hidden vector는 표준 LM과 같이 vocab을 통해 output softmax로 feeding된다. 본 모든 실험 과정에서 무작위로 각 sequence의 `WordPiece token`들의 15%를 masking했다. `denoising auto-encoders`와는 다르게 전체 input을 재구축하는 것이 아닌 masking된 단어를 예측한다.
- Extracting and composing robust features with denoising autoencoders, Vincent et al., 2008
  - Bengio 교수!

이로 인해 bidirectional하게 모델을 pre-training시키는 것이 가능해졌지만, fine-tuning에선 [MASK] token이 등장하지 않기 때문에 pre-training과 fine-tuning 사이에서 downside에 불균형(mismatch)이 생긴다. 이를 완화(mitigate)시키기 위해 'masked'된 토큰을 실제 [MASK] 토큰으로 항상 대체시키진 않는다. 학습 데이터 생성자는 예측을 위해 무작위로 15%의 token position을 고른다. 만일 $i-th$ 토큰이 골라졌다고 가정하자. 그러면 BERT는 $i-th$ 토큰을 (1) 80%의 확률로 [MASK] 토큰으로, (2) 10%의 확률로 임의의 토큰으로 변화시키고 (3) 10%의 확률로 그대로 둔다. 그러면 cross entropy loss로 original token를 예측하기 위해 $T_i$이 사용될 것이다.

#### Task #2: Next Sentence Prediction (NSP)
QA와 NLI같은 NLP에서 중요한 downstream task들은 두 문장 사이의 `relationship`을 이해하는 것에 기반을 둔다. 이 두 문장 사이의 "관계"는 보통 LM으로는 직관적으로 포착되지 않는다. 모델이 문장 사이의 관계를 이해할 수 있도록 단일 언어 코퍼스에서 생성된 `binarized next sentence prediction task`를 pre-train시킨다. 특별한 경우로 각 사전학습 예제에서 sentence A와 sentence B를 뽑았다고 가정하자. 각 step별 50%의 확률로 B는 A의 실제 다음 문장(이를 $\text{IsNext}$라고 라벨링), 그리고 50%의 확률로 B를 corpus의 임의의 문장으로 추출된다(이를 $\text{NotNext}$라고 라벨링). $C$는 다음 sentence prediction을 위해 사용된다(최종 모델은 NSP에서 97-98%의 정확도를 달성했다). 이렇게나 단순함에도 불구하고 QA와 NLI 모두에서 너무나 좋은 효과를 보여줬다!!
- $C$는 NSP에서 학습됐기 때문에 fine-tuning이 없으면 의미있는 문장 표현을 내포하지 못한다.

NSP(Next sentence Prediction) task는 아래 두 논문에서 사용된 표현 학습 objective과 긴밀한 연관성이 있다.
- Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning, Jernite et al., 2017    
- An efficient framework for learning sentence representations, Logeswaran and Lee, 2018

그러나 선행 연구에서는 오직 문장 임베딩만이 down-stream task로 전달되는데 반해 BERT는 end-task model의 모수를 초기화하기 위해 모든 parameter를 전이(transfer)한다.

#### Pre-training data
사전 학습 과정은 현존하는 LM pre-training 자료들을 따른다. pre-training corpus로 BooksCorpus(800M words)와 English Wikipedia(2,500M words)를 사용했다. Wikipedia의 경우 오직 텍스트 구절만 추출하고 리스트, 표, 머리말(header)는 배제(ignore)했다. 또 wikipedia같은 문서 단위의 말뭉치를 사용하는 것이 길고 연속적인 sequence를 추출하기 위한 Billion Word Benchmark와 같은 셔플된 문장 단위의 말뭉치를 사용하는 것보다 더 중요하게 작용했다.

### 3.2 Fine-tuning BERT
Transformer의 self-attention mechanism은 BERT가 적절하게 input과 output을 바꿔(swap) 단일 텍스트 혹은 쌍으로 된 텍스트을 가지는 많은 downstream task를 모델링할 수 있게 만들기 때문에 Fine-tuning 과정은 간단(straightforward)하다. 쌍으로 된 text의 경우, 가장 흔한 패턴은 bidirectional cross attention을 적용하기 전에 텍스트 쌍을 독립적으로 부호화하는 것이다.
- A decomposable attention model for natural language inference, Parikh et al., 2016
- Bidirectional attention flow for machine comprehension, Seo et al., 2017

즉, text-pair를 encode하는 과정과 bidirectional cross attention을 적용하는 두 과정이 존재한다. BERT는 이 두 단계를 통합하기 위해 self-attention mechanism을 사용한다. concat된 text-pair를 self-attention으로 부호화하는 것은 두 문장 간의 bidirectional cross attention을 포함하기 때문이다.

각 task에서 단순하게 BERT에 task 특화 input과 output을 넣어주고 모든 parameter를 end-to-end로 fine-tuning했다. pre-training의 모든 input sentence A와 sentence B는 (1) paraphrasing의 sentence pair, (2) `hypothesis-premise pairs in entailment`, (3) QA의 question-passage 쌍, (4) text 분류 혹은 sequence tagging의 degenerate text-$\empty$ 쌍과 유사하다. Output에서 token representation은 sequence tagging 혹은 question answering과 같은 token 단위 task의 output layer로 feeding되며 [CLS] 표현은 entailment나 sentiment analysis와 같은 분류 output layer로 feeding된다.

사전 학습과 비교하여 fine-tuning은 비교적 cost가 적게 든다. 본 연구의 모든 결과는 pre-training model과 정확히 같게 시작했으나 단일 Cloud TPU로 한시간, GPU로는 수 시간만이 걸렸다.

## 4. Experiments
(생략)

## 5. Ablation Studies

### 5.1 Pre-training Tasks
두 pre-training objective로 BERT의 deep bidirectionality의 중요성을 말하고자 함

**No NSP**: MLM은 실시, NSP는 실시 X
**LTR & No NSP**: MLM 대신하여 Left to Right, 즉 Left-context-only model LTR-LM 모델을 학습. Left-only 제약은 fine-tuning에서도 동일하게 실시. 그리고 NSP도 실시 X. OpenAI의 GPT와 비교하기 위해 위와 같이 setting했지만 large dataset, input representation, fine-tuning scheme는 본 BERT 모델과 동일하게 사용

(뒤에 내용은 BERT_BASE보다 못하다를 시사)
- SQuAD에서 LTR은 심각하게 성능이 떨어짐 왜? 직관적으로 LTR모델이니까!
    - token-level hidden state가 right-side context 정보를 가지고 있지 않으니까!
    - 위를 뒷받침하기 위해 bi-LSTM을 쌓으니 성능이 굉장히 많이 상승됐으나 BERT_BASE만 못하더라
- ELMo와 같이 LTR, RTL 모델을 독립적으로 학습시키고 concat하는 것도 가능하겠구나 라고 느꼈지만
    - (1) 단일 bidirectional model은 비용이 두 배로 들어가며
    - (2) 이는 QA같은 경우 직관적이질 못함. RTL은 말이 안됨.
    - (3) 또 deep bidirectional model은 모든 layer에서 양 방향의 맥락을 사용하기 때문에 ELMo는 BERT보다 성능이 떨어짐

### 5.2 Effect of Model Size
BERT 모델의 layer, hidden unit, attention heads의 수를 조절하며 성능을 평가
표를 보면 전부 늘릴 수록 성능은 증가. 하지만 늘어나는 연산량은?

feature-based와 finetune-based의 차이를 계속해서 강조한다.

모델의 크기를 늘리면 기계 번역, 언어 모델링과 같은 대규모 작업에 지속적인 개선이 이어질 것으로 알려져 있고 본 BERT의 LM Perplexity가 모델 사이즈를 늘릴 수록 계속 낮아지는 것으로 확인할 수 있다.

pre-training의 효과가 여기서 부각되는데, 모델의 크기를 키운다, 즉 충분히 사전 학습을 받았을 경우 fine-tune에서 이점이 존재한다고 한다.

Peter는 [Dissecting Contextual Word Embeddings: Architecture and Representation](https://arxiv.org/pdf/1808.08949.pdf)에서 biLM 크기를 2개 층에서 4개 층으로 늘리는 것에서 엇갈린 결과(반대)를 보고했고

Melamud [context2vec: Learning generic context embedding with bidirectional LSTM](https://www.aclweb.org/anthology/K16-1006/)에서 지나가는 말로 hidden dimension size를 200에서 600으로 늘리는 것은 도움이 됐으나 1,000개 이상으로 늘리는 것은 더 큰 개선을 가져오지 못했다고 언급한다.

위 두 방법 모두 `feature-based` 접근 방식을 사용했다. 우리는 모델이 다운스트림 작업에 직접 미세 조정되고 임의로 초기화된 소수의 추가 매개변수만 사용할 경우, 작업 특정 모델은 다운스트림 작업 데이터에서도 더 크고 더 표현력이 뛰어난 사전 훈련된 표현으로부터 이익을 얻을 수 있다고 가정한다(고 한다.)

### 5.3 Feature-based Approach with BERT
BERT는 기본적으로 `fine-tune` 기반 접근 방식이지만 pre-training 모델로 부터 추출된 fixed vector를 사용하는 `feature` 기반의 접근 방식은 몇몇 이점을 가진다. (1) 모든 task가 Transformer encoder 구조로 쉽게 표현되지 않기 때문에 task-specific model 구조가 필요한 경우가 존재한다. (2) training data에서 비싸게(cost) 표현을 얻는 계산을 미리 할 수 있으며 이 표현들을 활용하여 값 싼 모델로 많은 실험을 돌릴 수 있다. (응>? 이건 fine-tune도 가지는 장점 아닌가...)

Table 7을 보면 fine-tune base가 제일 성능이 좋았고 fine-tune based든 feature-based든 bert가 짱이었다. (많은 내용 생략)

## 6. Conclusion
Deep bidirectional architecture로 rich한 표현력을 얻을 수 있게 됨
unsupervised pre-training은 NLU에서 중요한 part를 차지
특히 low-resource task에서 큰 이점을 가짐
BERT는 이러한 것들을 generalize
짱짱맨

## Appendix for "BERT"
- BERT의 구현 Details
- 실험 Details
- Ablation studies

### A. Additional Details for BERT

#### A.1 Illustration of the Pre-training Tasks

**Masked LM and the Masking Procedure**
`my dog is hairy`의 4번째 토큰을 masking한다고 가정하면
- 80% of the time:
    `my dog is hairy -> my dog is [MASK]`
- 10% of the time:
    `my dog is hairy -> my dog is apple`
- 10% of the time:
    `my dog is hairy -> my dog is hairy`
    - representation을 실제 관찰 단어로 bias

위 과정의 장점은 **Transformer encoder가 예측할 단어를 모르도록 한다는 점**. 이는 _모든_ 입력 토큰의 맥락적 표현 분포를 유지하도록 강제. 게다가 random replacement는 모든 토큰의 10~15% 정도만 일어나기 때문에 모델의 language understanding parameter(capacity라고 돼있지만 내용상?)에는 나쁜 영향을 끼치는 기색은 없었다.

기존의 LM 학습과 비교하여 MLM은 각 batch에서 15%의 예측값을 만든다. 때문에 model coverage를 위해 pre-training step이 더 필요할 수도 있다. C.1에서 MLM이 l2r 모델보다 제한적으로 모든 토큰을 예측하는 coverage에서 느리지만 training cost를 증가시킬수록 실험적으로 성능 개선이 됐다.(성능 개선이 된건지 속도 개선이 된건지?)

**Next Sentence Prediction**
NSP task는 아래 예제와 같이 만들어진다.

```
Input = [CLS] the man went to [MASK] store [SEP]
        he bought a gallon [MASK] milk [SEP]

Label = IsNext
```

```
Input = [CLS] the man [MASK] to the store [SEP]
        penguin [MASK] are flight ##less birds [SEP]

Label = NotNext
```

#### A.2 Pre-training Procedure
- 부제: 괴물같은 구글 놈들...

각 훈련 입력 sequence를 생성하기 위해 말뭉치로부터 두 `spans of text`를 추출한다.
- 이는 `sentences`라 부를 것인데 이는 단일 문장보다 보통 길고 짧을 수도 있다고 한다.

첫 문장은 A embedding, 둘 째 문장은 B embedding으로 B는 50% 확률로 A의 실제 다음 문장이 들어오며 나머지 50%의 확률로 임의의 문장이 들어온다. A와 B의 token length가 512개를 넘지 않게 sampling되며 LM masking은 정규 masking rate로 15%의 확률을 적용하여 WordPiece tokenization을 적용한 이후 적용되며 `no special consideration given to partial word pieces`

batch size는 256개의 sequence로 설정했다(256 sequences * 512 tokens = 128,000 tokens / batch). 1,000,000 step으로 40 epoch을 3.3 billion word corpus에서 돌렸다. Adam Optimizer를 사용했고 learning rate는 1e-4, $\beta_1$은 0.9, $\beta_2$는 0.999, L2 weight decay는 0.01을 사용했다. learning rate warmup은 10,000 step이후에 linear decay로 진행했다. 모든 layer에 0.1 dropout probability를 적용했고 relu말고 OpenAI GPT를 따라 gelu activation function을 사용했다. training loss는 MLM, NSP 우도의 평균의 합을 사용했다.

$\text{BERT}_{\text{BASE}}$의 학습은 Pod configuration의 4 Cloud TPUs로 수행됐다(총 16개의 TPU chip을 사용한다). $\text{BERT}_{\text{LARGE}}$의 학습은 16 Cloud TPUs로 수행됐다(총 64개의 TPU chip을 사용한다). 각 pre-training은 끝나는데 4일이 걸렸다.

attention이 sequence length의 quadratic(제곱)이기 때문에 sequence가 길어지면 부분적으로 cost가 더 많이 발생한다. 실험에서 pretraining 속도를 증진시키기 위해 각 step의 90%를 128 length로 제한했다. `Then, we train the rest 10% of the steps of sequence of 512 to learn the positional embeddings`

#### A.3 Fine-tuning Procedure
Fine-tuning에서 대부분의 초모수들은 pre-training과 동일하게 설정했으나 batch_size, learning_rate, training epoch는 다르게 setting했다. drpoout 확률은 항상 0.1로 유지했고 최적의 초모수는 task에 따라 다르지만 전반적으로 최적의 성능을 보였던 parameter는 아래와 같았다.
- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 2, 3, 4

또한 구글놈들은 small datasets보다 large dataset(예로 100k+의 label된 학습 데이터 샘플)은 초모수 선택에 덜 민감하다는 것을 밝혀냈다. Fine-tuning은 통상적으로 매우 빠르며 따라서 위의 매개 변수를 철저히 검색하고 개발 세트에서 가장 잘 수행되는 모델을 선택하는 것이 합리적임.

#### A.4 Comparisoin of BERT, ELMo, and OpenAI GPT
ELMo, GPT와 BERT를 비교해봅시다.
![comparison_elmo_gpt_bert](https://user-images.githubusercontent.com/37775784/79835421-a9f13f80-83e9-11ea-924a-f4c5bdd407e4.PNG)

우선, `BERT`와 `GPT`는 `fine-tuning` 접근 방식 그리고 `ELMo`는 `feature-based` 접근 방식이다.

GPT는 매우 큰 텍스트 말뭉치를 학습한 l2r Transformer LM이다. 사실 BERT의 많은 디자인 결정 사항은 가능한 GPT와 유사하게 만들려고 했고 한 두 가지 방법론을 바꾸려 시도했다. 이 두 가지 차이점은 (1) `bi-directionality`와 (2) `Masked Language Model`, `Next Sentence Prediction`이고 실험적으로 개선시켰으나 몇 가지 세부적인 치이점이 존재하여 아래에 기술한다.
- GPT는 BooksCorpus(800M 단어)에서 학습된 것에 반해 BERT는 BooksCorpus(800M 단어)와 Wikipedia(2,500M 단어)로 학습됐다.
- GPT는 fine-tuning 단계에서만 sentence separator([SEP])과 classifier token([CLS])를 사용하지만 BERT는 [SEP], [CLS], 그리고 문장 A/B embedding은 사전 학습 중에 학습한다.
- GPT는 1M step으로 학습하며 32,000개 단어를 batch로 사용하는데 반해 BERT는 1M step을 학습할 때 128,000개의 단어를 batch로 사용한다.
- GPT는 모든 fine-tuning 실험에서 learning rate를 5e-5를 사용하는데 반해 BERT는 task에 특화된 fine-tuning learning rate를 사용하여 dev set에서 최상의 성능을 가지도록한다.

위 차이점의 효과를 확인하기 위해 ablation 실험을 실시하여 두 pre-training task(MLM과 NSP)와 bidirectionality가 효과적이라는 사실을 증명함.

#### A.5 Illustrations of Fine-tuning on Different Tasks
(생략)
