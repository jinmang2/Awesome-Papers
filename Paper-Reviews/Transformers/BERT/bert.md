# BERT 논문 리뷰

## BERT?
- Bidirectional Encoder Represenations from Transformer
- `AllenAI`의 `ELMo`, `OpenAI`의 `GPT`와 다르게 `BERT`는 **"양 방향의 맥락*"* 을 **"동시에"** 학습할 수 있는 **"Deep Bidirectional Representations"** 을 사전 학습함
- pre-trained BERT 모델은 QA, language inference 등에서 task 특화된 구조로 수정할 필요없이 광범위한 nlp task에서 고작 output layer 하나를 추가하여 fine-tuning한 것만으로 SOTA를 달성

![img](https://lh3.googleusercontent.com/proxy/-HTcJqMiLfCveFm_8-ckc9jIKuYAUYCFezRCQp9kNJo1Rh3Y915kGQUsYWZ-OIvFLu6vlpp4wvbAAjNHZjWKO1uRjV8PK821wiN75K6HRnzhbMPdXUnXazgyx6hwhrc)

## Pre-training LM은 NLP task에서 매우 뛰어났음!
Language Model pre-training은 자연어 처리 task에서 매우 효과적이었음
- [시초:] Semi-supervised sequence learning, Dai and Le, 2015
- [ELMo:] Deep contextualized word representations, Peters et al., 2018a
- [GPT1:] Improving language understanding with unsupervised learning, Radford et al, 2018
- [Fine-tune based:] Universal language model fine-tuning for text classification, Howard and Ruder, 2018

token-level task에서도 뛰어났고
- token-level task란, model이 output을 token 단위로 fine-grained(잘게 쪼개진)하게 생산할 것을 요구되는 task
- named entity recognition과 question answering 등이 존재
- introduction to th conll-2003 shared task: language-independent named entity recognition, Tjong Kim Sang and De Meulder, 2003
- SQuad: 100,000+ questions for machine comprehension of text, Rajpurkar et al., 2016

sentence-level task에서도 우수한 성능을 보임
- natural language inference(NLI)
  - A large annotated corpus for learning natural language inference, Bowman et al., 2015 (EMNLP)
  - A board-coverage challenge corpus for sentence understanding through inference, Williams et al., 2018
- Paraphrasing
  - Automatically constructing a corpus of sentential paraphrases, Dolan and Brockett, 2005


![img](https://mino-park7.github.io/images/2018/12/%EA%B7%B8%EB%A6%BC1-bert-openai-gpt-elmo-%EC%B6%9C%EC%B2%98-bert%EB%85%BC%EB%AC%B8.png)

## 기존 Pre-training 모델, ELMo, GPT의 문제?
Pre-training을 적용시키는 방법은 두 가지가 존재!
- **feature-based approach** : 특정 task를 수행하는 network에 pre-trained language representation을 추가적인 feature로 제공. 즉, 두 개의 network를 붙여서 사용한다고 보면 됩니다. 대표적인 모델 : ELMo(Peters et al., 2018)
- **fine-tuning approach** : task-specific한 parameter를 최대한 줄이고, pre-trained된 parameter들을 downstream task 학습을 통해 조금만 바꿔주는(fine-tuning) 방식. 대표적인 모델 : Generative Pre-trained Transformer(OpenAI GPT) (Radford et al., 2018)

ELMo는 양방향을 independent하게 학습을 하고
- 이를 Shallow Bidirectional이라 부름
- 단방향 concat 단방향

GPT는 단방향 Transformer Decoder(생성모델이기 때문)을 사용
- Unbidirectional

ELMo와 GPT1의 objective function(pre-training시)는 동일!
- 이전 토큰으로 다음 토큰 예측(방향만 다를 뿐)
- Language Model

## BERT의 pre-training 2 objective: MLM, NSP
BERT는 위에서 언급한 unidirectional 한 제약을 `masked language model(MLM)` pre-training objective를 사용하여 문제를 해결하고자 함!
- MLM pre-training objective는 Cloze task(Cloze procedure: A new tool for measuring readability, Taylor, 1953)에서 영감을 얻었다고 함!


masked langauge moddel은 input token의 일부분을 random하게 masking!
그리고 objective는 masking된 token id를 오직 문맥으로 original vocab id을 예측하는 것이 목표!

`left-to-right` LM pre-training, 그러니까 GPT랑 다.르.게! MLM objective는 deep-bidirectional transformer를 pre-training할 수 있게 left와 right context를 연결해주는(fuse) 표현을 가능케해줌!
- `left-to-right`와 `right-to-left`를 간결하게 쓴 말!

MLM에 더하여 `next sentence prediction` task를 사용!
- text-pair 표현을 동시에 pre-training하는 task

## 본 논문의 contribution
1. Bidirectional Pre-training (fine-tuning based)
  - GPT와 다르게 Language representation에 있어 `bidirectional`한 pre-training 방식의 중요성을 부각시킴
  - ELMo와도 다름! ELMo는 `left-to-right`와 `right-to-left` LM을 각각 독립적으로 학습한 후에 shallow concat을 사용한다구!

2. NLP task에 specific하지 않고 universal하게 적용 가능! SOTA신기록 까지!
  - BERT는 sentence-level과 token-level task에서 SOTA를 당성
  - 많은 task-specific한 구조에서도 출중!
  - 이는 task-specific하게 구조를 완전 수정하지 않고도 달성!!(맞나:?)

3. 11개의 NLP task에 대해 SOTA를 개선
  - 코드는 https://github.com/google-research/bert 에서 확인 가능

## Model Architecture
BERT는 Transformer의 Encoder 부분을 활용!
- http://nlp.seas.harvard.edu/2018/04/03/attention.html

**Notation**
본 연구에서 Layer(i.e., Transformer Block)의 수를 $L$로, hidden size를 $H$, self-attention heads의 수를 $A$로 표기하도록 한다.
  - 모든 경우에서 feed-forward/filter size를 4H로 설정했다.
  - 즉, $H=768$인 경우엔 $3072$, $H=1024$인 경우엔 $4096$이 된다.

본 연구에서 두 모델을 제시한다;
- $\text{BERT}_{\text{BASE}}$ (L=12, H=768, A=12, Total Parameters=110M)
- $\text{BERT}_{\text{LARGE}}$ (L=24, H=1024, A=16, Total Parameters=340M)

$\text{BERT}_{\text{BASE}}$는 OpenAI GPT와의 비교를 위해 같은 모델 size를 가지도록 setting함. 그.러.나! BERT Transformer는 `bidirectional self-attention`을 사용하고 GPT Transformer는 몯느 token이 왼쪽의 context에만 집중할 수 밖에 없는 제한된 self-attention을 사용한다!!
- GPT: `Transformer decoder`, left-context-only version
- BERT는 `Transformer encoder`를 사용, bidirectional하게!!

## BERT Input/Output
두 가지 경우로 나뉘어짐
- Single Sentence
- Pair of Sentences

다양한 down-stream task를 처리하기 위해 한 개의 문장이든 두 개의 문장이든 하나의 token sequence로 나타냄.

논문에서 소개하는 방법 및 규칙은 아래와 같다.
- Wordpiece Embedding 수행
- 첫 token은 항상 special classification token([CLS])
  - 위 토큰의 final hidden state는 분류 문제에서 sequence 표현을 aggregate하는데 사용
- 문장의 쌍이 input으로 들어온 경우, special token([SEP])을 문장 사이에 넣고 token이 문장 A에 있는지, 문장 B에 있는지 학습한 embedding을 추가한다(masking을 넣는다고 이해해라).

input embedding을 $E$로 표기, special [CLS] token의 final hidden vector를 $C\in\mathbb{R}^H$로, $i^{th}$번째 input token의 최종 hidden vector를 $T_i\in\mathbb{R}^H$로 표기한다.

주어진 token에 대해 input 표현은 token, segment, position embedding을 더하여 구성된다.

![img](https://user-images.githubusercontent.com/1250095/50039788-8e4e8a00-007b-11e9-9747-8e29fbbea0b3.png)

## Task #1: Masked LM

#### 왜 이전 모델들은 양방향으로 학습하지 않았을까?
- 직관적으로 양방향의 맥락을 이해하는 것이 단순하게 단방향으로 학습하는 것보다 강력할 것
- 왜 그렇게 하지 못했을까? 양방향성으로 본다는 것은 간접적으로 **"자신을 보는 것"** 을 가능케 한다.
- 때문에 모델이 cheating하는 것이 가능해지기 때문에 불가능했던 것.

#### Masking해서 cheating을 방지하자.
- deep bidirectional representation을 학습하기 위해 본 연구에선 input token을 무작위로 일부 `masking`하고 그 `masking`된 token을 예측하도록 했다.
- `masked LM(MLM)`, 1953 Cloze procedure에서 영감을 받음
- 이 경우, mask token의 최종 hidden vector는 표준 LM과 같이 vocab을 통해 output softmax로 feeding
- 본 모든 실험 과정에서 무작위로 각 sequence의 `WordPiece token`들의 15%를 masking했다.
- `denoising auto-encoders`와는 다르게 전체 input을 재구축하는 것이 아닌 masking된 단어를 예측
  - Extracting and composing robust features with denoising autoencoders, Vincent et al., 2008
    - Bengio 교수!

#### Masking을 확률적으로 실시
- 이로 인해 bidirectional하게 모델을 pre-training시키는 것이 가능해짐
- 그러나 fine-tuning에선 [MASK] token이 등장하지 않기 때문에 pre-training과 fine-tuning 사이에서 downside에 불균형(mismatch)이 생김
- 이를 완화(mitigate)시키기 위해 'masked'된 토큰을 실제 [MASK] 토큰으로 항상 대체시키진 않음
- 아래의 방식으로 진행합니다.
```
My dog is hairy라는 문장이 있다고 가정하자.

학습 데이터 생성자는 예측을 위해 무작위로 15%의 token position을 고른다.
만일 4번째 토큰이 골라졌다고 가정하자.
그러면 BERT는 4번째 토큰을
  (1) 80%의 확률로 [MASK] 토큰으로,
      My dog is hairy -> My dog is [MASK]
  (2) 10%의 확률로 임의의 토큰으로 변화시키고
      My dog is hairy -> My dog is apple
  (3) 10%의 확률로 그대로 둔다.
      My dog is hairy -> My dog is hairy
그러면 cross entropy loss로 original token를 예측하기 위해 T_i가 사용될 것이다.
```

위 과정의 장점은 **Transformer encoder가 예측할 단어를 모르도록 한다는 점**. 이는 _모든_ 입력 토큰의 맥락적 표현 분포를 유지하도록 강제. 게다가 random replacement는 모든 토큰의 10~15% 정도만 일어나기 때문에 모델의 language understanding parameter(capacity라고 돼있지만 내용상?)에는 나쁜 영향을 끼치는 기색은 없었다.

기존의 LM 학습과 비교하여 MLM은 각 batch에서 15%의 예측값을 만든다. 때문에 model coverage를 위해 pre-training step이 더 필요할 수도 있다. C.1에서 MLM이 l2r 모델보다 제한적으로 모든 토큰을 예측하는 coverage에서 느리지만 training cost를 증가시킬수록 실험적으로 성능 개선이 됐다.(성능 개선이 된건지 속도 개선이 된건지?)

#### Task #2: Next Sentence Prediction (NSP)
QA와 NLI같은 NLP에서 중요한 downstream task들은 두 문장 사이의 `relationship`을 이해하는 것에 기반을 둔다. 이 두 문장 사이의 "관계"는 보통 LM으로는 직관적으로 포착되지 않는다. 모델이 문장 사이의 관계를 이해할 수 있도록 단일 언어 코퍼스에서 생성된 `binarized next sentence prediction task`를 pre-train시킨다. 특별한 경우로 각 사전학습 예제에서 sentence A와 sentence B를 뽑았다고 가정하자. 각 step별 50%의 확률로 B는 A의 실제 다음 문장(이를 $\text{IsNext}$라고 라벨링), 그리고 50%의 확률로 B를 corpus의 임의의 문장으로 추출된다(이를 $\text{NotNext}$라고 라벨링). $C$는 다음 sentence prediction을 위해 사용된다(최종 모델은 NSP에서 97-98%의 정확도를 달성했다). 이렇게나 단순함에도 불구하고 QA와 NLI 모두에서 너무나 좋은 효과를 보여줬다!!
- $C$는 NSP에서 학습됐기 때문에 fine-tuning이 없으면 의미있는 문장 표현을 내포하지 못한다.

NSP(Next sentence Prediction) task는 아래 두 논문에서 사용된 표현 학습 objective과 긴밀한 연관성이 있다.
- Discourse-Based Objectives for Fast Unsupervised Sentence Representation Learning, Jernite et al., 2017    
- An efficient framework for learning sentence representations, Logeswaran and Lee, 2018

그러나 선행 연구에서는 오직 문장 임베딩만이 down-stream task로 전달되는데 반해 BERT는 end-task model의 모수를 초기화하기 위해 모든 parameter를 전이(transfer)한다.

## Pre-training data
사전 학습 과정은 현존하는 LM pre-training 자료들을 따른다. pre-training corpus로 BooksCorpus(800M words)와 English Wikipedia(2,500M words)를 사용했다. Wikipedia의 경우 오직 텍스트 구절만 추출하고 리스트, 표, 머리말(header)는 배제(ignore)했다. 또 wikipedia같은 문서 단위의 말뭉치를 사용하는 것이 길고 연속적인 sequence를 추출하기 위한 Billion Word Benchmark와 같은 셔플된 문장 단위의 말뭉치를 사용하는 것보다 더 중요하게 작용했다.

### 3.2 Fine-tuning BERT
- BERT의 fine-tuning은 간단하다.
- NLP의 여러 태스크가 있을 것이다.
  - Paraphrasing
  - Textual Entailment
  - Question Answering
  - Text Classification
  - Sentiment Analysis
- token representation은 task의 output layer로,
- [CLS] representation은 분류 output layer로 feeding (어떤 task인지로!)

즉, text-pair를 encode하는 과정과 bidirectional cross attention을 적용하는 두 과정이 존재한다. BERT는 이 두 단계를 통합하기 위해 self-attention mechanism을 사용한다. concat된 text-pair를 self-attention으로 부호화하는 것은 두 문장 간의 bidirectional cross attention을 포함하기 때문이다.

각 task에서 단순하게 BERT에 task 특화 input과 output을 넣어주고 모든 parameter를 end-to-end로 fine-tuning했다. pre-training의 모든 input sentence A와 sentence B는 (1) paraphrasing의 sentence pair, (2) `hypothesis-premise pairs in entailment`, (3) QA의 question-passage 쌍, (4) text 분류 혹은 sequence tagging의 degenerate text-$\empty$ 쌍과 유사하다. Output에서 token representation은 sequence tagging 혹은 question answering과 같은 token 단위 task의 output layer로 feeding되며 [CLS] 표현은 entailment나 sentiment analysis와 같은 분류 output layer로 feeding된다.

사전 학습과 비교하여 fine-tuning은 비교적 cost가 적게 든다. 본 연구의 모든 결과는 pre-training model과 정확히 같게 시작했으나 단일 Cloud TPU로 한시간, GPU로는 수 시간만이 걸렸다.

#### A.1 Illustration of the Pre-training Tasks





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
