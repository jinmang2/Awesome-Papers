# Neural Architectures for Named Entity Recognition

#### Authors
- Guillaume Lample, (1)
- Miguel Ballesteros, (1), (2)
- Sandeep Subramanian, (1)
- Kazuya Kawakami, (1)
- Chris Dyer, (1)

#### Institution
(1): Carnegie Mellon University
(2): NLP Group, Pompeu Fabra University

## Abstract
SOTA NER은 손수 제작한 feature와 작은 지도학습용 말뭉치를 효율적으로 학습하기 위해 도메인 특화 지식에 심히 의존적. 본 논문에서 아래 두 개의 모델 구조를 소개
- biLSTM-CRF
- transition-based labels segments (shift-reduce parsers에 영감을 받음)

위 모델은 단어 정보의 아래 두 source에 의존적
- 지도학습 말뭉치를 학습한 `문자 기반 단어 표현`
- 주석되지 않은 말뭉치(unlabeled)를 학습한 `비지도 단어 표현`

## 1. Introduction
NER은 도전적인 문제이나 현업에서 labeled된 데이터는 매우 소량임. 이렇게 적은 데이터로 일반화하는 것은 어려움. 때문에 가제티어(gazetteers)와 같은 언어 특화 지식 소스 및 정사영 특징을 조심스럽게 구축해왔으나 언어에 특화된 리소스 및 특징은 새로운 언어, 새로운 도메인에 적용되긴 비용이 비쌈. label되지 않은 말뭉치를 비지도 학습하는 것은 지도말뭉치 대비, 언어모델을 일반화할 대체 전략이지만, 이렇게 비지도 feature에 의존하는 시스템들도 hand-engineered features, specialized knowledge resources를 대체하기 보단 이를 보강하기 위해 해당 feature를 사용해왔다.

본 연구에선 위와 대비되는 NER을 위한 neural architectures를 소개하고자 한다. 본 논문에서 소개할 모델은 아래 두 직관을 포착할 수 있게 설계됐다.

---
**First**, `이름`이 때론 다양한 토큰으로 구성되기 때문에, 각 토큰에 대해 어떤 태그를 붙일 것인지에 대해 `공동 추론(reasoning jointly)`이 중요하다.
이를 위해 두 모델을 비교한다.
1. biLSTM-CRF
2. S-LSTM을 사용한 입력 문장의 chunk를 구축, label을 지정하는 새로운 모델

---
**Second**, `이름이 되는` 토큰 단위의 증거. 이는 아래 두 증거를 포함한다.
- `orthographic evidence`: `name`이라 태그된 단어의 형상은?
- `distributional evidence`: `name`이라 태그된 단어가 말뭉치에서 어디서 발생하는 경향이 있는지?

orthogonal sensitivity를 포착하기 위해 문자 단위 단어 표현 모델을 사용했고
distributional sensitivity를 포착하기 위해 Word2Vec과 문자 단위 단어 표현 모델을 결합하여 사용했다.

우리의 단어 표현은 이들의 결합으로 표현되며 dropout training을 사용하여 모델이 위 evidence를 신뢰할 수 있게 학습할 수 있게 만듦. (음...? 근거는?)

---

영어, 네덜란드어, 독일어, 스페인어에 대한 실험으로 NER SOTA를 달성했고 뭐 여러가지로 잘 됬다고 한다. `transition-based 모델`도 `LSTM-CRF`보다 성능이 좋진 않았지만 성능이 매우 굳!

## 2. LSTM-CRF Model
위 구조는 아래 두 논문과 유사하다고 함
- Collobert et al. (2011)
- Huang et al., (2015)

### 2.1 LSTM
LSTM에 대한 설명. 몰랐던 부분만 발췌
- (Bengio et al., 1994) 이론적으로 RNN이 long dependencies를 학습할 수 있다지만 실제론 sequence의 최근 input에 편향되는 경향이 존재, 실패한다.
- 이에 대한 대안에 LSTMs. memory-cell을 추가하여 long-range dependencies를 잡도록 설계
- 양방향 LSTM으로 설계 가능. bidirectional LSTM은 Graves와 Schmidhuber가 2005년에 제시
- 위 양방향 context정보를 포착하도록 설계하면 방대한 tagging api에서 효율적으로 사용할 수 있을 것.

### 2.2 CRF Tagging Models
아직 잘 모르기에 여긴 잘 읽어보자.
- 아주 간단한, 그러나 놀랄만큼 효과적인 tagging model은 (Ling et al., 2015b)에 나온 것과 같이 각 output $y_t$에 대해 독립적인 tagging 결정을 만드는 feature로 $h_t$를 사용한다.
- POS Tagging과 같이 간단한 문제엔 성공적이었던 CRF는 출력 레이블간에 강력한 종속성이있는 경우 독립 분류 결정이 제한됨.
- NER이 위와 같은 경우인데, 해석가능한 tag sequence를 특징 짓는 `문법`은 독립 가정으로 모델링할 수 없는 어려운 제약을 부과하기 때문
    - 예로, I-PER은 B-LOC를 따를 수 없다.
- 고로 독립적으로 tagging 의사결정을 모델링하기 보단, Conditional Random Field를 사용해 동시에 모델링하는 방법을 택한다.
    - 이 모델은 Lafferty가 2001년에 제안

모델 수식 설명 가즈아~

주어진 Input Sentence $X$에 대해,
$$\text{X}=(\text{x}_1,\text{x}_2,\dots,\text{x}_n)$$

$\text{P}$를 biLSTM의 출력 점수의 행렬이라 일컫자. $\text{P}$는 $n \times k$의 사이즈를 가지며 $k$는 구분 tag의 수, $P_{i,j}$는 문장의 $i^{th}$단어의 $j^{th}$tag의 점수를 의미한다. 예측 sequence $y$가 아래와 같을 때,
$$\text{y}=(y_1,y_2,\cdots,y_n)$$

score $s$를 아래와 같이 정의한다.
$$s(\text{X},\text{y})=\sum_{i=0}^{n} A_{y_i,y_{i+1}}+\sum_{i=1}^{n} P_{i,y_i}$$

갑자기 튀어나온 $\text{A}$는 전이 scores 행렬. $A_{i,j}$는 tag $i$에서 tag $j$로 전이(transition)할 점수를 표현. $y_0$과 $y_n$은 문장 tag의 _start_ 와 _end_ 를 의미. 앞의 두 tag를 추가하기 때문에 $\text{A}$는 size $k+2$를 가지는 정방 행렬.

가능한 모든 tag sequences의 softmax는 sequence $\text{y}$에 대한 확률을 도출한다.
$$p(\text{y}|\text{X})=\cfrac{e^{s(\text{X},\text{y})}}{\sum_{\hat{\text{y}}\in\text{Y}_{\text{X}}}e^{s(\text{X},\hat{\text{y}})}}$$

학습 도중, 정답 tag sequence의 로그확률을 maximize
$$\begin{aligned}
\log{(p(\text{y}|\text{X}))}&=s(\text{X},\text{y})-\log\bigg(\sum_{\hat{\text{y}}\in\text{Y}_{\text{X}}}e^{s(\text{X},\hat{\text{y}})}\bigg)\\
&=s(\text{X},\text{y})-\text{logadd}_{\hat{\text{y}}\in\text{Y}_{\text{X}}}e^{s(\text{X},\hat{\text{y}})}\cdots(1)\\
\end{aligned}$$

$\text{Y}_{\text{X}}$는 문장 $\text{X}$에 대한 모든 가능한 tag sequences(IOB 형식을 검증하지 않은 경우도). 위의 공식으로부터, 우리의 네트워크가 유효한 출력 라벨의 시퀀스를 생산하도록 만드는 것을 확인할 수 있다. 디코딩하는 동안 다음과 같이 주어진 최대 점수를 얻는 출력 시퀀스를 예측한다.
$$\text{y}^\ast=\argmax_{\hat{\text{y}}\in\text{Y}_{\text{X}}}s(\text{X},\hat{\text{y}})\cdots(2)$$

위에서 output 사이에 bigram interaction으로 모델링했기 때문에 Eq(1)의 summation과 Eq(2)의 사후 sequence $\text{y}^\ast$의 최댓값은 동적 프로그래밍을 사용하여 계산될 수 있다.(왜? 공부할 것 투성이군...)

### 2.3 Parameterization and Training
각 토큰에 대한 각 tagging decision과 관련된 점수(즉, $P_{i,y}$'s)들은 _biLSTM으로 계산된 단어-문맥(a)_ 과 _bigram compatibility scores(즉, $A_{y,y^{\prime}}$'s)로 결합된_ 임베딩들 간의 dot product로 계산된다.
- (a): 이는 Ling et al. (2015b)의 POS tagging 모델과 정확히 일치한다.
- compatibility: 적합성, 호환성

이 구조는 Figure 1.에서 확인할 수 있다.
![bilstmcrf](https://user-images.githubusercontent.com/37775784/80556403-6f515d80-8a0e-11ea-87e5-89291f87aba2.PNG)
- `Circle`은 observed variable을,
- `Diamond`는 각 부모 노드의 결정 함수를,
- `Double circle`은 random variable을 의미한다.

이 모델의 모수는 아래와 같다.
- $\text{A}$: bigram compatibility scores 행렬
- 행렬 $\text{P}$를 만들 모수들,
    - biLSTM
    - linear feature weights
    - word embeddings

$x_i$: 문장의 모든 단어에 대한 단어 embedding sequence
- 어떻게 임베딩할 지는 Section 4에서 다룸

$y_i$: 위에 해당하는 tags

word embedding sequence는 biLSTM의 input으로 feeding.

이러한 표현들은 $c_i$로 concat되며 구분된 tag의 숫자만큼의 크기를 가진 layer로 선형 사영된다. 이 계층의 softmax output을 사용하는 대신, 우리는 앞에서 설명한 대로 모든 단어 y에 대한 최종 예측값을 산출하면서 인접 태그를 고려하기 위해 CRF를 사용한다. 추가적으로 $c_i$와 CRF 계층 사이에 hidden layer를 추가하면 성능을 약간 상승시킬 수 있다는 것을 확인했다. 모든 결과에는 이 추가적인 layer가 삽입된다. 모수는 label된 말뭉치의 주어진 관측 단어의 NER tag의 관측 sequence의 Eq(1)을 최대로 만들며 학습된다.

---
근데, 보면 annotated corpus가 내가 생각하는 label이 안된 말뭉치가 아닐 수도?? 결국 NER tagging이 된 dataset으로 진행하는 것 아닌가?

---

### 2.4 Tagging Schemes
NER은 문장의 모든 단어에 named entity 라벨을 할당하는 task. 단일 named entity는 문장 내부의 몇몇 token을 span할 수 있다. 문장은 보통 IOB format으로 표현된다.
- B-label(Beginning): 토큰이 named entity의 beginning일 때
- I-label(Inside): 토큰이 named  entity의 내부에 있고 첫 번째 토큰이 아닐 때
- O-label(Outside): Otherwise
(...음? 설명이 뭔가 안 와닿는다...)
- https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
- https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format
- https://text-processing.com/demo/tag/ 이런 것도 있네

자, 본 연구에서는 아래 S와 E labeling을 추가한 IOBES tagging scheme를 사용한다!
- 이는 IOB의 변종
- S-label: singleton entities에 대한 정보를 부호화
- E-label: named entities의 끝을 명시적으로 표기

이 방법을 사용하여 신뢰도가 높은 I-label로 단어를 태그하면 후속 단어의 선택이 I-label 또는 E-label로 좁혀지지만 IOB 체계는 후속 단어가 다른 레이블의 내부가 될 수 없다는 것을 결정할 수 있을 뿐이다. Ratinov and Roth (2009)와  Dai et al. (2015)은 IOBES와 같은 좀 더 표현적인 태깅 방식을 사용하는 것이 근소하게 모델 성능을 향상시킨다는 것을 보여주었으나 IOB 태깅 체계에 비해 크게 개선된 점은 관찰하지 못했다.

## 3. Transition-Based Chunking Model
(생략)


## 4. Input Word Embeddings

모델의 input layers는 각 단어의 vector 표현. 제한된 NER 학습 데이터로부터 단어 유형의 독립적인 표현을 학습하는 것은 어려움: 추정하기 위해 너무 많은 모수가 필요함. 많은 언더들이 orthographic(형상학?) 혹은 morphological(형태학) 적인 요소(evidence)를 가지기 대문에 단어 철자에 민감한 표현을 원한다. 고로 (4.1)의 char-based model을 활용, 문자 표현으로부터 단어 표현을 구축한다.

두 번째 직관은 상당히 다양한 이름들이 대량의 말뭉치에서 규칙적인 맥락 하에 등장할 것이라는 것. 그러므로 단어에 민감한 (4.2) Pretrained embeddings를 사용하여 대량의 말뭉치로부터 학습된 embedding을 사용한다.

최종적으로, 모델이 단일 혹은 다른 표현에 너무 의존하는 것을 방지하기 위해 (4.3)의 Dropout training을 활용, 일반화 성능에서 중요한 부분을 차지한다.

### 4.1 Charcter-based models of words

본 연구가 이전 연구와 구별되는 중요한 부분은 단어로부터 prefix, suffix 정보를 손수 제작하여 학습하는 대신 문자 단위의 특징을 학습한다. **문자 단위 임베딩을 학습하는 것은 당면한 과제와 특정한 표현을 학습할 수 있다는 장점이 있다.**

---
- 오...? 인용한 어구인가 뭐지... 그래? 문자 단위로 학습하는게?

---
Ling과 Ballesteros는 그들의 논문에서 POS tagging과 LM / Dependency parsing과 같은 task에서 OOV 문제를 핸들링하고 형태학적으로 더 풍성한 단어를 사용할 수 있음을 밝혔다.

---
- 아... 인용인가보네. 아니 저 두 논문은 뭐길래 저러지?
```
Ling et al., (2015b)
Wang Ling, Tiago Lu´ıs, Lu´ıs Marujo, Ramon Fernandez ´
Astudillo, Silvio Amir, Chris Dyer, Alan W Black, and
Isabel Trancoso. 2015b. Finding function in form:

"Compositional character models for open vocabulary word representation."

In Proceedings of the Conference on Empirical Method

--------------------------------------------------------

Ballesteros et al., (2015) - 본 논문의 저자기도 함!

Miguel Ballesteros, Chris Dyer, and Noah A. Smith. 2015.

"Improved transition-based dependency parsing by modeling
characters instead of words with LSTMs."

In Proceedings of EMNLP
```

- Chris Dyer, Miguel Ballesteros, Wang Ling, Austin Matthews, and Noah A. Smith. 2015. `Transitionbased dependency parsing with stack long short-term memory.` In Proc. ACL. / 이런 논문도 있네?

- Miguel Ballesteros, Yoav Golderg, Chris Dyer, and Noah A. Smith. 2016. `Training with Exploration Improves a Greedy Stack-LSTM Parser`. In arXiv:1603.03793.

---

![akdsj](https://user-images.githubusercontent.com/37775784/80562819-d75e6e80-8a23-11ea-961f-7795c40bba8b.PNG)

위 Figure 4는 문자로부터 단어 임베딩을 어떻게 생성할지를 묘사. 임의로 초기화된 문자 lookup table은 모든 문자에 대한 embedding값을 포함. 단어의 모든 문자에 해당하는 문자 임베딩은 정방향 및 역방향 LSTM에 직접 및 역순으로 주어지고 문자로부터 파생된 단어 임베딩은 biLSTM의 양 방향 표현을 concat하여 만들어진다. 문자 단위의 표현은 단어 lookup table의 문자 단위 표현으로 concat된다. 실험하는 동안 lookup table에 없는 단어 임베딩은 UNK Embedding으로 매핑한다. UNK Embedding을 학습시키기 위해 singletons을 50%의 확률로 UNK Embedding으로 변화시켰다. 모든 실험에서 bi-char-LSTM의 hidden dimension은 각각 25로 그 결과 차원이 50인 단어의 문자 기반 표현이 된다.

RNN과 LSTM과 같은 반복 모델은 매우 긴 시퀀스를 인코딩할 수 있지만, 가장 최근의 입력에 편향된 표현을 가지고 있다. 우리는 forward LSTM의 최종 표현이 단어의 접미사(suffix)가 되길 원하고 backward LSTM의 최종 표현이 단어의 접두사(prefix)가 되길 원한다. 대안 방법으로 Convolution Networks와 같은 방법을 사용하는 것이 문자로부터 단어 표현을 학습하는데 제안되었다. (Zhang et al., 2015 / Kium et al., 2015)
그러나, convnet은 input의 position-invariant 특징을 찾도록 설계됐다.
- https://towardsdatascience.com/translational-invariance-vs-translational-equivariance-f9fbc8fca63a
- http://adityaarpitha.blogspot.com/2015/11/position-invariant.html
- https://www.osapublishing.org/ao/abstract.cfm?uri=ao-36-14-3035
- https://www.nature.com/articles/nn1519

이는 image recognition 등에서는 적절한 방법이지만(예로 그림에서 고양이가 어디에 등장하는지 무관) 우리가 필요로 하는 정보는 position dependent하다(접두사와 접미사는 stem과 다른 정보를 부호화한다.)이며 또 우리는 LSTMs를 단어와 문자 사이의 관계를 모델링하기 위한 더 나은 function class로 만드는 것이다.

---
- ??? 그래서 결국 어떻게 했다는 얘기?

---

### 4.2 Pretrained embeddings
Collobert et al. (2011)과 같이 본 연구에선 lookup table을 초기화하기 위해 사전 학습된 단어 임베딩을 사용한다. 무작위로 초기화된 단어 임베딩보다 사전학습된 단어 임베딩을 사용하여 더 큰 성능 향상을 이룰 수 있었고 이 임베딩은 skip-n-gram(Ling et al., 2015a)와 word2vec(Mikolov et al., 2013a)의 변종을 사용하여 사전학습했다. 이 임베딩은 학습 중 fine-tuned된다. (feature-based? real fine-tune? 아마 후자)

- 아래를 데이터로 학습했고 window size 8, 단어 빈도 컷 4, 뭐 영어는 100차원 나머진 64차원으로 진행했다고 한다. 원문은 아래에 있다.
```
Word embeddings for Spanish, Dutch, German
and English are trained using the Spanish Gigaword
version 3, the Leipzig corpora collection, the German
monolingual training data from the 2010 Machine
Translation Workshop and the English Gigaword version 4
(with the LA Times and NY Times portions removed) respectively.
We use an embedding dimension of 100 for English,
64 for other languages, a minimum word frequency cutoff of 4, and
a window size of 8.
```

### 4.3 Dropout training
초기 실험에서는 문자 수준의 임베딩이 사전 학습된 단어 표현과 함께 사용될 때 전체적인 성능을 향상시키지 못했다. 모델이 양쪽 표현에 더욱 의존하도록 Hinton의 Dropout training을 사용했고 biLSTM의 input으로 넣어주기 전에 최종 embedding layer에 dropout mask를 적용했다. dropout을 적용한 후에 모델 성능이 매우 뛰어나게 상승한 것을 확인할 수 있었다.

## 5. Experiments

### 5.1 Training

**1문단**
- 모수 업데이트를 위해 역전파 알고리즘 사용
- SGD with 0.01 learning rate
- gradient clipping with 5.0
- Adadelta, Adam 등이 수렴 속도는 빨랐지만 SGD with gradient clipping의 성능을 넘진 못함.

**2문단**
- LSTM-CRF는 차원 100을 가진 bi-LSTM 단일 레이어를 사용
- 위 100 차원을 수정하는 것은 성능에 어떠한 영향도 끼치지 못함
- dropout rate는 0.5
- 위보다 더 높은 비율을 세팅하면 성능이 하락, 낮게 세팅하면 학습 속도가 느려짐

**3문단**
- 각 stack별 100차원을 가지는 두 layer로 적층한 stack-LSTM도 고려
- 구성함수에 사용된 action 임베딩은 각각 16차원을 가지며, 출력 임베딩은 20차원.
- dropout rate를 다양하게 실험, 각 언어별 최상의 dropout rate로 점수를 보고
- 추론을 위해 전체 문장을 처리할 때 까지 국소 최적점을 찾는 greedy model을 사용했으며 beam search와 탐색적 학습을 통해 성능 개선을 이뤄낼 수 있었다.
    - beam search: Zhang and Clark, 2011
    - training with exploration: Ballesteros et al., 2016

### 5.2 Data sets
뭐, CoNLL-2002, CoNLL-2002으로 테스트했다네요? 본문 내용 참고
```
We test our model on different datasets for named
entity recognition. To demonstrate our model’s
ability to generalize to different languages, we
present results on the CoNLL-2002 and CoNLL2003
datasets (Tjong Kim Sang, 2002; Tjong
Kim Sang and De Meulder, 2003) that contain
independent named entity labels for English, Spanish, German and Dutch.
All datasets contain four
different types of named entities:
locations, persons, organizations, and
miscellaneous entities that
do not belong in any of the three previous categories.
Although POS tags were made available for
all datasets, we did not include them in our models.
We did not perform any dataset preprocessing, apart
from replacing every digit with a zero in the English
NER dataset.
```

### 5.3 Results
(생략)

### 5.4 Network architectures
(생략)

## 6. Related Work
(생략)


## 7. Conclusion
(생략)
