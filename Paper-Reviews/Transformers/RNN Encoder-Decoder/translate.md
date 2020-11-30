# RNN Encoder-Decoder for SMT

## Abstract
2개의 RNN으로 이루어진 RNN Encoder-Decoder 모델 소개
- Encoder: Map `sequence of symbols` to `fixed-length vector representation`
- Decoder: Map `vector representation` to `another sequence of symbols`

Maximize P(target seq | source seq) (Contidional probability)

Encoder와 Decoder를 동시에 학습! (Jointly train)

기존의 log-linear model에 RNN Encoder-Decoder를 feature로 추가하여 계산한 phrase pairs의 조건부 확률 P(S_target | S_source)를 사용하여 SMT의 성능을 향상!

SMT: Statistical Machine Translation

정성적으로, RNN Encoder-Decoder가 언어학의 [phrase](https://en.wikipedia.org/wiki/Phrase)의 semantic/syntactic한 의미있는 표현을 학습한다는 것을 보였다.

## 1. Introduction

### DNN의 성공적인 사례들, SMT에서도 효과적
Deep Neural Network는 object dectection / speech recognition에서 큰 성과를 거둠

게다가 NLP의 여러 task에서도 성공적으로 사용되고 있음
- Language Modeling
- Paraphrase Detection
- Word Embeddign Extraction

Statistical Machine Translation 필드에서도 DNN은 약속된 결과를 보여왔음

(Schwenk, 2012)가 phrase-base SMT system에서 DNN이 거둔 성공적인 사례들을 요약해둠

### 본 논문의 focusing

본 논문에선 Novel Neural Network architecture for phrase-based SMT system에 초점을 맞추고자 함

`RNN Encoder-Decoder`라는 2 개의 RNN이 encoder와 decoder part로 구성된 NN architecture를 소개
- Encoder: Map `variable-length source sequence` to `fixed-length vector`
- Decoder: Map `vector representation` to `variable-length target sequence`

위 두 network는 P(S_target|S_source)를 최대화시키며 **동시에** 학습됨

또한 본 논문에서 `memory capacity`와 `ease of training`를 향상시키기 위해 **정교한** `hidden unit`을 사용할 것을 제안

### scoring approach to translation

translate English -> French

모델이 English phrase에 대응되는 French phrase의 확률을 학습하도록 훈련

위 모델을 standard phrase-based SMT system으로 사용 가능
- `phrase table`의 각 `phrase pair`에 점수를 매겨서!
- 뒤에서 나오지만, `(input_seq, output_seq)` pair의 `P(output_seq|input_seq)`를 사용!

`RNN Encoder-Decoder`을 사용해 phrase pair에 점수를 매기는 방식은 번역 성능을 상승시키는 것으로 밝혀짐

### Qualitative Analysis and indirectly Quantitative Analysis

기존의 번역 모델과 본 논문의 학습된 RNN Encoder-Decoder 모델의 phrase score를 를 비교, 정석적인 분석을 실시

정성적인 분석은 `Encoder-Decoder`가 phrase table의 `linguistic regularities`를 잘 포착하고

간접적으로 번역 성능의 정량적인 향상을 설명 가능케 함

또한 RNN Encoder-Decoder는 semantic, syntactic한 phrase의 구조의 `continuous apce representation`를 학습한다는 것을 밝힘

참고: [Linguistic Regularities in Continuous Space Word Representations](https://www.aclweb.org/anthology/N13-1090/)

## 2. RNN Encoder-Decoder

### 2.1 Preliminary: Recurrent Neural Network
RNN은 아래 <img src="https://latex.codecogs.com/svg.latex?\Large&space;h,\;y,\;x"/>로 이루어진 neural network
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;h"/>: hidden state
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;y"/>: optional output
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;x=(x_1,x_2,\dots,x_T)"/>: variable-lenght sequence

<img src="https://latex.codecogs.com/svg.latex?\Large&space;At\;each\;time\;step\;t,{\;\;}h_{<t>}\;is\;updated\;by"/>

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_{<t>}=f(h_{<t-1>},x_t)\cdots(1)"/>
- where <img src="https://latex.codecogs.com/svg.latex?\Large&space;f"/>: non-linear activation function
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;f"/>는 `logistic sigmoid function`혹은 `LSTM`이 될 수 있음

RNN은 다음 symbol을 예측하게 학습시킴으로써 전체 sequence의 확률 분포를 배운다.

이 경우, 각 timestep <img src="https://latex.codecogs.com/svg.latex?\Large&space;t"/>에서의 output은 아래와 같은 조건부 확률이다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(x_t|x_{t-1},\dots,x_1)">

예를 들어 `multinomial distrivution`(1-of-K coding)은 `softmax` 함수를 사용해 출력될 수 있다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(x_{t,j}=1|x_{t-1},\dots,x_1)=\cfrac{exp({w_j}h_{<t>})}{\sum_{j^{'}=1}^{K}exp(w_{j^{'}}h_{<t>})}\cdots(2)"/>
- j=[1,K]: possible symbols
- w_j: rows of a Weight Matrix W

위의 확률을 결합하여 sequence x의 확률을 계산할 수 있다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(x)=\prod_{t=1}^{T}p(x_t|x_{t-1},\dots,x_1)\cdots(3)"/>

위와 같이 분포를 학습하여 각 time step별 symbol을 반복적으로 추출하여 새로운 sequence를 추출하는 것은 간단하다.

### 2.2 RNN Encoder-Decoder
확률적인 관점에서, 새로운 모델은 sequence가 주어졌을 때 sequence가 등장할 조건부 확률을 학습하는 일반적인 방법이다.
(Seq2Seq!!)

즉, <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(y_1,\dots,,y_{T'}|x_1,\dots,x_{T'}">
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;T">와 <img src="https://latex.codecogs.com/svg.latex?\Large&space;T^'">는 다를 수 있다.

#### Encoder
Encoder는 `RNN`으로 input sequence x의 각 symbol을 sequentially하게 읽는다

각 symbol을 불러올 때마다 RNN의 hidden state h은 Eq(1)과 같이 update된다.

sequence를 전부 읽은 후에 RNN의 hidden state는 전체 input sequence의 요약 c가 된다.

#### Decoder
제안된 모델의 Decoder는 hidden state h_<t>가 주여졌을 때 다음 symbol y_t를 예측하여 output sequence를 _생성_하는 `RNN`이다

그러나 앞서 설명한 `RNN`과 다르게 y_t와 h_<t>는 y_{t-1}과 input sequence의 요약 c에 영향을 받는다.
(Figure 1.을 참고하라.)

때문에 각 time t에서 decoer의 hidden state는 아래와 같이 update된다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_{<t>}=f(h_{<t-1>},y_{t-1},c)">

유사하게 다음 symbol의 조건부 확률을 아래와 같이 계산한다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(y_t|y_{t-1},y_{t-2},\dots,y_1,c)=g(h_{<t>},y_{t-1},c)">

위 Encoder-Decoder의 구조는 아래와 같다.

![encoderdecoder1](https://user-images.githubusercontent.com/37775784/77247746-430c2980-6c77-11ea-8779-8481c64fa2c1.PNG)


#### Objective
제안된 `RNN Encoder-Decoder`의 두 RNN은 아래 조건부 로그우도함수를 최대화시키기 위해 동시에 학습된다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;\max_{\theta}\cfrac{1}{N}\sum_{n=1}^{N}\log{p\theta}(y_n|x_n)\cdots(4)">
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;\theta">: set of the model parameters
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;(x_n,y_n)">: (input seq, output seq) pair

또 앞선 계산들은 input부터 decoder의 output까지 모두 미분가능(differentiable)하기 때문에 model parameter를 추정하는데 gradient-based algorithm을 사용한다.

#### Usage
`RNN Encoder-Decoder`를 학습시킨 후에 모델은 아래 두 가지 방법으로 사용 가능하다.
- input sequence가 주어졌을 때, 모델을 사용하여 target sequence를 생성한다.
- Eq(3)과 (4)의 확률값 <img src="https://latex.codecogs.com/svg.latex?\Large&space;p\theta(y|x)">로 input과 output sequence pair에 _score_를 매긴다.


## 2.3 Hidden Unit that adaptively Remembers and Forgets
본 논문에서 LSTM과 비슷한, 그러나 더 쉽게 계산 및 구현이 가능한 새로운 현태의 `hidden unit` (f in Eq(1))을 제안하고자 한다.

아래 그림과 같은 `hidden unit`이다.

![encoderdecoder2](https://user-images.githubusercontent.com/37775784/77247781-a0a07600-6c77-11ea-9fc0-2a53b052ea68.PNG)

### reset gate

`reset gate` <img src="https://latex.codecogs.com/svg.latex?\Large&space;r_j">는 아래의 식으로 계산된다
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;r_j=\sigma\big([W_r{x}]_j+[{U_r}h_{t-1}]_j\big)">

  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma">: `logistic signoid function`
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;[.]_j">: vector의 j번째 요소
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;x">: input
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_{t-1}">: previous hidden state

### update gate
`update gate` <img src="https://latex.codecogs.com/svg.latex?\Large&space;z_j">도 reset gate와 유사하게 계산된다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;z_j=\sigma\big({[W_z{x}{]}_j+{[{U_z}h_{t-1}]}_j}\big)">

### hidden unit
제안된 hidden unit <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_j">의 실제 activation은 아래와같이 계산된다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_j^{<t>}=z_j{h_j^{<t-1>}}+(1-z_j)\tilde{h}_j^{<t>}">

  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;where\;\tilde{h}_j^{<t>}=\phi\big({[Wx]}_j+{[U(r\odot{h_{t-1}})]}_j\big)">

### Explane formulation

#### reset gate
`reget gate`가 0에 가까워지면 hidden state는 이전 상태를 무시하고 입력값으로 reset한다.

이는 hidden state가 이 후 symbol과 관련이 없는 정보를 _drop_ 하게 만들고 더욱 `compact`한 표현을 만들게 된다.

#### update gate
반대로 `update gate`는 이전 hidden state를 현재 hidden state에 얼마나 반영할지를 결정한다.

이는 LSTM의 memory cell과 유사하며 RNN이 장기 정보를 기억하도록 만든다.

게다가 이는 `leaky-integration unit`의 변형된 형태로 고려할 수도 있다.

### Long-Short Dependencies related on reset and update gate
각 hidden unit이 분리된 `reset gate`와 `update gate`를 가지고 있기 때문에 각 hidden unit은 다른 [time scale](https://www.merriam-webster.com/dictionary/timescale)의 dependencies을 모델이 이해하도록 학습한다.

이렇게 **short dependencies** 을 배운 unit은 `reset gate`가 활성화되는 경향을 보이며 반대로 **Longer dependencies** 를 포착하는 unit은 주로 `update gate`가 활성화된다.

실험적으로 저자는 위의 새로운 unit을 사용하는 것이 중요하다는 사실을 발견했고

흔히 자주 사용하는 `tanh`로 유의미한 결과를 얻지 못했다고 기술한다.

## 3. Statistical Machine Translation
SMT의 목적은 source sentence e가 주어졌을 때 translation f를 찾는 것이며 아래 확률을 maximize해야 한다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(f|e)\proptop(e|f)p(f)">

  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(e|f)">: _translation model_
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(f)">: _language model_

그러나 대부분의 SMT system들은 feature를 추가하여 log linear model <img src="https://latex.codecogs.com/svg.latex?\Large&space;\log{p(f|e)}">을 모델링한다.
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;\log{p(f|e)}=\sum_{n=1}^{N}w_n{f_n}(f,e)+\log{Z(e)}\cdots(9)">

  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;f_n">: n-th feature
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;w_n">: weight
  - <img src="https://latex.codecogs.com/svg.latex?\Large&space;Z(e)">: weight와 독립적인 normalization constant
  - weight는 종종 development set에서 `BLEU score`를 최대화시키게 최적화된다.

(Koehn et al., 2003)과 (Marcu and wong, 2002)에서 소개된 phrase-based SMT framework에서 _translation model_ <img src="https://latex.codecogs.com/svg.latex?\Large&space;\log{p(e|f)}">는 source sentence와 target sentence가 매칭될 translation 확률로 factorizing된다.

WLOG(Without loss of generality), 이 아래부터 각 구문 쌍에 대해서도 p(e|f)를 translation model로 언급한다.

이 확률은 log-linear model(Eq(9))에서 추가된 feature로 여겨지고 BLEU score를 최대화시키게 조절된다.

(Bengio et al., 2003)에서 NNLM이 제안된 이래로 NN은 SMT system에서 광범위하게 사용돼왔다.

많은 경우에서 NN은 `rescore translation hypotheses`(n-best lists)를 사용하곤 한다.
(Schwenk et al., 2006을 참고하라)

그러나 최근에는 source sentence를 추가하여(on-line) 이 문장의 표현을 사용하여 번역된 문장을 NN으로 scoring하는 것이 주 관심하이다.

### 3.1 Scoring Phrase Pairs with RNN Encoder-Decoder
본 논문에서 RNN Encoder-Decoder를 phrase pairs의 table에서 학습시키고 이를 SMT decoder를 튜닝할 때 log-linear model의 추가 feature로 사용할 것을 제안했다.

RNN Encoder-Decoder를 학습시킬 때 원 말뭉치의 각 phrase pair의 빈도를 무시한다.

(1) 이는 정규화된 빈도수에 따라 large phrase table에서 phrase pair를 랜덤하게 선택하는 연산 비용을 줄이고

(2) RNN Encoder-Deoder가 단순히 발생 빈도에 따라 phrase pairs의 rank를 학습시키지 않다는 것을 확실히 하기 위해 해당 measure를 차용했다.

이러한 선택을 하게된 이유는 이미 phrase table에 있는 번역 확률은 original corpus의 phrase pairs의 빈도를 반영하고 있기 때문이다.

RNN Encoder-Decoder의 고정된 용량(capacity)로 우리는 대부분의 model capacity가 언어 규칙성을 학습하도록 집중한다.

`Linguistic Regularities`
- 그럴 듯한(plausible) 혹은 인정하기 어려운(implausible) 번역의 모호함
- 그럴 듯한 번역의 `manifold`(region of probability concentration)를 학습하는 것

(중략)

### 3.2 Related Approaches: Neural Networks in Machine Translation
실험결과를 소개하기 전에 SMT에 NN을 적용시킨 몇몇 작업에 대해 의논해보기로 한다.

(중략)

## 4. Experiments
(중략)

### 4.1 Data and Baseline System
(중략)

#### 4.1.1 RNN Encoder-Decoder
encoder와 decoder에서 제안된 gate의 1,000개의 hidden units을 가짐

각 input symbol x_<t>와 hidden unit의 input matrix는 two lower-rank matrices로 근사된다.
(output matrix도 똑같음)

rank-100 matrix를 사용했고 각 단어별 100D로 embedding

(중략)
