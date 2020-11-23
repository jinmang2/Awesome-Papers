# Neural Machine Translation By Jointly Learning to Align and Translate

## Abstract
SMT와 다르게 NMT는 번역 성능을 최대화하도록 튜닝되는 single NN을 만드는데 집중한다
NMT를 위해 제안된 모델을 Encoder-Decoder의 형태를 가지며 Encoder는 sourece sentence를 fixed-length vector로 부호화하며 Decoder는 이로부터 번역을 생성한다.
본 논문에선 이 fixed-length vector를 사용하는 것이 기본 encoder-decoder 구조의 모델 성능에 병목(bottelneck) 현상을 유발하는 것을 어림짐작하고  source sentence로 부터 target word와 연관이 있는 부분을 자동으로 soft-search하는 모델을 제안한다.
해당 접근법으로 English-French 번역 성능을 기존 SOTA 대비 향상시켰고 qualitative analysis로 모델로부터 얻어진 soft-alignment가 우리의 직관과 맞아 떨어졌다는 것을 보여준다.

## 1. Introduction

NMT는 ~, ~, ~에서 제안된 새로운 접근 방법. 기존의 phrase-based 번역과는 다르게(수많은 세부항목으로 나누어 번역) NMT는 input sentence에 적합한 번역을 도출하는 것을 그 목적으로 함
대부분의 NMT는 encoder-decoder의 양상을 띔. Encoder는 입력 문장을 읽고 이를 고정된 길이의 vector로 부호화. Decoder는 부호화된 벡터로부터 번역을 도출. 전체 Encoder-Decoder 시스템은 주어진 문장에 맞는 번역을 내뱉을 확률을 최대화시키며 학습된다.
encoder-decoder 접근법의 잠재적 이슈는 NN이 주어진 문장을 고정된 길이의 벡터로 바꿀 때 필요한 정보를 **압축** 해야할 필요가 있다는 것이다. 이는 NN이 학습 말뭉치의 문장보다 긴 문장에 대처하기 어렵다는 것을 의미한다. Cho는 입력 문장의 길이가 길수록 encoder-decoder의 성능이 악화되는 것을 보였다.
encoder-decoder 접근법의 가장 큰 문제는 전체 입력 문당을 단 하나의 고정된 길이의 벡터로 부호화하려고하지 않는 다는 점이다. 위와 같이 하는 대신, **입력 문장을 벡터의 수열로 부호화하고 번역으로 복호화할 때 적절히 이들 벡터 중의 부분집합을 고르는 방법을 제안한다.** 이는 NMT모델이 문장 길이에 관계없이 입력 문장을 고정된 길이 벡터로 변환하며 정보를 낭비하는 일을 없게 만들어준다. 우리는 이러한 접근 방식이 긴 문장에 대해 더 잘 대처한다는 것을 보였다.
본 논문에서 (1) **정렬(align)과 번역(translate)** 을 동시에 학습하는 법을 제안하고 (2) 이 방법이 기본 encoder-decoder 접근법의 성능을 크게 향상시킴을 보인다. 이러한 성능 향상은 대체로 모든 길이의 문장에 대해 이루어졌으며 특히 긴 문장에 대해서는 명백히 드러났다. English-French 번역 문제에서 단일 모델로 기존 phrase-based system의 성능에 필적하거나 더 우수한 성능을 달성한다. 게다가 제안도니 모델은 언어적으로 그럴듯한 source sentence와 target sentence의 정렬을 찾아낸다.

## 2. Background: Neural Machine Translation
NMT는 $\argmax_{y}{p(y|x)}$를 최대화시키며 학습, parameterized model
RNN Encoder-Decoder 구조의 모델 제안
NMT는 괄목할 성적을 거둠!

### 2.1 RNN Encoder-Decoder

본 논문에서 제안할 모델: novel architecture that learns to align and translate simultaneously.

위 모델의 기반이 되는 모델이 RNN Encoder-Decoder

간단히 수식을 작성하면,
$x=(x_1,\cdots,x_{T_{x}})$: sequence of vectors
$c$: context vector

가장 흔한 RNN 접근 방식은 아래와 같음

Encoder
$h_t=f(x_t,h_{t-1})\cdots(1)$
$c=q(\{h1,\cdots,h_{T_{x}}\})$
$\text{where }h_t\in\mathbb{R}^n\text{ : hidden state at time }t$
$f,q$는 LSTM, GRU, sigmoid와 같은 non-linear function

Decoder
다음 단어 $y_{t^\prime}$은 주어진 context vector $c$와 이전에 예측된 모든 단어 $\{y_1,\cdots,y_{t^\prime-1}\}$의 영향을 받는다. 즉,
$p(y)=\prod_{t=1}^{T}p(y_t|\{y_1,\cdots,y_{t-1}\},c)\cdots(2)$
$\text{where }y=(y_1,\cdots,y_{T_y})$

최종적으로 RNN으로 각 조건부 확률은 아래와 같이 모델링됨
$p(y_t|\{y_1,\cdots,y_{t-1}\},c)=g(y_{t-1},s_t,c)\cdots(3)$
**hybrid RNN 혹은 de-convolutional neural network도 사용될 수 있음을 기억하라.**

## 3. Learning to Align and Translate

자, 새로운 architecture는 (1) ENCODER: bidirectional RNN (2) DECODER: 번역으로 복호화하는 중에 source sentence를 통해 검색하는 emulator로 이루어져 있다!

### 3.1 DECODER: General Description
새로운 모델 구조에서 우리는 Eq(2)의 각 조건부 확률을 아래와 같이 재정의한다.

$$p(y_i|y_1,\dots,y_{i-1},x)=g(y_{i-1},s_i,c_i)\cdots(4)$$
$\text{where }s_i\text{ is an RNN hidden state for time i, computed by}$
$$s_i=f(s_{i-1},y_{i-1},c_i)$$
**기억하라** 기존의 접근과는 다르게 각 조건부 확률은 각 target word $y_i$에 대해 독립적인 context vector $c_i$가 주어진다.

context vector $c_i$는 encoder가 입력된 문장을 mapping하는데 사용한 _annotations_ $(h_1,\cdots,h_{T_{x}})$의 수열에 종속된다. 각 annotation $h_t$는 입력 문장의 $i$번째 단어 주변에 강하게 집중하고 있는 전체 입력 문장에 대한 정보를 포함한다. 이게 무슨말이냐, 전체 입력 문장에 대한 정보를 가지고 있지만 해당 $i$번째 주변의 정보를 특히나 더 강조한다는 의미이다.

context vector $c_i$는 각각 annotations들 $h_i$의 가중합으로 계산된다.
$$c_i=\sum_{j=1}^{T_x}\alpha_{ij} h_j\cdots(5)$$
각 annotation $h_j$의 가중치 $\alpha_{ij}$는 softmax로 구해진다.
$$\alpha_{ij}=\cfrac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})}\quad\text{where }e_{ij}=a(s_{i-1},h_j)\cdots(6)$$
$a$는 position $j$에서의 입력값과 position $j$에서의 출력값이 얼마나 잘 들어맞는지 점수로 만든 _alignment_ 모델이다. 해당 점수는 RNN hidden state $s_{i-1}$과 입력 문장의 $j$번째 annotation $h_j$에 의해 얻어진다.

alignment model $a$는 제안된 구조의 모든 요소에 대해 동시에 feedforward neural network를 학습시켜 모수화한다. **기존의 Machine Translation과는 다르게 alignment는 잠재 변수를 고려하지 않는다.** 대신에 alignment model은 cost function의 gradient를 역전파할 수 있는 soft alignment를 직접 계산한다. 이 gradient는 정렬 모델과 전체 변환 모델을 공동으로 훈련시키는 데 사용될 수 있다.

각 annotation들의 가중합을 구하는 것은 가능한 정렬들의 기댓값 _expected annotation_ 을 계산하는 것으로 이해할 수 있다. $\alpha_{ij}$를 source word $x_j$에서 정렬되거나 번역된 target word $y_j$에 대한 확률이라 하자. 그러면 $i$번째 context vector $c_i$는 모든 annotation 확률 $a_{ij}$의 expected annotation이다.

확률 $a_{ij}$혹은 이와 관련된 energy $e_{ij}$는 다음 상태 $s_i$를 결정하고 $y_i$를 생성하는 이전 hidden state $s_{i-1}$에 대한 annotation(주석) $h_j$의 중요성을 반영한다. 직관적으로 이는 decoder에서의 **Attention Mechanism** 을 암시한다. Decoder는 주어진 문장에서 어디에 집중을 해야하는지를 결정한다. Decoder가 Attention Mechanism을 갖게 함으로써 source sentence의 모든 정보를 고정된 길이의 벡터로 부호화시키는 부담에서 벗어날 수 있다. 위 새로운 접근 방식으로 입력 문장에 대한 정보는 annotation sequence에 걸쳐 전파되며 decoder가 이를 선택적으로 검색하여 정보를 탐색한다.

**내 이해**: annotation은 Encoder의 hidden state이다!

### 3.2 ENCODER: Bidirectional RNN for annotating Sequences

Eq(1)에서 사용된 RNN은 입력 문장 $x$를 첫 symbol $x_1$에서 마지막 symbol $x_T$ 순서로 읽는다. 그러나 제안된 scheme에서 우리는 각 단어의 annotation이 preceding words뿐만 아니라 following words까지 요약하길 바란다. 고로 본 논문에선 speech recognition에서 성공적인 성적을 거둔 Bidirectional RNN (BiRNN, Schuster and Paliwal, 1997)을 사용할 것을 제안한다.

BiRNN은 forward와 backward RNN으로 구성되어 있다. forward RNN은 $\overrightarrow{f}$와 같이 쓰며 입력 문장을 정방향으로 읽고 ($x_1\;to\;x_{T_{x}}$) forward hidden states ($\overrightarrow{h}_1,\cdots,\overrightarrow{h}_{T_{x}}$)의 수열을 계산한다. backward RNN는  $\overleftarrow{f}$와 같이 쓰며 역방향으로 입력 문장을 읽으며($x_{T_{x}}\;to\;x_1$) backward hidden states ($\overleftarrow{h}_1,\cdots,\overleftarrow{h}_{T_{x}}$)의 수열을 계산한다.

우리는 위에서 forward hidden state $\overrightarrow{h}_j$와 backward hidden state $\overleftarrow{h}_j$를 연결하여(concatenating) 각 단어 $x_j$에 대한 annotation(주석)을 얻는다. 즉,
$$h_j=\big[\overrightarrow{h}_j^\top;\overleftarrow{h}_j^\top\big]^\top$$
위 annotation $h_j$는 preceding words와 following words 둘의 요약을 가지고 있다. RNN의 최근 입력에 대해서 더 잘 표현하려는 경향 때문에 annotation $h_j$는 $x_j$ 주변의 단어에 집중하게 된다. annotation의 수열은 디코더에 의해 사용되며 alignment model은 차후 context vector를 계산한다.

## 4. Experiment Settings
생략

### 4.1 Dataset
생략

### 4.2 Models
두 type의 모델을 학습
- ENN Encoder-Decoder(RNNencdec)
- Proposed Model(RNNsearch)

각 모델을 30 words / 50 words 데이터에 대해 각 2번 학습

RNNencdec의 Encoder와 Decoder는 1,000개의 hidden unit을 각각 가짐
RNNsearch의 encoder는 각 1,000개의 hidden unit을 가지는 forward-backward RNN으로 구성, decoder 또한 1,000개의 hidden unit을 가짐
두 case에서 각 target 단어의 조건부 확률을 계산하기 위해 single maxout hidden layer를 가진 multilayer network를 사용
각 모델을 학습시키기 위해 Adadelta와 minibatch Stochastic Gradient Descent 알고리즘을 사용
각 SGD는 방향별로 80개 문장의 minibatch를 계산에 사용
대략 5일동안 각각의 모델을 학습
모델이 학습되면 조건부 확률을 최대화하는 번역을 찾기 위해 beam search를 사용
Sutskeverㅇ느 이 접근을 그의 NMT 모델의 번역을 생성시키기 위해 사용했다.

## 5. Results

### 5.1 Quantitative Results
생략

### 5.2 Qualitative Analysis

#### 5.2.1 Alignment
**차후 제대로 읽고 다시 번역할 것**

제안된 접근방식은 주어진 문장과 생성된 번역의 각 단어별 soft-alignment를 파악하는 직관적인 방식을 제공한다. 이는 시각화 가능하다. - 그림 첨부 -. 각 행렬의 행은 annotation과 관련된 가중치를 가리킨다. 이는 target 단어를 생성할 때 source sentence의 각 위치가 중요하다는 것을 상기시킨다.

우리는 그림에서 English와 French간의 단어가 아주 monotonic하다는 것을 확인할 수 있다. diagonal 확인
그러나 non-trivial한, non-monotonic alignments한 것들도 확인할 수 있었다. **형용사와 명사는 주로 다르게 정렬된다.** RNNSearch는 [zone]과 [Area]를 [European], [Economic]을 건너띄고 매칭시킬 수 있다.

hard-alignment와 대비되는 soft-alignment의 강점을 아주 명백하다! 뭐 블라블라

#### 5.2.2 Long Sentences
긴 것도 잘한다더라 훌륭하다

## 6. Related Work

### 6.1 Learning to Align

뭐 많다네 다음데 또 보자 지금 할거 많어

### 6.2 Neural Networks for Machine Translation

벤지오 짱짱맨 NPLM에서 진척과정 설명

## 7. Conclusion

#### RNNencdec의 문제점
NMT의 conventional 접근법 RNN Encoder-Decoder는 입력 문장을 고정된 길이의 벡터로 부호화하는데 이는 긴 문장을 번역하는데 문제가 존재한다.

#### RNNsearch 제안
본 논문에서 이 issue를 해결할 novel architecture를 제안, 기본 RNNencdec을 확장시켜 target 단어를 생성할 때 입력 단어 혹은 인코더에 의해 계산된 annotations(주석)에서 검색하여 찾는 모델을 설정한다. 이는 주어진 문장의 모든 정보를 고정된 길이의 벡터로 부호화해야하는 부담을 덜어주며 타겟 단어와 좀 더 연관있는 단어에 집중하도록 해준다. 이는 NMT가 긴 문장도 잘 번역할 수 있게 해준다. 기존의 기계 번역 시스템과 달리, 정렬 메커니즘을 포함한 번역 시스템의 모든 조각들은 정확한 번역본을 제작하는 더 나은 로그확률을 위해 공동으로 훈련된다.

#### 적용
좋았다

#### 우리가 NLU에 기여하는 바는 상당히 크다 짱짱맨

#### 남은 과제
unknown, rare words handling
