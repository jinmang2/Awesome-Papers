# A Neural Probabilistic Language Model

### Abstract
- 통계 언어 모델링의 목적은 sequences of words의 결합 확률 함수를 학습하는 것
- 이는 curse of dimensionality(차원의 저주)때문에 매우 어렵고 이 논문은 이를 타파하기 위한 아키텍처를 제시
- 각 단어에 대한 분산 표현(의미 유사도)와 문장의 결합 확률 함수를 동시에(simultaneously) 학습
- 이미 학습한 문장의 단어와 유사한 sequence of words가 들어올 경우, 이에 대한 확률이 높기 때문에 일반화될 수 있다
- SOTA trigram model 업그레이드 시켰다. 짱짱맨.
- 2003년에 나온 Bengio교수의 논문

### Introduction
- 언어 모델에서 학습을 방해하는 주요 요인은 `curse of dimensionality`
- Vocab size가 100,000일 경우 10^50-1개의 free parameters가 생김(ㅎㄷㄷ..)

#### 기존 통계 모델의 문제점
- 통계적 언어 모델은 이전 문장이 주어질 경우 다음 단어가 나올 조건부 확률로 쓸 수 있음

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;P(w_1^T)=\prod_{t=1}^{T}{P(w_t|w_1^{t-1})}"/>

    where <img src="https://latex.codecogs.com/svg.latex?\Large&space;w_t"/> is the t-th word, and writing subsequence <img src="https://latex.codecogs.com/svg.latex?\Large&space;w_i^j=(w_i,\;w_{i+1},\cdots,w_{j-1},\;w_j)"/>

- 자연어의 통계 모델을 구축 할 때, 단어 순서의 이점 및 단어 순서에서 시간적으로 더 가까운 단어가 통계적으로 더 의존적
- So, `n-gram` 모델로 다음 단어의 조건부 확률 테이블을 구축. i.e. <img src="https://latex.codecogs.com/svg.latex?\Large&space;P(w_t|w_1^{t-1})\;{\approx}\;P(w_t|w_{t-n+1}^{t-1})"/>
- 이 때, 학습 코퍼스에 존재하지 않은 n-gram이 포함된 문장이 나타날 확률 값을 0으로 분류.
- 이를 [back-off trigram model](https://pdfs.semanticscholar.org/969a/9ec5f24dabcfb9c70c7ee04625075a6c0a98.pdf), [smooothed (or interpolated) trigram model](https://www.semanticscholar.org/paper/Interpolated-estimation-of-Markov-source-parameters-Jelinek/6a923c9f89ed53b6e835b3807c0c1bd8d532687b#citing-papers)으로 보완할 수도 있지만 완전하지 않음
- 위 해당 모델은 두 가지의 결점(flaw)가 존재
    - 1~2 단어 이상의 문맥을 고려하지 않는다.
    - 단어 간의 유사성을 계산할 수 없다.
- `The cat is walking in the bedroom`
- `A dog was running in a room` generalize!! (위의 통계 모델은 이 유사성을 계산하지 못함.)

#### Fighting the Curse of Dimensionality with its Own Weapons
- 제안할 모델의 아이디어는
    1. 단어 사전의 단어들을 `feature vector`(a real-valued vector in Real^m)로 분산표현하여 단어들 사이의 유사성을 만든다.
    2. 이러한 word sequence의 feature vector를 결합 확률 함수(`joint probability function`)으로 표현
    3. 위 함수의 parameter와 word feature vector를 동시에 학습
- number of features는 단어 사전의 size보다 작게 설정(실험적으로 m=30, 60, 100으로 설정)
- `probability function`은 이전 단어가 주어졌을 때 다음 단어가 나올 조건부 확률의 곱으로 표현 (multi-layer neural network를 실험으로 사용)
- 이 함수에는 가중치 감소 페널티를 추가하여 학습 데이터 또는 정규화된 criterion의 log-likelihood를 최대화하기 위해 반복적으로 조정할 수있는 매개 변수가 존재
- 각 feature vector는 사전 지식으로 초기화할 수 있음

#### Relation to Previous Work (2003)
- NN으로 고차원 이진 분산 표현을 실시하는 아이디어는 이미 [여기서](https://papers.nips.cc/paper/1679-modeling-high-dimensional-discrete-data-with-multi-layer-neural-networks.pdf) 소개
- [Learning distributed representations of concepts](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.408.7684&rep=rep1&type=pdf)
- [Hinton교수의 연구가 성공적으로 적용된 사례가 있음]()
- [Neural Network를 language model에 적용시키려한게 완전 새로운 것은 아님](https://www.sciencedirect.com/science/article/abs/pii/036402139180002M)
- 위의 논문과는 대조적으로, 본 연구에서는 [문장에서 단어의 역할을 배우기보다는 단어 시퀀스 분포의 통계 모델을 배우는 데 집중함](https://www.semanticscholar.org/paper/Sequential-neural-text-compression-Schmidhuber-Heil/79521a6d8814f9162ed1f7028e9e007c4df7181a)
- 제안한 모델에서는 discrete random or deterministic variable의 유사성을 특징화시키지 않고 `distributed feature vector`, `continuous real-vector for each word`를 사용한다.
- 이러한 단어의 vector-space 분산 표현 방식은  [`Information Retrieval`](https://wordrepr.danieldk.eu/schuetze-1993.pdf)분야에서도 많이 사용한다.
- **An important difference is that here we look for a representation for words that is helpful in representing compactly the probability distribution of word sequences from natural language text.**
    - 중요한 차이점은 자연어 텍스트에서 word sequence의 확률 분포를 간단히 표현하는 데 도움이되는 단어의 표현을 찾는 것.
- 이 Embedding이 모델의 성능을 크게 다르게 만들더라!!

### Proposed Model: two Architectures

#### Definition and Proposition
- `Training set`: a sequence <img src="https://latex.codecogs.com/svg.latex?\Large&space;w_1{\cdots}w_T"/> of words <img src="https://latex.codecogs.com/svg.latex?\Large&space;w_t{\in}V"/>, where the vocabulary <img src="https://latex.codecogs.com/svg.latex?\Large&space;V"/> is a large but finite set.
- `Objective`: To learn a good model <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(w_t,{\cdots},w_{t-n})=\hat{P}(w_t|w_1^{t-1})"/>
- 실험에서 `perplexity`로 알려진 <img src="https://latex.codecogs.com/svg.latex?\Large&space;1/\hat{P}(w_t|w_1^{t-1})"/>의 geometric average를 report.
    - 이는 exponential of the average negative log-likelihood로 계산
- `Model Constraint`:

    for any choice of <img src="https://latex.codecogs.com/svg.latex?\Large&space;w_1^{t-1}"/>,

   <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sum_{i=1}^{|V|}f(i,w_{t-1},w_{t-n})=1"/>

- 조건부 확률의 곱으로, 단어의 임의 sequence들의 결합 확률를 얻을 수 있음.

#### Model Describe
- 논문에서 <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(w_t,{\cdots},w_{t-n})=\hat{P}(w_t|w_1^{t-1})"/> 함수의 분해를 아래 두 파트로 나눠서 진행

    1. A mapping <img src="https://latex.codecogs.com/svg.latex?\Large&space;C"/> from any element of <img src="https://latex.codecogs.com/svg.latex?\Large&space;V"/> to a real vector <img src="https://latex.codecogs.com/svg.latex?\Large&space;C(i){\in}\mathbb{R}^m"/>
        - vocabulary의 각 단어를 `distributed feature vector`로 매핑
        - <img src="https://latex.codecogs.com/svg.latex?\Large&space;C"/>는 <img src="https://latex.codecogs.com/svg.latex?\Large&space;|V|{\times}m"/>(of free parameters) 행렬로 표현됨

    2. <img src="https://latex.codecogs.com/svg.latex?\Large&space;C"/>: The probability function over word.
        - `The direct architecture`
            - a function `g` maps a sequence of feature vectors for words in context <img src="https://latex.codecogs.com/svg.latex?\Large&space;(C(w_{t-n}),{\cdots},C(w_{t-1}))"/> to a probability distribution over words in <img src="https://latex.codecogs.com/svg.latex?\Large&space;V"/>.
            - 아래 objective function으로 <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{P}(w_t=i|w_1^{t-1})"/>을 추정
            - <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(i,w_{t-1},{\cdots},w_{t-n})=g(i,C(w_{t-1}),{\cdots},C(w_{t-n}))"/>
            - output layer에서 확률값을 얻기 위해 `softmax`를 사용하기 때문에 <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_i"/>가 i번째 단어의 output 확률일 때

                <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{P}(w_t=i|w_1^{t-1})=\cfrac{e^{h_i}}{\sum_j{e^{h_j}}}"/>
        - `The cycling architecture`
            - a function `h` maps a sequence of feature vectors <img src="https://latex.codecogs.com/svg.latex?\Large&space;(C(w_{t-n}),{\cdots},C(w_{t-1}),C(i))"/> to a scalar <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_i"/>
            - again using a softmax,

                <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{P}(w_t=i|w_1^{t-1})=\cfrac{e^{h_i}}{\sum_j{e^{h_j}}}"/>
            - <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(w_t,w_{t-1},{\cdots},w_{t-n})=g(C(w_t),C(w_{t-1}),{\cdots},C(w_{t-n}))"/>
            - `h`(e.g. a neural net)을 반복적으로 동작시키면서 각 시점마다 다음 후보 단어 i를 위해 feature vector <img src="https://latex.codecogs.com/svg.latex?\Large&space;C(i)"/>를 넣어주기 때문에 `cycling`이라고 명명

- `f`는 `C`와 `g`의 결합 함수
- `C`는 context의 모든 단어들이 공유
- 위 두 파트는 몇몇 파라미터를 공유
- `g`는 feed-forward or recurrent neural network 혹은 다른 모수 함수로 구현할 수 있다.

<img src="https://i.imgur.com/vN66N2D.png" width="50%" height="50%">

Figure 1: "Direct Architecture"

출처: [ratsgo's blog](https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/29/NNLM/)

### Speeding-up and other Tricks
pass

### Experimental Results
pass

### Conclusion
pass
