# Recurrent Highway Networks

Authors
- Julian Georg Zilly, (1)
- Rupesh Kumar Srivastava, (2)
- Jan Koutník, (2)
- Jürgen Schmidhuber, (2)

Institution
- (1): ETH Zürich, Switzerland
- (2): The Swiss AI Lab IDSIA (USI-SUPSI) & NNAISENSE, Switzerland

## Abstract
많은 sequential processing task는 step마다 복잡한 비선형 전이(transition) 함수를 요구로 합니다. 그러나 아주 많은, 그러니까 `"깊은"` transition 함수를 가진 RNN은 학습이 어렵습니다. LSTM이 대표적인 deep transition function이죠. 자, 이쯤에서 본론을 꺼내볼까요? 우리는 아래 내용을 소개하고자 합니다!
- `novel theoretical analysis of recurrent networks based on Geršgorin’s circle theorem`
- `Geršgorin’s circle theorem`은 modeling, optimization issue을 밝히고 LSTM cell에 대한 이해도를 증진시켜준 이론입니다.

이 분석에 기초하여 우리는 LSTM 아키텍처를 확장하여 하나 이상의 단계적 전환 범위를 허용하는 `Recurrent Highway Networks`를 제안합니다. 몇몇 언어 모델링 실험은 제안된 구조가 강력하고 효과적이라는 결론을 보여줍니다. Penn Treebank 말뭉치에서 같은 수의 모수를 사용하고 transition depth를 1에서 10으로 늘림에 따라 단어 단위 perplexity가  90.6에서 65.4로 증진시켰다. 문자 단위 예측(text8, enwik8)을 위한 대용량 Wikipedia 데이터셋에서 RHNs은 기존의 결과를 상회했고 문자당 1.27bit의 entropy를 거뒀다.

## 1. Introduction

#### Depth의 중요성

Schmidhuber (2015)의 논문에 따르면, 네트워크의 depth는 강력한 machine learning 패러다임으로써 neural networks의 재기에 제일 중요한 요소이다.
- [Learning complex, extended sequences using the principle of history compression, Neural Computation](https://dl.acm.org/doi/10.1162/neco.1992.4.2.234)

Bengio & Lecun (2007)와 Bianchini & Scarselli (2014)에 따르면, 심층 네트워크가 특정 `function classes`를 나타내는데 훨씬(`exponentially`) 효과적이라는 이론적 증거들이 있다고 말한다.
- [Scaling Learning Algorithms towards AI](http://yann.lecun.com/exdb/publis/pdf/bengio-lecun-07.pdf)
- [On the complexity of neural network classifiers: A comparison between shallow and deep architectures](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-44.pdf)

Recurrent Neural Networks는 그 순차적인 성질 때문에 CAP(Credit Assignment Path, 아래 problem이랑 다름!)를 가지고 time에 깊어진다. (umm.. 이해도가 낮아..)
- [The utility driven dynamic error propagation network](https://www.academia.edu/30351853/The_utility_driven_dynamic_error_propagation_network)
- [Generalization of backpropagation with application to a recurrent gas market model](https://www.researchgate.net/publication/223074905_Generalization_of_Backpropagation_with_Application_to_a_Recurrent_Gas_Market_Model)
- [Complexity of exact gradient computation algorithms for recurrent neural networks](https://scinapse.io/papers/121315764)

---
- Credit Assignment Problem (CAP): 기여도 할당 문제, 일견의 행동들이 모두 끝난 후에야 궤환신호를 얻을 수 있는 상황에서 그 때까지 수행된 일련의 행동들 중 어떤 행동에 기여도(credit)를 주고 또 어떤 행동에 벌점(blame)을 줄 것인지 결정
- 약간 강화 학습 느낌인데..?

---

그러나 계층 별 unit들의 군으로 구성된 현대 RNN의 특정 내부 함수의 매핑은 보통 **"깊이"** 에서 이점을 얻지 못한다. 예로, 다음 상태로의 업데이트는 보통 비선형성을 가미한 단일 학습가능한 linear transformation을 사용하여 모델링된다.
- [How to construct deep recurrent neural networks](https://arxiv.org/abs/1312.6026)
- 조경현 교수님, 벤지오 교수님이 여기에서 또...? 심지어 submission도 조교수님이 했네..?

#### Vanishing/Exploding Gradient Problem of RNN

불행히도, 네트워크를 깊게 만드는 것은 신경망의 모수들이 오차 역전파의 평균에 의해 최적화되는 것에서 문제가 생긴다.
- The representation of the cumulative rounding error of an algorithm as a taylor expansion of the local rounding errors
- [Applications of advances in nonlinear sensitivity analysis](https://link.springer.com/chapter/10.1007/BFb0006203)

기울기(gradient)의 크기가 역전파 도중 지수적으로 수훅되거나 증대될 수 있기 때문에 아래 논문들에 따르면 깊은 신경망은 흔히 `vanishing and exploding gradient problems`이라 불리는 문제로 고통받는다.
- [Untersuchungen zu dynamischen neuronalen Netzen](https://www.researchgate.net/publication/243781690_Untersuchungen_zu_dynamischen_neuronalen_Netzen)
- [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181)
- [Using the Output Embedding to Improve Language Models](https://www.aclweb.org/anthology/E17-2025/)

위와 같은 학습의 어려움은 표준 RNN에서 처음으로 연구됐다. 이 표준 RNN은 시간에 따른 깊이가 임의의 크기를 가지는 입력 sequence의 길이에 달렸다고 한다. (길수록 더 깊게?)

Hochreiter와 Schmidhuber의 LSTM은 recurrent network의 기울기 소실/폭주 문제를 해결하기 위해 도입됐다.
- [Long Short Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Learning to forget: Continual prediction with lstm](https://www.researchgate.net/publication/12292425_Learning_to_Forget_Continual_Prediction_with_LSTM)

#### Highway Network

기울기 소실 문제는 feedforward network를 매우 깊게 쌓고 훈련을 시킬 때 제한적인 요소로 작용하는데, LSTM cell에 기반을 둔 `Highway Layers`는 수백 층의 layer를 쌓아도 학습할 수 있게 만들어준다.
- [Highway networks](https://arxiv.org/abs/1505.00387)

feedforward 연결을 사용하여 아래 많은 domain에서 성능 향상을 거뒀다.
- speech recognition
  - [Highway long short-term memory RNNS for distant speech recognition](https://groups.csail.mit.edu/sls/publications/2016/YuZhang3_ICASSP-16.pdf)
- language modeling
  - [Character-aware neural language models](https://arxiv.org/abs/1508.06615)
  - [Exploring the limits of language modeling](https://arxiv.org/abs/1602.02410)

또한 이의 변종 `Residual Networks`은 computer vision의 많은 분야에서 효과적으로 사용됐다.
- [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)

#### RHNs
본 연구에선 LSTM cell에 대해 깊은 이해를 제공하는 RNN에 대한 수학적 분석을 우선 제공한다. 이러한 시각에 기반하여 `Recurrent Highway Networks` 줄여 RHNs를 소개하고자 한다. 이는 `long credit assignment paths not just in time but also in space (per time step)`을 가지는 LSTM networks이다.
- just in time: 동적과 비슷함, 적시생산시스템?
- per time step! 각각 time step마다?!
- long credit assignment path가 뭘까... 주의하면서 읽어보자!

deep RNNs의 이전 연구들과 다르게 depth를 늘리는 상위 방법론에 대해 논하고자 한다. 우리가 논할 "상위 방법론"은 강력함을 유지하며 sequential model을 효과적으로 학습할 수 있음은 물론, 이를 사용하면 기존의 benchmark 이상의 성능을 보이는 것이 가능하다.


## 2. Related Work on Deep Recurrent Transitions

#### Stacked RNNs
최근 몇 년간, RNN의 depth에 대해 계산적인 이점을 주는 흔한 util로 `stacking RNNs`를 들 수 있겠다. 이는 feedforward network에서 다중 hidden layer를 사용하는 것과 유사하다.
- [Learning complex, extended sequences using the principle of history compression](https://dl.acm.org/doi/10.1162/neco.1992.4.2.234)
- Schmidhuber 아재여... 인생 혼자 사시오? 1992년 논문

`stacked RNNs`를 학습시키는 것은 통상 공간, 시간적인 `credit assignment`를 요구하며 이를 실전에서 활용하기엔 어렵다.
- 아, 학습이 오래걸린단 얘기구나 credit assignment!
- cost에 관한 얘기였어
- 음... 이렇게 이해하면 안될 듯. 위의 초점은 space, time를 동시에 요구한다는 점인 거 같다 명훈아.

이러한 문제는 LSTM 기반의 stacking transformation 구조에서도 동일하게 관찰되고 있다.
- [Highway long short-term memory RNNS for distant speech recognition]()
- [Grid long short-term memory]()
- 위의 논문은 이미 적용한 논문..>? 문제에 대해 소개했나 보군
- 아래 논문은 grid,... ㅇㅎ... 둘다 2016, 2015논문

#### Micro time steps, real world data에선 성능 향상 X
step-to-step recurrent 상태 전이의 depth(즉, **recurrence depth**)를 증가시키는 흔한 방법으론 sequence의 step별 `micro time steps`를 위해 RNN tick을 허용하는 것이다...? 무슨 말이여
- Schmidhuber, 1991 [Reinforcement learning in markovian and non-markovian environments](https://papers.nips.cc/paper/393-reinforcement-learning-in-markovian-and-non-markovian-environments)
- Srivastaba et al., 2013 [First experiments with powerplay](https://arxiv.org/abs/1210.8385)
- Graves, 2016 [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)

이 방법으로 recurrence depth를 추가할 수 있지만, RNN은 이전 event의 memory를 위해 사용할 모수와 표준 깊은 비선형 과정을 스스로 학습해야 한다. Graves가 위의 논문에서 이 방법을 사용하여 단일 알고리즘 태스크에서 효과를 거뒀다고는 하지만 real world data에선 어떠한 성능 향상을 거두지 못했다.

#### DT-RNNs & Advanced Stacked-RNNs
Pascanu와 연구진들은 2013년에 recurrent transition에 다중 비선형 계층을 더하여 recurrence depth를 추가하는 방법론을 제시했는데, 이를 `Deep Transition RNNs (DT-RNNs)`와 `Deep Transition RNNs with Skip connections (DT(S)-RNNs)`로 그 결과를 정리했다.
- [How to construct deep recurrent neural networks](https://arxiv.org/abs/1312.6026)
- 오, Bengio교수, 조경현 교수님의 논문이자너?!

이론적으로 강력하지만 이 구조는 극심한 long CAP 문제가 발생하기 때문에 종종 기울기 역전파를 악화시킨다.
- 주석, CAP; 우리가 제안한 구조의 학습과의 비교를 subsection 5.1에 정리해뒀다.

Chung과 그 연구진들은 stacked RNN의 연속된 time step의 모든 state 사이에 별도의 connection을 더하여 recurrence depth를 늘리고자한 연구를 진행했다.
- [Gated feedback recurrent neural networks](https://arxiv.org/abs/1502.02367)
- 이 논문 또한 Bengio, 조경현 교수님 참여!

그러나 그들의 모델은 depth를 깊게하기 위해 굉장히 많은 extra connection을 요구하고, 가장 깊은 depth에 접근할 state의 일부만 제공하며 여전히 가장 긴 경로를 따라 gradient propagation issues에 직면한다.

#### Stacked vs Depths
recurrent layer를 쌓는 것과 비교하여 recurrence depth를 증가시키는 것은 RNN의 모델링 파워를 상당히 증가시킨다.

![rhns_1](https://user-images.githubusercontent.com/37775784/80911090-000d9d80-8d6f-11ea-80e3-2bfc1eefcd5f.PNG)

위 그림에 대해 설명해본다.
- stacking $d$ RNN layer는 $T$ time step을 가지는 hidden state 사이에서 $d+T-1$의 최대 CAP 길이를 가진다.
- $d$의 recurrence depth는 최대 경로 길이가 $d\times T$가 되도록 한다.
- 길수록 좋은 건가 보네...? 위에서 내가 이해한 것과 좀 다른 듯...
- 아 최대 경로! 노드에서 노드로!

이를 통해 depth를 깊게 하는 것이 더 강력하고 효용을 주지만 stacked-RNNs과 비교하여 학습시키기 어려운 이유를 설명할 수 있다.

다음 section에서 LSTM의 주요 mechanism에 초점을 맞추고 이를 RHN을 설계하기 위해 어떻게 사용했는지 설명한다.

## 3. Revisiting Gradient Flow in Recurrent Networks

#### Notations
- $\mathcal{L}$을 길이가 $T$인 입력 수열의 총 loss라 정의
- $t$ 시점의 표준 RNN의 intput과 output을 $\text{x}^{[t]}\in\mathbb{R}^m,\;\text{y}^{[t]}\in\mathbb{R}^n$
- $\text{W}\in\mathbb{R}^{n \times m},\;\text{R}\in\mathbb{R}^{n \times n}$은 input, recurrent 가중치 행렬
- $\text{b}\in\mathbb{R}^n$는 bias vector
- $f$는 point-wise non-linearity

#### Revisiting
Then, $\text{y}^{[t]}=f(\text{W}\text{x}^{[t]}+\text{R}\text{y}^{[t]}+\text{b})$는 표준 RNN의 dynamics

네트워크의 모수 $\theta$를 가지는 loss $\mathcal{L}$의 미분값은 연쇄 법칙을 사용하여 아래와 같이 확장될 수 있다.
$$\cfrac{d\mathcal{L}}{d\theta}=\sum_{1\leq t_2 \leq T}\cfrac{d\mathcal{L}^{[t_2]}}{d\theta}=\sum_{1 \leq t_2 \leq T}\sum_{1 \leq t_1 \leq t_2}\cfrac{\partial\mathcal{L}^{[t_2]}}{\partial\text{y}^{[t_2]}}\cfrac{\partial\text{y}^{[t_2]}}{\partial\text{y}^{[t_1]}}\cfrac{\partial\text{y}^{[t_1]}}{\partial\theta}\quad\cdots\quad(1)$$

Jacobian 행렬 $\frac{\partial\text{y}^{[t_2]}}{\partial\text{y}^{[t_1]}}$은 모든 time step에 미분 연쇄로 얻어질 수 있다.
$$\cfrac{\partial\text{y}^{[t_2]}}{\partial\text{y}^{[t_1]}}:=\prod_{t_1 < t \leq t_2}\cfrac{\partial\text{y}^{[t]}}{\partial\text{y}^{[t-1]}}=\prod_{t_1 < t \leq t_2}\text{R}^{\top}\text{diag}[f^\prime(\text{R}\text{y}^{[t-1]})]\quad\cdots\quad(2)$$
- 위 행렬은 time step $t_2$에서 time step $t_1$으로 오차를 전송하는 key factor가 된다.
- 위의 식에서 input과 bias는 생략했다.

#### Conditions for the gradient vanish or explode
이제 gradient vanish / explode에 대한 조건을 얻을 수 있다.
- $\text{A}:=\frac{\partial\text{y}^{[t]}}{\partial\text{y}^{[t-1]}}$을 temporal Jacobian,
- $\gamma$를 $f^\prime(\text{R}\text{y}^{[t-1]})$의 maximal bound,
- $\sigma_{max}$를 $\text{R}^\top$의 제일 큰 singular value라고 하자.

Then, Jacobian의 norm은 아래를 만족한다.
$$\Vert \text{A} \Vert \leq \Vert \text{R}^\top \Vert \bigg\Vert \text{diag}[f^\prime(\text{R}\text{y}^[t-1])] \bigg\Vert \leq \gamma\sigma_{max}\quad\cdots\quad(3)$$
- $Eq(2)$와 같이 vanishing gradients ($\gamma\sigma_{max} < 1$)에 대한 조건을 제공한다.
- $\gamma$는 활성화 함수 $f$에 종속적이다. e.g. $| tanh^\prime (x) | \leq 1,\;| \sigma^\prime (x) | \leq \frac{1}{4},\;\forall x\in\mathbb{R},\text{where}\;\sigma = sigmoid$
- 유사하게 $\text{A}$의 spectral radius $\rho$에 대해 $\Vert \text{A} \Vert \geq \rho$이기 때문에 만일 $\rho>1$이면 exploding gradients가 발생한다.


#### Geršgorin circle theorem
**a.k.a 초기화의 중요성!**

largest singular values $\sigma_{max}$와 spectral radius $\rho$ 문제에 대한 위의 설명은 기울기 소실 및 폭주에 대한 경계 조건을 조명하지만 고윳값(eigenvalues)들이 전체적으로 어떻게 분포하는지는 조명하지 않는다. `Geršgorin circle theorem`을 적용하여 이 문제를 조명하고자 한다.

$\begin{array}{ll}
\text{Thm> }\;\text{Geršgorin circle theorem}\\\\
\quad\text{ For any square matrix } \text{A}\in\mathbb{R}^{n \times m},\\
\end{array}$
$$\text{spec}(\text{A}) \subset \bigcup_{i\in\{1,\dots,n\}}\bigg\{ \lambda\in\mathbb{C} \;\bigg|\; {\Vert \lambda - a_{ii} \Vert}_{\mathbb{C}} \leq \sum_{j=1,j \neq i}^{n} |a_{ij}|\bigg\}\quad\cdots\quad(4)$$

즉, 행렬 $\text{A}$의 spectrum으로 구성되는 고윳값들은 complex circle의 합집합 내에 위치한다.
- complex circle? 행렬 $\text{A}$의 대각항의 값 $a_{ii}$을 중심으로 하는 원!!
- 이 원의 반지름(radius)은 $\sum_{j=1,j \neq i}^{n} |a_{ij}|$
    - 이는 $\text{A}$의 각 행의 대각항이 아닌 원소들의 절댓값의 합

#### GCT로 알아보는 Jacobian의 eigenvalues들의 distribution
위 GCT를 사용하여 $\text{R}$의 요소와 Jacobian의 고윳값의 가능한 위치간의 관계를 이해할 수 있다. 대각성분 $a_{ii}$를 옮기는 행위는 고윳값의 가능한 위치도 옮긴다. 큰 값의 off-dianogal 고윳값의 분포가 퍼지게 만든다. 작은 off-diagonal 요소들은 작은 radii를 뱉고 때문에 대각성분 $a_{ii}$ 주변에 고윳값의 분포를 제한한다.

![gct](https://user-images.githubusercontent.com/37775784/80912574-cf326600-8d78-11ea-8682-3b9872e85f60.PNG)

$\text{R}$이 표준 정규 분포($\text{zero-mean  Gaussian distribution}$)으로 초기화된다고 가정하자. 이로부터 우리는 다음과 같이 추론할 수 있다.
- 만일 $\text{R}$의 값이 0에 가까운 표준편차로 초기화되면, $\text{R}$에 굉장히 종속적인 $\text{A}$의 spectrum 또한 0 주변을 중심으로 초기화된다. $\text{A}$의 행은 위 Figure 2의 (1)번 게르슈고린 원에 해당될 수 있다. 이 경우 $\text{A}$의 고윳값들의 크기 $| \lambda_i |$는 1보다 작을 것이다. 추가적으로, 일반적으로 사용되는 L1/L2 가중치 정규화를 사용하면 고윳값의 크기 또한 제한된다.
- 반대로 $\text{R}$의 값이 큰 표준편차로 초기화된다면, $\text{A}$에 해당하는 게르슈고린 원들의 반지름들은 증가한다. 따라서, $\text{A}$의 spectrum이 1보다 큰 고윳값을 가져 gradient exploding을 초래할 수 있다. 반지름들이 행렬의 크기에 걸쳐 합산되므로, 행렬이 클수록 원 반지름이 더 커진다. 결론적으로 큰 행렬은 exploding gradients를 피하기 위해 작은 standard deviation에 상응하게 초기화되어야 한다.

LSTM의 변종들과 다르게 다른 RNN들은 일반적으로 time step에 걸쳐 그들의 Jacobian eigenvalues를 신속하게 조절하는(regulate) 직접적인 mechanisum이 없으며 본 연구진은 해당 메커니즘이 복잡한 sequence 처리를 학습하기 위해 효과적이고 필요하다고 가정한다.

Le와 연구진은 2015년 아래 논문에서 단위 행렬(identity matrix)과 off-diagonal의 small 임의의 값으로 $\text{R}$을 초기화하는 방법을 제안했다.
- [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](https://arxiv.org/abs/1504.00941)
- 오 힌튼 교수님..
- [off-diagonal](https://www.quora.com/What-is-off-diagonal)

이는 GCT로 묘사되는 상황을 바꾸는 행위이다. identity 초기화의 결과는 Figure 2의 게르슈고린 원 (2)로 표시된다. 대각성분 $a_{ii}$가 1로 초기화되기 때문에 GCT에서 묘사한 spectrum은 1 주변에 중심하고 gradient가 사라지지 않도록 보장한다. 그러나, 이는 유연한 해결책(remedy; 치료제)이 아니다. 학습 도중 몇몇 고윳값들은 쉽게 1보다 커질 수 있으며 이는 exploding gradients를 야기한다. 우리는 어림짐작하기로 위 Le의 논문에서 사용된 매우 작은 학습율이 이러한 이유라 생각한다.


## 4. Recurrent Highway Networks (RHN)

#### Highway Layer
Srivastava와 연구진들이 2015a에 쓴 논문의 Highway layers는 [적응형 계산](https://arxiv.org/abs/1603.08983)을 사용함으로써 아주 깊은 feedforward network의 학습을 쉽게 만들었다. 간단히 얘기를 꺼내보자.
- $\text{h}=H(\text{x}, \text{W}_H),\;\text{t}=T(\text{x}, \text{W}_T),\;\text{c}=C(\text{x}, \text{W}_C)$; outputs
- $H,\;T,\;C$; nonlinear transforms
- $\text{W}_H,\;\text{W}_T,\;\text{W}_C$; weight matrices (including biases)
- $T$와 $C$는 보통 sigmoid($\sigma$) nonlinearity를 사용
- 각각 _변환된_ input $H$와 original input $\text{x}$를 _수반하기_ 때문에 transform gate와 carry gate로 불린다.

Highway layer computation은 아래와 같음.
$$\text{y}=\text{h}\cdot\text{t}+\text{x}\cdot\text{c}\quad\cdots\quad(5)$$
- "$\cdot$"은 element-wise multiplication

#### RHN
$Recall\;that;$ 표준 RNN의 recurrent state transition은 아래와 같다.
$$\text{y}^{[t]}=f(\text{W}\text{x}^{[t]}+\text{R}\text{y}^{[t-1]}+b)$$

자, 우리는 Recurrent Highway Network(RHN) layer를 각 recurrent state transition에 하나, 혹은 다중 Highway layers를 가미하여 구축하는 것을 제안한다(요구하는 recurrence depth와 동일하게 되도록). 수식적으로,
- layer $\ell\in\{1,\dots,L\}$에서
- $\text{W}_{H,T,C}\in\mathbb{R}^{n \times m},\;\text{R}_{H_{\ell},T_{\ell},C_{\ell}}\in\mathbb{R}^{n \times n}$; nonlinear transform $H$와 $T$, $C$ gate의 가중치 행렬
- $\text{b}_{H_{\ell},T_{\ell},C_{\ell}}\in\mathbb{R}^{n}$; bias
- $s_{\ell}$; layer $\ell$에서의 중간 output 초깃값 $s_0^{[t]}=\text{y}^{[t-1]}$

Then, recurrence depth $L$을 가진 RHN layer는 아래와 같이 쓸 수 있다.
$$s_\ell^{[t]}=\text{h}_\ell^{[t]}\cdot\text{t}_\ell^{[t]}+\text{s}_{\ell-1}^{[t]}\cdot\text{c}_\ell^{[t]}\quad\cdots\quad(6)$$
$\quad\text{where}$
$$\begin{aligned}
\text{h}_\ell^{[t]}=&\;tanh(\text{W}_H\text{x}^{[t]}\mathbb{I}_{\{\ell=1\}}+\text{R}_{H_\ell}\text{s}_{\ell-1}^{[t]}+\text{b}_{H_\ell})\quad\cdots\quad(7)\\
\text{t}_\ell^{[t]}=&\;\quad\;\,\sigma(\text{W}_T\text{x}^{[t]}\mathbb{I}_{\{\ell=1\}}+\text{R}_{T_\ell}\text{s}_{\ell-1}^{[t]}+\text{b}_{T_\ell})\quad\;\,\cdots\quad(8)\\
\text{c}_\ell^{[t]}=&\;\quad\;\,\sigma(\text{W}_C\text{x}^{[t]}\mathbb{I}_{\{\ell=1\}}+\text{R}_{C_\ell}\text{s}_{\ell-1}^{[t]}+\text{b}_{C_\ell})\quad\;\cdots\quad(9)\\
\end{aligned}$$
- $\mathbb{I}_{\{\}}$; indicator function

![rhn3](https://user-images.githubusercontent.com/37775784/80931764-eadc5180-8df6-11ea-9e09-90f0e3c7e19e.PNG)

RHN 계산 그래프의 개략적인 그림은 위 Figure 3에서 확인할 수 있다. RHN layer의 output은 $L^{th}$번째 Highway layer의 output이며 즉 $\text{y}^{[t]}=\text{s}^{[t]}_L$.

$Note\;that$;
- $\text{x}^{[t]}$는 recurrent transition의 첫 번째 Highway layer ($\ell = 1$)에서만 직접 전달됨.
  - 이는 필수는 아니지만 편의상 선택요소
- $s_{\ell-1}^{[t]}$는 이전 time step의 RHN layer의 output.
- Subsequent Highway layer들은 이전 layer들의 output만을 통과시킴
- Figure 3의 중간의 Dotted Vertical Lines이 recurrent transition의 다중 Highway layers를 구분지음

$L=1$인 RHN layer는 본질적으로 LSTM layer의 기본 변형.
이는 조경현 교수 연구팀의 GRU와 이에 관한 연구들과 유사하게 LSTM의 본질적인 구성 요소를 유지함
- 자가적으로 연결된 additive cell들을 통한 정보 흐름을 통제하는 multiplicative gating units
- [GRU, Cho et al., 2014](https://www.aclweb.org/anthology/D14-1179.pdf)
- [Greff et al. (2015), LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069)
- [Jozefowicz et al. (2015), An empirical exploration of recurrent network architectures](https://dl.acm.org/doi/10.5555/3045118.3045367)

그러나 RHN layer는 $L>1$로 확장되며, LSTM이 훨씬 더 복잡한 상태 전환을 모델링하도록 확장시킴.

Highway와 LSTM layer의 다른 변형체들과 마찬가지로 RHN의 변형도 기본 개념을 바꾸지 않고 구축하는 것이 가능함
- 하나 혹은 여러 gate들을 항상 열려있게 고정
- 본 논문의 실험과 동일하게 gate들을 `coupling`

RHN layer의 간단한 수식은 RNN 기반의 GCT와 유사한 분석을 가능케 함. input과 bias를 생략하고,
- recurrence depth = $1$
- $\text{y}^{[t]}=\text{h}^{[t]}\cdot\text{t}^{[t]}+\text{y}^{[t-1]}\cdot\text{c}^{[t]}$ (r.d가 1이기 때문!)

temporal jacobian $\text{A}=\partial \text{y}^{[t]}/\partial\text{y}^{[t-1]}$은 아래와 같이 주어짐
$$\text{A}=\text{diag}(\text{c}^{[t]})+\text{H}^{\prime}\text{diag}(\text{t}^{[t]})+\text{C}^\prime\text{diag}(\text{y}^{[t-1]})+\text{T}^\prime\text{diag}(\text{h}^{[t]})\quad\cdots\quad(10)$$
- $\text{H}^\prime=\text{R}_H^\top\text{diag}[tanh^\prime(\text{R}_H\text{y}^{[t-1]})]\quad\cdots\quad(11)$
- $\text{T}^\prime=\text{R}_T^\top\text{diag}[\sigma^\prime(\text{R}_T\text{y}^{[t-1]})]\quad\cdots\quad(12)$
- $\text{C}^\prime=\text{R}_C^\top\text{diag}[\sigma^\prime(\text{R}_C\text{y}^{[t-1]})]\quad\cdots\quad(13)$


그리고 이의 spectrum은 아래와 같이 주어짐.
$$\begin{aligned}
\text{spec}(\text{A}) \subset& \bigcup_{i\in\{1,\dots,n\}} \bigg\{ \lambda \in \mathbb{C} \;\bigg\vert\\
&\Vert\lambda - \text{c}_i^{[t]} - \text{H}^\prime_{ii}\text{t}_i^{[t]}-\text{C}^\prime_{ii}\text{y}_i^{[t-1]}-\text{T}^\prime_{ii}\text{h}_i^{[t]} \Vert_\mathbb{C}\\
&\leq\sum_{j=1,j\neq i}^{n}\vert\text{H}^\prime_{ii}\text{t}_i^{[t]}+\text{C}^\prime\text{y}_i^{[t-1]}+\text{T}^\prime_{ij}\text{h}_i^{[t]}\vert\bigg\}\quad\cdots\quad(14)
\end{aligned}$$

$\text{Equation}(14)$는 $\text{A}$의 고윳값에서 gate들의 영향력을 잡아냄. 표준 RNN과 비교했을 때 RHN 계층이 게르슈고린 원의 중심과 반지름을 조절하는데 더 많은 flexibility를 가지고 있음을 알 수 있다.

만일 모든 carry gate가 전부 열려있고 모든 transform gate가 전부 닫혀있다고 가정하자. 그러면 $\text{c}=1_n,\text{t}=0_n$이고 sigmoid 함수는 `saturated`하기 때문에 (포화?) $\text{T}^\prime=\text{C}^\prime=0_{n \times n}$. 즉,
$$\text{c}=1_n,\text{t}=0_n \Rightarrow \lambda_i=1\quad\forall i\in\{1,\dots,n\}\quad\cdots\quad(15)$$
- Geršgorin 원의 반지름이 0으로 줄고 각 대각 성분이 1로 setting됐기 때문에 모든 고윳값은 1이 된다.

반대로 생각해보자. 즉, $\text{c}=0_n,\text{t}=1_n$인 경우에 고윳값들은 $\text{H}^\prime$이 된다. gate는 0에서 1 사이의 값을 가지기 때문에 $\text{A}$의 각 고윳값은 위 제약의 combination에 의해 동적으로 조정될 수 있다.

위 분석으로 얻은 key는 아래와 같다.
1. GCT는 temporal jacobian의 full spectrum의 행동과 해당 gating units의 효과를 관찰할 수 있게 만든다. 우리는 충분한 real-world data를 통해 다중 시간적 종속성을 학습하길 기대했지만 데이터에만 의존하는 것은 기울기 소실/폭주 문제를 피하기에 충분치 않다. RHNs 계층은 표준 RNN과 비교했을 시 기억, 망각 그리고 정보의 전달을 동적으로하기 위한 더 변하기 쉬운 setup을 제공한다.
2. Jacobian matrix의 움직임에 대한 효과틀 통해 비선형 게이트 함수가 네트워크 역학의 빠르고 정밀한 규제를 통해 학습을 촉진시킬 수 있음. Depth는 함수에 표현력을 강화시키기 위해 많이 사용되는 방법이며 $H,T,C$ transformation의 다중 레이어를 사용할 idea가 되었다. 본 논문에서는 단순성과 학습의 용이함을 위해 RHN 계층을 $L>1$로 확장하는 것만 선택했지만 이러한 변환을 위해 plain layer를 쌓는 것도 유용할 것으로 기대함
3. spectrum을 control하는 RHN 계층의 유연함에 대한 연구는 LSTM과 Highway network와 그 변형들에 대한 이론적인 이해도를 바꿔놨다. Feedforward Highway Networks에서 layer transformation ($\frac{\partial\text{y}}{\partial\text{x}}$)의 Jacobian은 위 분석에서 temporal Jacobian을 대체한다. 각 Highway layer는 input의 다양한 요소을 얼마나 변형 혹은 그대로 수반할 지에 대한 조절에 대한 유연성을 증진시켜준다. 이러한 유연성은 네트워크 깊이가 높지 않은 경우에도 고속도로 층에서 성능을 개선한 유력한 원인이다. ([Kim et al., 2015; Character-aware neural language models](https://arxiv.org/abs/1508.06615))

## 5. Experiments

**Setup:** 본 작업에서 carry gate를 Highway network에서 제안한 것과 유사하게 아래와 같은 setting으로 transform gate와 couple(묶는 행위)시킨다.
$$C(\cdot)=1_n - T(\cdot)$$

이 Coupling은 GRU에서도 사용된다. 이는 고정된 수의 unit에 대한 모델 크기를 줄이고 값의 폭주(`unbounded blow-up of state values`)를 방지하여 안정적인 학습을 수행할 수 있게 하지만, 이는 모델에게 특정 태스크에 차선책이 될 수 있는 modeling bias를 부여한다.
- Greff et al., 2015 [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069)
- Jozefowicz et al., 2015 [An empirical exploration of recurrent network architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

LSTM 네트워크와 유사한 출력 비선형성이 이 문제를 해결하기 위해 사용될 수 있다.

최적화와 Wikipedia 실험을 위해 학습을 시작할 때 transform gate가 닫혀있도록 편향시킨다.

network를 여러겹 쌓는 것은 이미 유용하다고 알려져 있고 recurrence depth의 영향에 관심이 있기 때문에 본 연구에서 모든 네트워크는 단일 hidden RHN layer만을 사용한다.

**Regularization of RHNs:** 모든 RNN들과 동일하게 알맞은 정규화는 RHN의 좋은 일반화를 위해 필수불가결. 우리는 아래 논문에서 나온 근사적 변형 추론에 근거한 dropout 정규화 기술을 채택
- [A theoretically grounded application of dropout in recurrent neural networks](https://arxiv.org/abs/1512.05287)

위 기술에 의해 정규화된 RHNs을 variational RHNs이라 일컫자.

Penn Treebank 단어 수준 LM task에서 쌍 비교를 위한 input과 output mappings에 weight-tying을 적용/비적용한 결과도 보고. 이 정규화는 아래 논문에서 소개됨
- Inan & Khosravi (2016); [Improved learning through augmenting the loss](https://pdfs.semanticscholar.org/5762/b7deff7e95febe193196d548379ff34b34f1.pdf)
- Press & Wolf (2016); [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)

---
Weight tying?
- [Beyond Weight Tying: Learning Joint Input-Output Embeddings for Neural Machine Translation](https://arxiv.org/abs/1808.10681)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [Tying weights in neural machine translation (stackoverflow)](https://stackoverflow.com/questions/49299609/tying-weights-in-neural-machine-translation)

---

### 5.1 Optimization
RHN은 deep transitions recurrent network의 최적화가 가능하도록 설계된 구조를 가짐. 때문에 우선적으로 실험적으로 증명해야 할 것은 higher recurrence depth를 가진 RHNs이 어떤 최적화 알고리즘에 쉽게 최적화될 지를 찾는 것(간단한 gradient methods 방식).
