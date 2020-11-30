# Highway Networks

- Swiss AI Lab IDSIA

## Abstract
neural network의 depth가 성공을 좌우한다는 것은 이론적, 실험적인 증거가 지천에 널려있음. 그러나 depth를 증가시키면 학습이 어려워지기 때문에 여전히 문제가 남아있음. 자, 본 논문에서는 `ease gradient-based training of very deep networks`를 제안하고자 함. 이를 `highway networks`라 부를 것임. `Highway Networks`는 _information highway_ 의 많은 계층에 걸쳐 방해받지 않는 정보 흐름을 허용함. 이는 `LSTM`에서 영감을 받았으며 이 정보 흐름을 정규화하는 gating unit을 적용. 수백의 layer를 쌓아도 `highway network`는 단순 gradient descent를 통해 직관적으로 학습. 깊어져도 효율적으로 학습이 가능!

## 1. Introduction & Previous Work

#### 깊게 쌓으면 쌓을수록 neural network의 성능은 좋아짐
- ImageNet 분류에서의 예시

#### 뭔소린지 잘 모르겠다!! 단순 feed-forward와 recurrent에 대한 비교,,?
- n bit parity problem
    - [Solving the N-bit parity problem using neural networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608099000696)
    - [A Solution for the N-bit Parity Problem Using a Single Translated Multiplicative Neuron](https://link.springer.com/article/10.1023/B:NEPL.0000011147.74207.8c)
    - [N-bit parity neural networks: new solutions based on linear programming](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.5442&rep=rep1&type=pdf)
    - single feed-forward로 n개의 binary input units, 매우 많은 hidden layer를 쌓아야하지만
    - recurrent hidden unit은 임의의 n에 대해 3 units과 5 weights로 문제를 푸는 것이 가능, input bit string을 1 bit로 처음만 읽고 이를 flip (?)

#### Deep network 학습을 쉽게 하기위한 기존 연구들!
- Developing better optimizers
    - [Training deep and recurrent networks with hessian-free optimization](https://www.cs.utoronto.ca/~jmartens/docs/HF_book_chapter.pdf)
    - [On the importance of initialization and momentum in deep learning](https://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    - [Identifying and attacking the saddle point problem in high-dimensional non-convex optimization](https://papers.nips.cc/paper/5486-identifying-and-attacking-the-saddle-point-problem-in-high-dimensional-non-convex-optimization.pdf)
        - 닙스네... 워
- Well-designed initialization strategies
    - [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
        - Xavier Glorot, Yoshua Bengio
    - [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)
        - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- Another initialization strategies
    - [Random walk initialization for training very deep feedforward networks](https://arxiv.org/pdf/1412.6558.pdf)
    - [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)
- Activation Function!
    - [Maxout networks](https://arxiv.org/pdf/1302.4389.pdf)
    - [Compete to compute](https://papers.nips.cc/paper/5059-compete-to-compute)
        - 닙스네, max-pooling, ReLU 등 설명이 써있네!
- Skip-Connection, 정보의 흐름을 개선시키는데 초점
    - [Deep learning made easier by linear transformations in perceptrons](http://yann.lecun.com/exdb/publis/pdf/raiko-aistats-12.pdf)
        - Yan LeCun(corres.author)
    - [Generating sequences with recurrent neural networks](https://arxiv.org/abs/1308.0850)
        - Alex Graves, 토론토 CS
    - [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
        - Google!, North Califonia/Michigan university
    - [Deeply-supervised nets](https://arxiv.org/abs/1409.5185)
        - DSN

- shallow Teacher to deeper student network!
    - [FitNets: Hints for thin deep nets](https://arxiv.org/pdf/1412.6550.pdf)
        - Bengio(corres.author)
        - knowledge distillation, Hint-based learning
        - Teacher-Student networks
    - [Learning complex, extended sequences using the principle of history compression](https://www.mitpressjournals.org/doi/10.1162/neco.1992.4.2.234)
        - neural history compressor for sequences
        - distilled!
- layer-wise?!
    - [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
    - 이 접근은 직관적인 학습에 비해 덜 매력적이라네요

#### Vanishing Gradient in standard recurrent networks
- [Untersuchungen zu dynamischen neuronalen Netzen](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)
    - 아니 이건 뭐... 독일어여 뭐여
    - 동적 신경망에 관한 연구라
- 신경망을 깊게 쌓을 경우 vanishing gradient문제도 심각
- non-linear transformation을 겹겹이 쌓을 경우 gradient 전파에 부진을 초래
- 위와 같은 이유들 때문에 deep-network의 이점을 조사하는 것이 힘듦

#### 극!뽂!
- 위 문제를 해결하기 위해 LSTM에서 영감을 얻음
    - [Long short-term memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
    - [Learning to forget: Continual prediction with LSTM](https://pdfs.semanticscholar.org/e10f/98b86797ebf6c8caea6f54cacbc5a50e8b34.pdf)
- 레이어를 통과하며 정보를 더욱 쉽게 전파하는 very deep feedforward network의 수정된 architecture를 제시
- LSTM은 adaptive gating mechanism으로 동작함!
- `attenuation(감쇠)`없이 수 많은 layer를 가로질러 정보가 흐를 수 있는 계산 경로를 가능케 함
- 이 path를 `information highways`라고 부르자!
- 이는 기존의 `plane networks`와 대조되는 `highway networks`를 yield!

#### Contribution
- SGD를 사용하여 매우 깊은 highway networks를 학습!
    - plane network는 depth가 증가할 수록 최적화가 힘들어짐...
- computing 예산이 제한된 Deep network를 highway network로 전환하여 single stage로 직관적인 학습이 가능
    - FixNets에서도 개선했지만 이는 2-stage로 진행!

## 2. Highway Networks

#### Notation.
- vector와 행렬은 boldface
- transformation function은 italic체
- zeros vector와 ones vector를 $0$과 $1$로 표기
- $\text{I}$는 identity matrix
- $\sigma(x)=\frac{1}{1+e^{-x}},x\in\mathbb{R}$
- $(\cdot)$은 element-wise multiplication

#### Plain feedforward neural network
- $L$개의 layer로 구성, $l\in\{1,2,\dots,L\}$
- 각 layer는 input $\text{x}_l$에 non-linear transformation $H$(parameterized by $\text{W}_{\text{H},{l}}$)를 적용시켜 output $\text{y}_l$를 출력
- $\text{x}_1$은 network input, $\text{y}_L$은 network의 output
- layer index와 bias를 생략하고 수식을 적으면

$$\begin{aligned}
y=H(\text{x},\text{W}_{\text{H}})\quad\cdots(1)
\end{aligned}$$

#### Highway Networks
- $H$는 non-linear 활성화 함수를 따르는 `affine transform`
- 보통 이를 convolution 혹은 recurrent의 형태로 사용
- Highway network를 위해 두 non-linear transforms $T(\text{x},\text{W}_{\text{T}})$와 $C(\text{x},\text{W}_{\text{C}})$를 추가!

$$\begin{aligned}
y=H(\text{x},\text{W}_{\text{H}})\cdot T(\text{x},\text{W}_{\text{T}})+x\cdot C(\text{x},\text{W}_{\text{C}})\quad\cdots(2)
\end{aligned}$$

- $T$를 `transform gate`, $C$를 `carry gate`라 명명
- $T$는 변환된 input을 $C$는 input을 carrying!
- 본 연구에서 $C=1-T$로 세팅하여

$$\begin{aligned}
y=H(\text{x},\text{W}_{\text{H}})\cdot T(\text{x},\text{W}_{\text{T}})+x\cdot (1-T(\text{x},\text{W}_{\text{T}}))\quad\cdots(3)
\end{aligned}$$

- $\text{x},\text{y},H(\text{x},\text{W}_{\text{H}}),T(\text{x},\text{W}_{\text{T}})$의 차원은 전부 같음 (eq(3)의 연산을 위해!)
- $\text{Note that:}$ 위 `layer transformation`은 Eq(1)보다 훨씬 유연함!
- 이를 아래와 같이 써보자! (특별한 case)

$$y=\begin{cases}x,&\text{if }T(\text{x},\text{W}_{\text{T}})=0\\H(\text{x},\text{W}_{\text{H}}),&\text{if }T(\text{x},\text{W}_{\text{T}})=1\end{cases}\quad\cdots(4)$$

- 유사하게 이 `layer transform`의 `Jacobian`을 아래와 같이 쓸 수 있음

$$\cfrac{d\text{y}}{d\text{x}}=\begin{cases}\text{I},&\text{if }T(\text{x},\text{W}_{\text{T}})=0\\H^\prime(\text{x},\text{W}_{\text{H}}),&\text{if }T(\text{x},\text{W}_{\text{T}})=1\end{cases}\quad\cdots(4)$$

- Transform gate의 출력에 따라 highway layer는 $H$와 input사이의 행동을 smoothing하는 것이 가능
- plain layer는 $y_i=H_i(\text(x))$의 다중 단일 unit으로 구성
- highway network는 **Block state** $H_i(\text{x})$와 **Transform gate output** $T_i(\text{x})$의 다중 block으로 구성

### 2.1 Constructing Highway Networks

#### 중간에 차원을 바꿀 전략
- $\text{x},\text{y},H(\text{x},\text{W}_{\text{H}}),T(\text{x},\text{W}_{\text{T}})$의 차원은 모두 같음
- 중간 representation의 size를 변화시키기 위해 적합한 sub-sampling 혹은 $\text{x}$를 zero-padding하여 얻어지는 $\hat{\text{x}}$으로 $\text{x}$를 대체
- 다른 대안으로는 plain layer를 사용하여 차원을 변경!
    - 본 연구에서 사용할 전략

#### Convolutional highway layers
- `ConvHighwayLayer`는 $H,T$에 대해 weight-sharing과 local receptive fields를 활용
- 때문에 동일한 크기의 receptive field(kernel size)를 사용
- 그리고 block state와 transform gate feature maps가 input size와 같은 크기를 가지도록 zero padding

### 2.2 Training Deep Highway Networks

#### Define Transform Gate
- Transform gate $T(\text{x})=\sigma({\text{W}_{\text{T}}}^T\text{x}+\text{b}_{\text{T}})$
- Simple initialization scheme
    - $H$는 independent
    - $b_T$는 음수(-1, -3 등)으로 초기화
    - 왜? network가 초기에 _carry_ 행동으로 편향되도록 하기 위해서!!!
    - 왜 그럴지는 한번 생각해보시라~~~
- 이 scheme는 LSTM에서 또 영향받았음
    - [Leaning to forget: Continual prediction with LSTM](https://pdfs.semanticscholar.org/e10f/98b86797ebf6c8caea6f54cacbc5a50e8b34.pdf)
    - LSTM에서 학습 중 long-term temporal dependencies를 학습하는 것을 돕기 위해 gate bias를 초기화하는 것에 영감을 받음
    - 자세한 것은 논문을 읽어보라
- $\text{Note that: }\sigma(x)\in(0,1),\forall x\in\mathbb{R}$, 즉, Eq(4)는 절대 나올 수 없음!

#### Sufficient for Negative bias initialization
- `negative bias initializatoin`이 다양한 활성화 함수 $H$와 다양한 zero-mean 초기 분포 $W_{H}$로 deep network를 학습시키는 것보다 훨씬 효율적이라는 사실을 실험적으로 밝힘
- 선행 연구에서 SGD는 1,000개의 layer를 거치면서 교착 상태에 빠지지 않았다. (음?)
    - stall: 교착상태에 빠지다
- initial bias를 `ConvHighwayLayer`의 경우 depth 10, 20, 30에 따라 -1, -2, -3으로 setting하는 것을 추천! (guideline으로 삼아줘~)


## 3. Experiments
- 모든 network는 momentum SGD로 훈련
- Learning rate는 지수적으로 감소
- learning rate는 $\lambda$에서 시작하여 고정된 schedule factor $\gamma$만큼씩 감소시킴
- CIFAR-10에서 최상의 성능을 내도록 설계됨
- 모든 `ConvHighwayLayer`는 block state $H$를 계산하기 위해 ReLU를 사용
- Caffe와 Brainstorm framework를 사용했다네요
- 나는 Pytorch로 할거지롱~

### 3.1 Optimization

#### Rigorous Optimization Experiments
- 본 연구의 `Highway Network`가 depth크기의 증가에 강건하다는 가설을 지지하기 위해 엄밀한 최적화 실험을 수행
    - 이는 normalized initialization을 거친 plain network와 비교 작업을 수행

#### Optimization 실험 설계
- MNIST dataset에서 다양한 depth의 plain, highway network를 학습시킴
- 모든 network는 _thin_ 함!
    - ~~
    - 모든 layer에서
(생략)

#### ㅇㄻㅇㄴ
(생략)

## 4. Analysis
(생략)

### 4.2 Layer Importance

#### Lesioning
- 초기엔 Transform Gate가 닫혀있도록 설계 (Carry Gate가 활성화되도록 bias함)
- 때문에 초기 모든 layer는 이전 layer의 activation을 복사

Q) 훈련이 실제로 이러한 행동을 변화시키는가, 아니면 최종 네트워크가 훨씬 더 적은 계층을 가진 네트워크와 본질적으로 동등한가?

- `To shed ligh on this issue`, 단일 layer의 장애(`lesioning`)가 전체 네트워크의 성능에 얼마나 영향을 미치는 가를 조사
- `Lesioning`: layer의 transform gate를 0으로 세팅하여 단순히 input을 복사하게 설정
- 각 layer별로 성능을 평가
- 결과는 figure 4에 정리

#### MNIST 결과
(생략)

#### CIFAR-100의 결과
(생략)

## 5. Discussion
(생략)
