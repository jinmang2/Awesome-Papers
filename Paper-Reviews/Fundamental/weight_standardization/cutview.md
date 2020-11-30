# Weight Standardization

## Abstract
- 딥러닝 학습 가속화를 위한 Weight Standardization(WS) 제안
- 학습에 1-2 images만 각 GPU가 가질 때의 micro-batch 학습 세팅에 초점을 둠
- micro-batch 학습은 어려움
- WS는 Group Normalization과 GPU당 1개씩의 image를 학습하여 위를 해결
- WS는 오직 2줄의 코드로 많은 batch size를 가지는 BN의 성능을 뛰어넘음
- WS는 conv-layer의 가중치를 정규화시킴으로써 뛰어난 성과를 얻음.
- 위는 loss와 gradient의 lipschitz 상수를 줄임으로써 loss landscape를 평탄화시켜서 가능한 일.
- WS가 효과적이라는 사실은 아래 여러 task에서 이미 증명.
    - image classification
    - object detection
    - instance segmentation
    - video recognition
    - semantic segmentation
    - point cloud recognition

## 2. Weight Standardization
- BN
    - 최적화 문제의 landscape을 smooth하게 만들어줌
    - [How does batch normalization help optimization? ](https://papers.nips.cc/paper/7515-how-does-batch-normalization-help-optimization.pdf)에선 BN이 loss function의 `Lipschitz constants`를 줄이고 gradient를 더욱 `Lipschitz`하게 만듦
        - $|f(x_1)-f(x_2)| \leq L|x_1-x_2|$, gradient exploding을 미연에 방지 가능
    - 이는 즉, loss가 더 나은 $\beta$-smoothness를 가지게 함.
- 저자는 BN이 optimizer가 직접 최적화하는 `weights`가 아닌 `activations`와 관련하여 `Lipschitz constants`를 고려함.
    - 즉, weights를 표준정규화하여 landscape을 더욱 smooth하게 만들 수 있음을 논할 것임.
    - weight에서 activation으로 smoothing 효과를 전이하고 activation과 weights에 대한 smoothing 효과가 추가됨.

### 2.1 Weight Standardization

#### standard convolutional layer with its bias term set to 0

$$y=\hat{W}*x$$
- $\hat{W}\in\mathbb{R}^{O\times I}$: weights in the alyer
- $*$: convolution operation
- $O=C_{out}$: number of the output channels
- $I=C_{in}\times\text{kernel size}$

#### formula of weight standardization
- original weights $\hat{W}$를 직접 최적화하는 것이 아닌, $W$의 함수로($\hat{W}=WS(W)$) weight $\hat{W}$를 reparameterize(재모수화)시키고 $W$를 SGD로 loss $\mathcal{L}$을 최적화

$$\begin{aligned}
\hat{W}&=\big[ \hat{W}_{i,j} | \hat{W}_{i,j} = \cfrac{W_{i,j}-\mu w_{i,\cdot}}{\sigma w_{i,\cdot}+\epsilon} \big]\\
y&=\hat{W}*x\\
\end{aligned}$$
$\text{where}$
$$\mu w_{i,\cdot}=\cfrac{1}{I}\sum_{j=1}^{I}W_{i,j},\;\sigma w_{i,\cdot}=\sqrt{\cfrac{1}{I}\sum_{i=1}^{I}(W_{i,j}-\mu w_{i,\cdot})^2}$$

- 위는 단순히 표준 정규화라고 생각하면 됨.
- j는 input axis, 즉 아래 그림과 같이 가중치 표준 정규화를 실시

![ws1](/assets/ws1.PNG)

- WS는 BN과 유사하게 conv 계층의 독립적인 각 output channel들의 가중치의 1차, 2차 모멘텀을 조절함
    - 향후 수식적으로 분석하기
- 많은 초기화 방법들이 위와 비슷한 방법으로 가중치를 초기화함.
- 위와의 차별점으로 **WS는 역전파 동안 gradient가 정규화되는 것에 초점을 둠.**
- $Note\;that$: $\hat{W}$에 어떠한 `affine transformation`도 가하지 않음.
- 논문에서의 가정
    - BN과 GN과 같이 정규화된 계층은 conv 계층을 다시 정규화할 것
    - affine transformation의 첨가는 학습에 해로움

위를 보이기 위해 우선 WS가 gradient를 정규화하는 효과를 논함

### 2.2 WS normalizes gradients
set $\epsilon=0$.
