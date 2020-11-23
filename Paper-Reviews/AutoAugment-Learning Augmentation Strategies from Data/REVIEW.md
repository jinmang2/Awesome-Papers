## Abstract
- Data Augmentation Policy를 수동으로 찾는 것이 아닌, 자동으로 검색하는 AutoAugment라는 Policies를 제안
- 구현 시 Policy는 여러 Sub-Policy로 구성되는 Search Space를 설계하였으며, 그 중 하나는 각 Mini-Batch의 각 이미지에 대해 Random하게 선택
- Sub-Policy는 Translation, Rotation, Shearing 기능이 적용되는 확률(Probabilities) 및 크기(Magnitudes) 두 가지 작업으로 구성 
- Search Argorithm을 사용하여 Neural Network가 Target Dataset에서 가장 높은 Validation Accuracy를 얻을 수 있는 Best Policiy를 Searching

***

## Introduction
- Data Augmentation은 Data를 Random하게 증가시켜 Data의 양과 Diversity를 모두 증가시키는 효과적인 방법
- Image Domain에서 일반적인 Augmentation은 Translating by Pixel, Flipping 등이 있음
- 직관적으로 Data Augmentation은 Data Domain의 불일치에 대한 Model을 학습하는데 사용
- 지금까지 Vision분야의 Community는 더 나은 Network Architecture를 설계하는 것에 큰 초점을 맞추어왔음
- 더 많은 Invariances를 통합하는 더 나은 Augmentation 방법을 찾는 것은 별로 신경을 쓰지 않았음
- 예를 들어, ImageNet에서 2012년에 도입 된 Augmentation 방법은 약간의 변경만 있고 아직까지 표준으로 남아있음
- 또한, 특정 Dataset에 대해 Augmentation Method의 성능 향상이 발견 된 경우도 다른 Dataset에 효과적으로 Transfer 되지 않았음 (예를 들어, Horizontal Flip은 CIFAR-10 Dataset에 대해선 효과적이지만, MNIST에 대해선 그렇지 않음)
- **이 논문에서의 목표는 Target Dataset에 대한 효과적인 Augmentation Policy 탐색의 자동화!**

***

## Related Work
- Dataset으로부터 Augmentation Policies를 찾는 접근 방식은 기본적으로 architecture search로 부터 영감을 받음
- Simple Transformation, Bayesian approach, GAN을 이용한 Data Augmentation은 기존에 진행 되었던 연구들임

***

## AutoAugment: Searching for best Augmentation policies Directly on the Dataset of Interest
- 저자들은 최적의 Augmentation Policy를 찾는 문제를 Discrete Search Problem (개별 검색 문제)으로 정의를 함
- 이 방식은 크게 두 가지 컴포넌트(Search Algorithm & Search Space)로 구성이 되어있음
- 고수준에서 보면, Search Algorithm(RNN Controller로 구현)은 Data Augmentation Policy인 S를 추출하는 역할
(Policy는 image processing operation의 사용, operation의 강도, 각 batch마다 operation을 사용할 확률에 대한 정보를 포함)
- 이 방식에서의 핵심은 Policy S는 고정된 Train Network의 Architecture에서 사용된다는 점, 그리고 Validation Accuracy인 R이 Controller에 업데이트가 됨

***

## Search Space Details 
- Policy는 5개의 sub-policies로 구성, 각 sub-policy는 2개의 image operation이 연속적으로 구성
- 각 Operation은 2개의 Hyperparameter로 구성
1) Operation을 적용할 확률
2) Operation의 강도

***

**Figure 1**

<img src="https://github.com/bluein/Paper-Review/blob/master/AutoAugment-Learning%20Augmentation%20Strategies%20from%20Data/image/img01.JPG" width=550 height=350 />

**Flow**:
- Controller RNN은 Search Space로 부터 Augmentation Policy를 예측
- 고정된 Architecture를 가진 Child Network는 Accuracy R을 달성하기 위한 Convergence 학습 진행
- Reward R은 Gradient Method를 통해 Controller를 업데이트, 더 나은 Policies를 생성

***

**Figure 2**

<img src="https://github.com/bluein/Paper-Review/blob/master/AutoAugment-Learning%20Augmentation%20Strategies%20from%20Data/image/img02.JPG" width=550 height=270 />

**Flow**
- mini-batch 안의 모든 이미지들에대해서, 변헝된 이미지를 생성하여 Network에 학습시키기 위해 sub-policy를 균일하고 랜덤하게 선택 
- 각 sub-policy는 2개의 operation으로 구성, 각 operation은 2개의 숫자 값으로 구성: operation 적용 확률, operation 강도
- 그러나, 만약 Operation이 적용되었을 때의 Operation 강도는 고정! 
- 동일한 sub-policy를 사용하더라도 하나의 image가 다른 mini-batch에서 다르게 변환 될 수있는 방법을 보여줌으로써 sub-policy를 적용 할 때의 확률성.. **저자들이 강조한 부분!!**

***

## Search Algorithm Details
- search algorithm은 기본적으로 강화 학습 방식을 사용
- search algorithm은 2개의 컴포넌트로 구성: 
1. RNN 기반의 Controller 
2. Training Algorithm (Proximal Policy Optimization Algorithm)
- 각 Step 마다 Controller는 Softmax에 의해 나온 Decision으로부터 Predict (그리고 난 후, Prediction은 Next Step으로 Embedding)
- 전체적으로 controller는 5개의 Sub-Policies를 Predict하기 위한 30개의 Softmax Prediction이 있는데, 각 2개의 Operation과 각 Operation은 Operation Type, Magnitude(강도), Probability가 요구 됨

***

## The training of controller RNN
- Controller는 Reward Signal과 함께 학습이 되고, Child Model의 Generalization을 향상시키는 Policy를 어떻게 찾는 것인지가 핵심!
- 실험에서는, Validation Set을 Child Model의 Generalization을 측정하는 수단으로 사용









