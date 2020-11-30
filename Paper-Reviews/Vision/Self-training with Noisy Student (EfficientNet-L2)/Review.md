## Abstract
- EfficientNet을 Label이 된 ImageNet 이미지에 대해서 먼저 학습한 후, 학습된 모델을 Teacher Model로 사용하여 Unlabel 된 이미지 데이터들에 대해 Psuedo Label을 생성 (이 과정을 반복 - Teacher & Student Model)
- Psuedo Label을 생성하는 동안 Teacher Model에는 Noise(Dropout, Stochastic Depth, Data Augmentation)를 주입하지 않음
- 반면, Student Model에는 Noise Injection을 적용 -> Teacher Model보다 더 Generalize하기 위함 
- 기존 방법들 (ex. Knowledge Distillation)과의 차이점 -> Unlabeled Image를 사용하는 점과 Student Model이 Teacher Model만큼 크다는 점
- ImageNet Top-1 Accuracy 88.4%
- ImageNet Top-5 Accuracy 98.7%

***

## Noising Student
- Student Model에게 의도적으로 주는 두 가지 타입의 Noise 
1) Input Noise: Data Augmentation (RandAugment)
2) Model Noise: Dropout, Stochastic Depth

- Unlabeled Image에 Noise Injection 했을 때의 장점
1) Decision Function에서 Local Smootheness를 강화한다는 이점이 있음
2) Data Augmentation을 사용하였을 때, 예를 들어 Student가 Tranlate를 할 때의 이미지가 Tranlate 되지 않은 이미지와 동일한 Class를 가져야 함. 이 Invariant Encourages는 Student Model이 Teacher Model을 뛰어 넘어 더 어려운 이미지를 예측 가능케 함

- Dropout 및 Stochastic Depth가 노이즈로 사용되는 경우, Teacher는 Inference Time (Psuedo Label을 생성하는 동안)에서 앙상블처럼 행동하지만 
Student는 단일 모델처럼 행동 함. 다시 말해, Student는 보다 강력한 앙상블 모델을 모방해야 함

- Noisy Student는 Data Filtering, Balancing과 같은 추가 트릭으로 더 잘 작동 함. 특히 Teacher Model이 일반적으로 도메인 외부 이미지이므로 신뢰도가 낮은 이미지를 필터링. ImageNet의 모든 클래스에는 비슷한 수의 Label이 지정된 이미지가 있으므로 각 클래스에 대해 Label이없는 이미지 수의 균형을 조정해야 함. 이를 위해 이미지가 충분하지 않은 클래스에서 이미지를 복제. 이미지가 너무 많은 클래스의 경우 가장 Confidence한 이미지를 가져옴.

***

## Architecture
- EfficientNet-L2는 EfficientNet-B7보다 더 Deep 하지만 낮은 Resolution을 사용
- 많은 Label이 없는 이미지에 맞는 더 많은 매개 변수를 가짐 
- 모델 크기가 크기 때문에 EfficientNet-L2의 Training Time은 EfficientNet-B7보다 약 5 배 더 걸림

***

## The Importance of Noise in Self-training
- Teacher Model에서 생성 된 Soft Psuedo Label을 사용하기 때문에, Student이 Teacher Model과 정확히 동일하게 훈련되면 Label이 없는 데이터의 교차 엔트로피 손실이 0이되고 Training 신호가 사라짐
- 따라서 자연스럽게 발생하는 문제는 Student가 Soft Pseudo Label을 사용하여 Teacher보다 우수한 성과를 낼 수있는 이유
- Teacher의 지식 만 배우는 것이 아니라 Student에게 추천해야한다는 가설을 세움
- Label이 없는 데이터 양과 Teacher Model 정확도가 다른 두 가지 시나리오에서 Noise의 중요성을 연구
- 두 경우 모두, Label이 있는 이미지를 유지하면서 Student Model을 학습 할 때 Label이 없는 이미지의 Augmentation, Stochastic Depth 및 Dropout을 점차적으로 제거
- 이렇게 하면 Label이 없는 이미지에 대한 Noise의 영향을 Label이 있는 이미지에 Overfitting을 방지하는 영향으로부터 분리 할 수 ​​있음
- 또한 Psuedo Label을 생성 할 때 Noise를 비활성화해야하는지 여부를 연구하기 위해 Noise가 많은 Teacher와 Noise가 없는 Teacher를 사용하는 것을 비교

***

## Training Flow 
<img src="https://github.com/bluein/Paper-Review/blob/master/Self-training%20with%20Noisy%20Student%20(EfficientNet-L2)/image/img01.JPG" width=550 height=400 />

## Comparision
<img src="https://github.com/bluein/Paper-Review/blob/master/Self-training%20with%20Noisy%20Student%20(EfficientNet-L2)/image/img02.JPG" width=550 height=450 />
- EfficientNet과 동일한 Architecture 사용, 반복 학습을 수행하지 않음

***

## Robustness Results on ImageNet-A, ImageNetC and ImageNet-P
- 세 가지 Robustness Test Dataset에서 88.4 %의 Best Accuracy를 달성
- ImageNetA, ImageNet-C 및 ImageNet-P는 Blurring, Fogging, Rotation 및 Scaling과 같은 일반적인 손상 및 섭동이있는 이미지가 포함
- 특히, ImageNet-A는 SOTA의 Accuracy를 크게 떨어 뜨리는 어려운 이미지로 구성되어 있음
- 이 Test Set들은 "Robustness" 벤치 마크로 간주

## A Study of Iterative Training
- Iterative training is effective in producing increasingly better models !!

***

## Conclusion
- 이전 연구에서는 SOTA ImageNet model을 개선하기 위해 수십억 개의 Soft Labeled Data가 필요했음
- 이 연구에서는 Unlabeled Image를 사용하여 SOTA ImageNet model의 Accuracy와 Robustness를 크게 향상시킴
- Self-Training은 Unlabeled Data를 대규모로 활용하기 위한 간단하고 효과적인 알고리즘
- Teacher Model을 넘어서 학습하기 위해 Noisy Student라고 불리는, Noise를 추가하여 Student Model을 개선
- 핵심은 Accuracy와 Robustness를 둘 다 가져간다는 점!





