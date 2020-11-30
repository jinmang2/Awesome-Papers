# Real-World Anomaly Detection in Surveillance Videos
- CVPR 2018
- http://openaccess.thecvf.com/content_cvpr_2018/papers/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.pdf

## Abstract
- 감시 카메라는 실제 이상현상을 포착함
- 본 논문에서는 정상과 비정상 비디오를 모두 활용하여 이상현상을 학습하도록 제안
- Deep MIL(Multiple Instance Learning)을 활용한 학습을 제안
- 각 비디오를 하나의 bag으로, video segments를 MIL의 하나의 instance로, 그리고 각 anomalous video segment에 높은 anomaly scores를 예측할 수 있게 Deep anomaly ranking model을 학습
- 저자는 추가로 sparsity와 temporal smoothness constraint를 제시
- Large-scale dataset
    - 13 realistic anomalies label
    - 128 hours of videos
    - 1,900 vidoes
    - untrimmed video
    - 이상/비이상의 이진 분류 혹은 13개의 이상행동 탐지의 다중 분류  task로 사용 가능
- 실험 결과: 우리의 MIL method는 SOTA임 우후훗

## 1. Introduction
- CCTV 데이터는 급증 / 필요성도 급증
- 그러나 보안 요원이 이 영상을 항시 감시할 수 없음, 인간의 모니터링에는 한계가 존재
- 즉, 이를 포착하는 모델을 제안
- 인간의 노고와 시간을 덜기 위해 intelligent computer vision 알고리즘이 필요
- anomaly detection 모델의 goal은 적시에 이상 현상을 알리는 것.
- normal patterns과 다른 video를 식별, Understaing video
- 실세계의 이상 현상은 정교하고 다양함 >> 어렵다는 얘기..
- 때문에 이전에 알고있던 이상 현상의 정보에 의지하는 것이 아닌 최소한의 감독으로 수행되어야 함
    ```
    - 해석을 잘못한 것 같아서 본문 내용을 남김
    - Therefore, it is desirable that the anomaly detection algorithm does
      not rely on any prior information about the events
    - In other words, anomaly detection should be done with minimum supervision
    ```
- [28, 42]의 논문은 Sparse-coding 기반의 representative method를 제안, SOTA의 성적을 거둠
- 위 방법론에서 video의 초기부분에만 normal event가 포함되어 있으므로 이를 기반으로 normal event dictoinary를 작성, 사용
- 이 때, main idea는 normal event dictionary에서 이상 현상을 정확하게 재구축할 수 없다는 것
- 그러나 이상 현상은 시간에 따라 역동적으로 변하기 때문에 normal 행동과 다른 경우 alarm을 울려서 이를 알려줌
    ```
    - 근데 왜 small portion initial 부분에만 normal event가 포함되어 있지?
    - 해석을 잘못한거야 아님 논문의 주장을 너무 생략한거야?
    ```

### Movitation and contributions.
- [28. 42] 논문과 다른 기존의 접근 방식은 normal pattern을 학습하여 비 이상을 탐지할 수 있다고 가정함
- 그러나 이 가정은 언제나 true가 아님! **normal event를 과연 어떻게 정의할 것인가? 이는 굉장히 어려운 일!**
- **이상과 비이상의 경계는 모호함**, 그리고 실제 상황에서 경우에 따라 **같은 행동이 이상/비이상이 되는 경우도 비일비재함**
- 본 논문에서는 weakly label된 training video를 활용한 anomaly detection 모델을 제안하겠노라 닝겐들아.
- 즉, 이런 거시다! **각 비디오는 normal과 anomaly를 가지고 있으나 어디 시점에 있는지는 모른다!!**
- 이렇게하면 비디오별로 labeling하는 것이 가능해져서 많은 양의 video를 쓸 수 있다! 신기하지 닝겐들아?
- weakly-supervised learning 어떻게? MIL로 할거시다 닝-겐
- ranking model로 학습시킬 것임
- test하면서, long-untrimmed video를 segment로 나누고 모델에 input으로 넣어줌

### Summary: This paper makes the following contributions;
- MIL Solution 제시 by leveraging only weakly labeled training videos
    - sparsity와 temporally smoothness constraints를 가진 MIL ranking loss는 DNN에서 각 video segment별 anomaly scores를 계산
- 1,900건의 large dataset을 소개
- 실험 결과 SOTA 짱짱맨
- 우리의 Dataset은 untrimmed video 활동 인식에 도전적인 benchmark가 될 듯.
    - 왜? complexity of activities and large intra-class variations
    - baseline과 C3D[37], TCNN[21]의 결과를 제공

## 2. Related Work

### Anomaly Detection
- Anomaly detection은 computer vision에서 중요한 challenge [40, 39, 7, 10, 5, 20, 43, 27, 26, 28, 42, 18, 26]
- CCTV에선 공격성과 폭력성을 detect하는 시도가 있었음 [15, 25, 11, 30]
- Datta et al.은 사람의 사지와 행동에서 폭력성을 포착하는 모델을 제시
- Kooij et al.[25]는 음성도 활용하여 공격적인 동작을 포착
- Gao et al.도 뭐 violent flow descriptor를 제안했다고 함. 뭐 그렇대.
- 최근에, Mohammadi et al.[30]에선 폭력/비폭력 비디오를 heuristic하게 구분하는 접근법을 제안했다고 함. 우.와.
- 기존의 violent와 non-violent 패턴 판별에서 [39, 7]의 저자들은 normal motion을 추적, 이를 토대로 anomaly를 탐지
- 이를 추적하는 일이 쉽지 않기에, 몇몇 접근은 global motion patterns를
    - histogram based [10],
    - topic modeling [20],
    - motion patterns [32],
    - social force models [29],
    - mixtures of dynamic textures model [27],
    - Hidden Markov Model on local spatio-temporal volumns [26],
    - context-driven method [43]으로 파악하는 접근방식을 사용
- normal behavior의 training video가 주어지면 위의 접근 방식들은 normal motion patterns의 분포를 학습, 이를 토대로 비이상을 탐지
- [28, 42]의 연구자들은 normal behaviors의 dictionary를 학습하기 위해 sparse representation을 사용
- test를 진행하며, large reconstruction errors를 가진 패턴은 비정상 행동으로 간주
- Image 분류에 DNN이 아주 성공적인 결과를 거뒀기에, 몇몇 접근에서 NN이 활용됨 [24, 37]
- 그러나, training에 annotation을 하기가 어렵고 노동력이 많이 요구됨
- 최근 [18, 40]에선 normal behavior를 학습하기 위해 AutoEncoder를 사용했고 이상을 탐지하기 위해 reconstruction loss를 사용
- 우리의 접근은 normal behaviors를 고려할 뿐만 아니라 weakly labeled training data를 활용, anomalous behaviors도 탐지

### Ranking
- Rank를 학습하는 것은 Machine Learning의 한 연구 분야입니당★ 꺄!
- 이는 각각의 점수가 아닌 각 instance별 서로 연관된 scores를 증가시키는데 집중합니다.
    - 절대적이 아니라 상대적이라고 이해하면 됩니다!
- Joachims et al.[22]는 검색 엔진에서 검색(retrieval) quality를 증진시키기 위해 rank-SVM을 제안.
- Bergeron et al.[8]은 successive linear programming을 사용해 multiple instance ranking problems을 푸는 알고리즘을 제시
- 그리고 이를 수소 분자 표현(?)에 적용하여 시연했다.
- 최근, Deep Ranking Networks는 computer vision application에서 많이 사용되고 있고 SOTA 성적을 달성하고 있다
- 이들은 아래 문제에서 DRN을 사용했다.
    - Feature Learning [38]
    - Highlight detection [41]
    - Graphics Interchange Format (GIF) generation [17]
    - Face detection and verification [33]
    - Person re-identification [13]
    - Place recognition [6]
    - metric learning and image retrieval [16]
- 모든 deep ranking methods는 positive, negative samples에 대한 표시가 아주 많이 필요하다.
    - 노동력 으어마어마하게 들어간다는 소리
- **위 방시쿠들과 다르게!!! 꺄!** 우리는 anomaly detection을 regression 문제로 풀었다.
    - re...regression이라규! feature vector를 anomaly score(0-1)로 매핑했다능!
- 학습에서 segment-level label의 어려움을 덜기 위해, MIL 방식을 활용

## 3. Proposed Anomaly Detection Method
![img](https://github.com/jinmang2/anomaly_detection_on_video/blob/master/img/anomaly_detection.PNG?raw=true)
- 학습 도중에 CCTV data를 특정 숫자의 segment로 분할하는 접근 방식
- 이 segment는 bag의 instance가 된다.
- positive(anomalous)과 negative(normal) bags를 사용하여 deep MIL ranking loss를 사용한 anomaly detection model을 학습시킨다.

### 3.1 Multiple Instance Learning
- SVM을 활용한 기존 지도학습에선 아래와 같은 optimization function을 사용

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;\min_{W}\frac{1}{k}\sum_{i=1}^{k}{\overbrace{max(0,\;1-y_i(W\cdot\phi(x)-b))}^{\textcircled{1}}}+\frac{1}{2}{||W||}^2\;\cdots\;(1)"/>

    - Where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\textcircled{1}"/> is the hinge loss
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_i"/> represents the label of each example
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\phi(x)"/> denotes feature representation of an image patch or a video segment
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;b"/> is a bias
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;k"/> is the total number of training examples
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;W"/> is the classifier to be learned

- robust한 classifier를 학습시키기 위해, positive와 negative sample에 대한 정확한 annotation이 필요
- 지도학습 기반 anomaly detection의 맥락에서, 분류기는 video의 각 segment의 시간적 의미를 가진 annotation이 필요
- 그러나 각 비디오의 temporal annotations를 얻는 것은 시간적/노동적 loss가 상당함
- MIL은 정확한 temporal annotations을 가져야 한다는 가정을 조금 완화시킴! (짜식들아 MIL갓을 찬양하라!)
- MIL에서, 비디오의 이상 행동의 정확한 시간적 위치는 알 수 없음
- 대신, 전체 video에 video 단위의 label이 필요함
- anomalies를 포함하는 video는 positive로 labeling되고 없는 video는 negative로 labeling
- 그 다음, positive video를 positive bag <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{B}_a"/>로 표현
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{B}_a=(p^1,\;p^2,\;\dots,\;p^m)"/>    where m is the number of instances in the bag
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;Assume\;that:"/> 적어도 이중 하나의 instance는 anomaly.
- 유사하게, negativew video를 negative bag <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{B}_n"/>로 표현
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{B}_n=(n^1,\;n^2,\;\dots,\;n^m)"/>
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;Assume\;that:"/> anomaly인 instance는 없음.
- positive bag의 어느 instance가 anomaly인지 불명확하기 때문에 해당 bag의 maximum score로 obejective function을 optimize한다.
    ```왜냐? max를 취하면 anomaly한 instance의 점수가 제일 높을테니까! (오... 개똒똒)```

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;\min_{W}\frac{1}{z}\sum_{j=1}^{z}{max(0,\;1-Y_{B_j}(\max_{i\in\mathcal{B}_j}(W\cdot\phi(x_i))-b))}+\frac{1}{2}{||W||}^2\;\cdots\;(2)"/>

    - Where <img src="https://latex.codecogs.com/svg.latex?\Large&space;Y_{\mathcal{B_{j}}}"/> denotes bag-level label
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;z"/> is the total number of bags

### 3.2 Deep MIL Ranking Model
- Anomaly behavior를 정확하게 정의하는 것은 매우 어려운 일[9]
- 굉장히 주관적이고 사람마다 다양함
- 데이터도 충분하지 않아 분류 대신 low likelihood pattern detection 문제로 다뤄지는 경우가 많음 [10, 5, 20, 26, 28, 42, 18, 26]
- 우리는 다르다능! 우리는 regression으로 풀거라능!
- anomalous video는 높은 anomaly score를 가질거라능! **normal에 비해서**

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(\mathcal{V}_a)>f(\mathcal{V}_n)\;\cdots\;(3)"/>

    - where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{V}_a\;and\;\mathcal{V}_n"/> represent anomalous and normal video segment
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(\mathcal{V}_a)\;and\;f(\mathcal{V}_n)"/> represent the corresponding predicted anomaly scores raning from 0 to 1, respectively

- 그러나 segment 단위로 labeling을 하게되면... 아니 그걸 누가 할건데? 너가 할거야? 니가 그렇게 빨라?
- (3)식은 사실상 불가능, MIL ranking objective function을 제안!

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;\max_{i\in\mathcal{B_a}}f(\mathcal{V}_a^i)>\max_{i\in\mathcal{B_n}}f(\mathcal{V}_n^i)\;\cdots\;(4)"/>

    - max는 각각의 bag에 취하는 것!

- bag의 각 instance에 집중하는 것이 아닌, positive와 negative bag에서 가장 높은 anomaly score를 가진 두 instance의 관계에 집중
- positive bag에서 가장 높은 anomaly score를 가지는 segment는 true positive instance
- negative bag에서 가장 높은 anomaly score를 가지는 segment는 positive instance로 보이나 실제로는 normal instance
- 위 negative instance는 false alarm을 울리는 분류하기 힘든 instance로 고려
- (4)를 사용하여 positive instance와 negative instance의 anomaly score의 차이를 벌리고 싶음
- 때문에 hinge-loss formulation의 ranking loss는 다음과 같이 쓸 수 있음

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;l(\mathcal{B}_a,\mathcal{B}_n)=\max(0,\;1-\max_{i\in\mathcal{B}_a}f(\mathcal{V}_a^i)+\max_{i\in\mathcal{B}_n}f(\mathcal{V}_n^i))\;\cdots\;(5)"/>

- 후... 아직 문제가 남아있소이다
- anomalous video의 temporal structure(시간구조)를 무시한다는 점이에요..
- 첫 째, real-world scenarios에서는, anomaly는 때로 짧은 시간에 발생한다.
- 이 경우, anomalous bag의 instance들의 scores는 sparse해지고 이는 anomaly를 포함하는 segment가 적다는 것을 의미한다.
- 둘 째, video가 segment의 sequence이기 때문에, anomaly score는 video segment간에서 매끄럽게 변해야 한다.
- 그러므로, 인접한 video segment의 점수 차를 최소화시켜 이들의 anomaly score를 매끄럽게 변화시키는 것에 집중해야한다.
- **Sparsity**와 **Smoothness** Constraint를 추가한 loss function은 아래와 같다.

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;l(\mathcal{B}_a,\mathcal{B}_n)=\max(0,\;1-\max_{i\in\mathcal{B}_a}f(\mathcal{V}_a^i)+\max_{i\in\mathcal{B}_n}f(\mathcal{V}_n^i))+\lambda_1\overbrace{\sum_i^{n-1}(f(\mathcal{V}_a^i)-f(\mathcal{V}_a^{i+1}))^2}^{\textcircled{1}}+\lambda_2\overbrace{\sum_i^n{f(\mathcal{V}_a^i)}}^{\textcircled{2}}\;\cdots\;(6)"/>

    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\textcircled{1}"/> indicates the **temporal smoothness** term
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\textcircled{2}"/> represents the **sparsity** term

- MIL ranking loss에서, error는 positive와 negative bag의 최고점수의 video segment로부터 역전파된다.
    - `어떻게 구현해야할까ㅣ... 허허허 갈 길이 멀구료`
- purpose: network가 positive bag의 anomalous segment에 대해 높은 score로 예측하는 generalized model로 학습하는 것 (아래 figure 참고)

![fig8](https://user-images.githubusercontent.com/37775784/75893967-e0ddb700-5e76-11ea-8032-c9d1a1c38129.PNG)

- 최종 objective function은 아래와 같음

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}(\mathcal{W})=l(\mathcal{B}_a,\mathcal{B}_n)+\lambda_3{||W||}_F\;\cdots\;(7)"/>

    - where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{W}"/> represents model weights

### Bags Formations
- 각 video를 같은 크기의 겹치지 않는 temporal segment로 나누고 이를 bag instance로 사용
- 주어진 video segment에 대해, 3D convolution features를 추출[37]
- 이를 1. computational efficiency와 2. video 행동 인식에서 capturing appearance와 motion dynamics의 evident capability 때문에 3D feature representation을 사용

## 4. Dataset

### 4.1 Previous datasets
pass

### 4-2. Our datasets
- 이전 datasets의 문제 때문에 본 논문에서는 MIL ranking loss의 평가를 위해 large-scale dataset을 새롭게 구축
- 이는 `Abuse, Arrest, Arson, Assault, Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism`의 13개 이상 행동을 포함하는 long untrimmed surveillance videos
- 위 이상 행동들은 공공 안전 상에서 중요한 영향ㅇ르 끼치는 요인들로 선정

![table1](https://user-images.githubusercontent.com/37775784/75894972-539b6200-5e78-11ea-8c83-6a9f4c6b8730.PNG)

#### Video Collection
- YouTube와 LiveLeak에서 text search queries를 사용하여 video를 수집
- Google Translator를 활용하여 다른 언어로도 검색해서 가능한 많은 video를 수집
- 다음 항목들의 video는 제거
    - manually edited
    - prank videos
    - not captured by CCTV cameras
    - taking from news
    - captured using
    - a hand-held camera
    - containing compilation
- anomaly가 clear하지 않은 경우도 무시
- pruning과정을 거친 후에, 950개의 anomalies video가 수집됨
- 같은 과정을 거쳐 950개의 normal video를 얻어 총 1,900개의 동영상으로 dataset을 구축

#### Annotation
- 논문의 방법론에 의하면, video-level의 label만 필요
- 그러나 Figure 8.과 같이 성능을 측정하기 위해서는 temporal annotation을 알아야 함
- 저자는 동일한 동영상을 여러 주석에 할당하여 각 예외의 시간 범위에 레이블을 지정했다고 함
- 최종적인 temporal annotations는 각기 다른 annotator들의 annotation에 평균을 취하여 얻음
- 몇달의 노고를 거쳐 데이터 셋을 완성했다고 합니다 ㅠㅠ

#### Training and testing sets
- training: 800 normal and 810 anomalous videos
- testing: 150 normal and 140 anomalous videos
- some of the videos have multiple anomalies

## 5. Experiemtns

### 5.1 Implementation Details
- C3D[37]의 FC6 layer에서 visual features를 추출
- features를 계산하기 앞서, video frame을 240X320 pixels로 바꾸고 frame rate를 30fps(frames per second)로 고정
- 모든 16 frame video clip(l2 normalization)에 대해 C3D features를 계산
    - 해석이 맞나? `We compute C3D features for every 16-frame video clip followed by l2 normalization.`
- video segment의 features를 얻기 위해 해당 segment에 대해 모든 16-frame clip features의 평균을 취함
- 위에서 계산한 features (4,096D)를 3-layer FC neural network에 넣어줌
- First FC layer는 512 units, Second는 32 units, 마지막 layer는 1 units를 가짐
- 각 FC layer에 60%의 dropout regularizaion[34]가 사용됨
- 1st FC layer엔 ReLU, 3rd FC layer엔 sigmoid activation function을 사용
- 초기 learning rate=1e-3으로 Adagrad[14] optimizer를 사용
- MIL ranking loss의 sparsity와 smoothness constraints의 parameter는 <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda_1=\lambda_2=8\times10^{-5},\;\lambda_3=0.01"/> for the best performance
- 각 video를 32개의 non-overlapping segments로 나누고 각 video segment를 bag의 instance로 고려
- The number of segments (32) is empirically set (경험적으로 얻어짐)
- multi-scale overlapping temporal segments에 대해서도 연구했으나 detection accuracy에 큰 도움이 되지 않음
- mini-batch로 30 positive, 30 negative bags를 random하게 선택
- Theano[36]을 활용하여 automatic differentiation으로 gradient를 계산
- 그 다음 (6)과 (7)을 계산, 모든 batch에 대해 loss를 역전파

#### Evaluation Metric
- 이전 연구와 동일하게[27] ROC(Receiver Operating Characteristic)과 AUC(Area Under the Curve)를 사용하여 methodology를 평가
- EER(Equal Error Rate)는 사용하지 않음[27]

### 5.2 Comparison with the State-of-the-art
pass

### 5.3 Analysis of the Proposed Method

#### Model training
- 본 논문의 assumption은 positive, negative videos가 video-level로 labeling 되어 있다는 것
- network는 자동적으로 video의 anomaly 위치를 예측하도록 학습이 됨
- 위 목표를 당성하기 위해 network는 training iteration 동안 anomalous video segments에 대해 높은 score를 토해내도록 학습해야 함
- 1,000 iteration에선 network는 이상/비이상 video segment에 대해 동일하게 높은 점수를 부여
- 3,000 iteration을 넘기고 나선 network가 normal segments에 대해 낮은 점수를 부여하고 anomalous segment에 대해서는 높은 점수를 유지하기 시작
- iteration을 높이고 network가 더 많은 video를 보게 만드니 모델이 자동적으로 localize anomaly를 정확하게 학습하게 됨
- 비록 본 논문에서 어떠한 segment level annotations을 사용하지 않았지만, network는 anomaly의 시간적 위치를 anomaly scores를 통하여 예측할 수 있게되었다.

#### False alarm rate
- 실제 세계에 적용했을 때, 감시 카메라의 주요 부분은 normal이다.
- 강건한 anomaly detection method는 normal video에 대해 FAR을 적게 울려야 한다.
- 그러므로, 저자는 본 방법론의 성능을 normal video에 대해서만 평가했다.
- 타 방법론 대비 FAR이 현저하게 낮게 나왔다. ([18]:27.2, [28]:3.1, Proposed:1.9)
- 이는 훈련에 이상 비디오와 일반 비디오를 모두 사용하면 깊은 MIL 순위 모델이보다 일반적인 일반 패턴을 학습하는 데 도움이 된다는 것을 반증함

### 5.4 Anomalous Activity Recognition Experiments
pass

## 6. Conclusion
- Propose a deep learning approach to detect real-world anomalies in suveillance videos
- 실제 이상 행동의 복잡함 때문에 normal data만 사용하는 것은 최적이 아님
- 본 논문에서는 이상/비이상 데이터 둘 다 사용하는 시도를 수행
- annotation(labeling)의 노동과 시간 소요를 피하기 위해 Deep MIL frameword with weakly labeled data를 제안
- 위 방식을 검증하기 위해 large-scale anomaly dataset을 구축
- 실험 결과는 baseline methods보다 통계적으로 우수한 성능을 보임
- 게다가 이상 행동 인식 task에서도 효과적으로 활용될 수 있는 demo를 선보임

## References
[1] http://www.multitel.be/image/researchdevelopment/research-projects/boss.php.

[2] Unusual crowd activity dataset of university of minnesota. In
http://mha.cs.umn.edu/movies/crowdactivity-all.avi.

[3] A. Adam, E. Rivlin, I. Shimshoni, and D. Reinitz. Robust real-time unusual event detection using multiple fixedlocation monitors. TPAMI, 2008.

[4] S. Andrews, I. Tsochantaridis, and T. Hofmann. Support vector machines for multiple-instance learning. In NIPS, pages
577–584, Cambridge, MA, USA, 2002. MIT Press.

[5] B. Anti and B. Ommer. Video parsing for abnormality detection. In ICCV, 2011.

[6] R. Arandjelovic, P. Gronat, A. Torii, T. Pajdla, and J. Sivic. ´
NetVLAD: CNN architecture for weakly supervised place
recognition. In CVPR, 2016.

[7] A. Basharat, A. Gritai, and M. Shah. Learning object motion
patterns for anomaly detection and improved object detection. In CVPR, 2008.

[8] C. Bergeron, J. Zaretzki, C. Breneman, and K. P. Bennett.
Multiple instance ranking. In ICML, 2008.

[9] V. Chandola, A. Banerjee, and V. Kumar. Anomaly detection: A survey. ACM Comput. Surv., 2009.

[10] X. Cui, Q. Liu, M. Gao, and D. N. Metaxas. Abnormal detection using interaction energy potentials. In CVPR, 2011.

[11] A. Datta, M. Shah, and N. Da Vitoria Lobo. Person-onperson violence detection in video data. In ICPR, 2002.

[12] T. G. Dietterich, R. H. Lathrop, and T. Lozano-Perez. Solv- ´
ing the multiple instance problem with axis-parallel rectangles. Artificial Intelligence, 89(1):31–71, 1997.

[13] S. Ding, L. Lin, G. Wang, and H. Chao. Deep feature learning with relative distance comparison for person
re-identification. Pattern Recognition, 48(10):2993–3003,
2015.

[14] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient
methods for online learning and stochastic optimization. J.
Mach. Learn. Res., 2011.

[15] Y. Gao, H. Liu, X. Sun, C. Wang, and Y. Liu. Violence detection using oriented violent flows. Image and Vision Computing, 2016.

[16] A. Gordo, J. Almazan, J. Revaud, and D. Larlus. Deep image ´
retrieval: Learning global representations for image search.
In ECCV, 2016.

[17] M. Gygli, Y. Song, and L. Cao. Video2gif: Automatic generation of animated gifs from video. In CVPR, June 2016.

[18] M. Hasan, J. Choi, J. Neumann, A. K. Roy-Chowdhury,
and L. S. Davis. Learning temporal regularity in video sequences. In CVPR, June 2016.

[19] G. E. Hinton. Rectified linear units improve restricted boltzmann machines vinod nair. In ICML, 2010.

[20] T. Hospedales, S. Gong, and T. Xiang. A markov clustering
topic model for mining behaviour in video. In ICCV, 2009.

[21] R. Hou, C. Chen, and M. Shah. Tube convolutional neural network (t-cnn) for action detection in videos. In ICCV,
2017.

[22] T. Joachims. Optimizing search engines using clickthrough
data. In ACM SIGKDD, 2002.

[23] S. Kamijo, Y. Matsushita, K. Ikeuchi, and M. Sakauchi.
Traffic monitoring and accident detection at intersections.
IEEE Transactions on Intelligent Transportation Systems,
1(2):108–118, 2000.

[24] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar,
and L. Fei-Fei. Large-scale video classification with convolutional neural networks. In CVPR, 2014.

[25] J. Kooij, M. Liem, J. Krijnders, T. Andringa, and D. Gavrila.
Multi-modal human aggression detection. Computer Vision
and Image Understanding, 2016.

[26] L. Kratz and K. Nishino. Anomaly detection in extremely
crowded scenes using spatio-temporal motion pattern models. In CVPR, 2009.

[27] W. Li, V. Mahadevan, and N. Vasconcelos. Anomaly detection and localization in crowded scenes. TPAMI, 2014.

[28] C. Lu, J. Shi, and J. Jia. Abnormal event detection at 150 fps
in matlab. In ICCV, 2013.

[29] R. Mehran, A. Oyama, and M. Shah. Abnormal crowd behavior detection using social force model. In CVPR, 2009.

[30] S. Mohammadi, A. Perina, H. Kiani, and M. Vittorio. Angry
crowds: Detecting violent events in videos. In ECCV, 2016.

[31] H. Rabiee, J. Haddadnia, H. Mousavi, M. Kalantarzadeh,
M. Nabi, and V. Murino. Novel dataset for fine-grained
abnormal behavior understanding in crowd. In 2016 13th
IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), 2016.

[32] I. Saleemi, K. Shafique, and M. Shah. Probabilistic modeling of scene dynamics for applications in visual surveillance.
TPAMI, 31(8):1472–1485, 2009.

[33] A. Sankaranarayanan, S. Alavi and R. Chellappa. Triplet
similarity embedding for face verification. arXiv preprint
arXiv:1602.03418, 2016.

[34] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and
R. Salakhutdinov. Dropout: A simple way to prevent neural
networks from overfitting. J. Mach. Learn. Res., 2014.

[35] W. Sultani and J. Y. Choi. Abnormal traffic detection using
intelligent driver model. In ICPR, 2010.

[36] Theano Development Team. Theano: A Python framework
for fast computation of mathematical expressions. arXiv
preprint arXiv:1605.02688, 2016.

[37] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri.
Learning spatiotemporal features with 3d convolutional networks. In ICCV, 2015.

[38] J. Wang, Y. Song, T. Leung, C. Rosenberg, J. Wang,
J. Philbin, B. Chen, and Y. Wu. Learning fine-grained image similarity with deep ranking. In CVPR, 2014.

[39] S. Wu, B. E. Moore, and M. Shah. Chaotic invariants
of lagrangian particle trajectories for anomaly detection in
crowded scenes. In CVPR, 2010.

[40] D. Xu, E. Ricci, Y. Yan, J. Song, and N. Sebe. Learning
deep representations of appearance and motion for anomalous event detection. In BMVC, 2015.

[41] T. Yao, T. Mei, and Y. Rui. Highlight detection with pairwise
deep ranking for first-person video summarization. In CVPR,
June 2016.

[42] B. Zhao, L. Fei-Fei, and E. P. Xing. Online detection of unusual events in videos via dynamic sparse coding. In CVPR,
2011.

[43] Y. Zhu, I. M. Nayak, and A. K. Roy-Chowdhury. Contextaware activity recognition and anomaly detection in video. In
IEEE Journal of Selected Topics in Signal Processing, 2013.
