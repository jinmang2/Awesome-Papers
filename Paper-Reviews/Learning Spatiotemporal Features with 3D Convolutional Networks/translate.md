# Learning Spatiotemporal Features with 3D Convolutional Networks
- 7 Oct 2015
- Facebook AI Research

## Abstract
- large-scale supervised video dataset을 훈련시키기 위한 Deep 3D ConvNets을 사용한 spatiotemporal feature 학습 방법을 제시
    - Spatiotemporal? 시공간!
- 본 논문에서 밝힌 내용들을 요약하면
    1. 3D ConvNets은 2D ConvNets보다 spatiotemporal features를 학습하는데 더 뛰어남
    2. homogeneous architecture(3 X 3 X 3 conv kernels)이 최상의 성적을 거뒀고 이를 C3D로 명명
        - SOTA지롱 짱짱맨!
        - Features is compact
        - 10-dimenstion feature로 UCF101에서 52.8%의 정확도를 얻었고
        - inference 속도가 매우 빨라 효과적
    3. 개념적으로 단순하고 학습에 적용시키기가 용이
    
## 1. Introduction
- Internet Multimedia의 발전은 급속도로 진행되고 있으며 이를 효과적으로 이해하고 분석하는 것은 필수적인 일
- Computer Vision community는 이러한 분석을 진행해왔고 아래와 같은 task를 풀어왔음
    - action recognition [26]
    - abnormal event detection [2]
    - activity understanding [23]
- 해당 문제들에는 개별 문제당 각기 다른 solution을 사용해옴
- 그러나, 대규모 video 작업을 homogeneous하게 해결할 generic video descriptor에 대한 수요가 증가하고 있음
- 효과적인 video descriptor에 대한 네 가지 특징들
    1. **generic**, 일반적이어야 함
        - 비디오들을 판별하면서 다른 타입의 video를 구분질 수 있어야 함
        - Internet videos: landscapes, natural scenes, sports, TV shows, movies, pets, food and so on
    2. **compact**, 간략해야 함
        - 수백만 개의 비디오로 작업 할 때 compact descriptor는 훨씬 확장 성이 뛰어난 작업을 처리, 저장 및 검색하는 데 도움이 됨
    3. **efficient**, 효율적이어야 함(계산에)
        - 실세계에서 매 분마다 수천개의 비디어를 처리해야하기 때문에 계산이 효율적이어야 함
    4. **simple to implement**, 구현이 쉬워야 함
        - 정교한 feature encoding method, classifier를 사용하는 대신에,
        - 좋은 descriptor는 간단한 모델로 작동해야 한다. (e.g. linear classifier)
        
- 과거보다 훨씬 빨리 일을 처리하는 deep learning에 영감을 받아[24] image feature를 추출하는데 다양한 pre-trained ConvNet 모델들이 만들어짐[16]
- Feature는 Transfer learning task에서 수행되는 마지막 Fully connected layer에 activation을 씌운 값 [47, 48]
- 그러나, 몇몇 image based deep features들은 motion modeling에서 video feature로 적합하지 않음
- 이 논문에서 video에 적합한 feature (Spatio + Temporal)를 추출하는 Deep 3D ConvNet을 제안하겠슴
- 본 논문에서 경험적으로 밝혀낸 것은, 다양한 video 분석 task들에서 간단한 linear classifier를 활용한 feature가 좋은 performance를 내고 있다는 것
- 비록 이전에 3D ConvNets이 제안되었지만[15, 18], large-scale 데이터셋에서 3D feature를 추출하는 방법과 video 분석 task들의 각기 다른 타입에 대해 최상의 performance를 얻을 수 있는 최신의 deep architectures를 제시하고자 함
- 3D ConvNet에서 추출된 feature는 객체, 장면, 행동 등을 함축한 정보를 포함하고 있으며, 각 task에 finetune을 요구하지 않고 다양한 task에서 유용하게 사용될 수 있음
- C3D는 좋은 descriptor가 가져야할 generic, compact, simple, efficient를 모두 가지고 있다! 우리 짱짱맨!@
- 요약하자면,
    - 3D ConvNets이 모델 표현력과 motion을 동시에 학습하는 좋은 방법론이라는 것을 경험적으로 밝혀냄
    - 한정된 architecture들 가운데 3 X 3 X 3 kernel이 가장 좋다는 것을 또 경험적으로 밝혀냄
    - 4개의 task와 6개의 benchmark에서 뛰어나거나 현재 최고의 성능을 따라잡은 성과를 보여줌. 계산도 빨라! 짱이지?
    
## 2. Related Work
- pass

## 3. Learning Features with 3D ConvNets
- 3D ConvNets의 기본적인 연산을 설명
- 다른 3D ConvNets 구조를 분석
- 어떻게 학습시켰는지 설명

![c3d_fig1](https://user-images.githubusercontent.com/37775784/75955754-d7496300-5ef9-11ea-9d4c-3c15756cc135.PNG)

### 3.1 3D convolution and pooling
#### 3D Convnet is Good!
- 3D ConvNet은 2D ConvNet과 비교했을 때 컨벌루션 및 3D 풀링 작업으로 인해 시간 정보를 더 잘 모델링 할 수 있음
- 2D ConvNet이 Spatial 정보만 접근할 때 3D ConvNet은 시공간 정보에 접근
- Image, Multi-Image에 적용된 2D convolution의 output은 모두 image이다.
- 즉, 2D ConvNet은 convolution 연산을 거친 후에 시간정보를 잃어버린다
- 오직 3D ConvNet만이 이를 보존할 수 있다.
- 똑같은 얘기가 2D와 3D pooling에도 적용된다.
- [36]에서 보면 2D Convolution에 다중 frame을 input으로 넣어주지만 convolution layer를 지난 후에 시간 정보는 압축되서 사라진다.
- 유사하게, 2D convnet을 사용한 [18]의 fusion 모델 또한 첫번째 conv layer에서 시간 정보를 손실한다.
- [18]의 Slow Fusion 모델만이 3D Convnet과 averaging pooling을 첫번째 layer에서 사용한다.
- [18] 논문에서 성능의 차이가 나는 것을 2D와 3D라고 저자는 유추하지만 3번째 conv layer에서 여전히 시간정보 손실이 일어난다.

#### Empirical Try
- 3D ConvNet의 최상의 architecture를 찾는 시도를 기록
- large-scale에서 실험하기엔 시간적 소요가 크기 때문에 medium-scale dataset인 UCF101로 최적의 architecture를 찾는 실험을 했다.
- 2D Conv에 따르면[37], 3 X 3 kernel의 작은 receptive field가 가장 최상의 결과를 보였다.
- 즉, 본 논문에서 spatial receptive field를 3 X 3으로 고정하고 temporal depth를 변화시켜서 실험

#### Notations:
- Video clips size: c X l X h X w
    - c: channels
    - l: length in number of frames
    - h, w: height and width of the frame
- kernel size: d X k X k
    - d: kernel temporal depth
    - k: kernel spatial size

#### Common network settings
- video clips를 input으로 101개의 action에 대한 class label을 예측
- 모든 video frame은 128 X 171로 re-size (이는 UCF101의 1/2 resolution(해상도))
- video들을 network의 input으로 사용될 때 겹치지 않게 16-frame clip으로 나눔
- input dimension: 3 X 16 X 128 X 171
- random crop을 사용, 학습 도중에 input clip의 3 X 16 X 112 X 112의 크기로 jittering 실시
- 5 convolution layer와 5 pooling layer를 가지고 2개의 FC layer, 마지막에 softmax los slayer가 action을 예측하는데 들어감
    - 각 conv layer다음엔 바로 pooling layer가 따라옴
- 5 conv layer의 filter의 수는 64, 128, 256, 256, 256
- 모든 conv kernel은 d의 size를 가짐 (d는 kernel temporal depth)
- 모든 conv layer에는 적절한 padding을 적용(both spatial and temporal)
- stride=1, that is input에서 output으로 어떠한 size의 변화도 없음
- 모든 pooling layer는 max pooling으로 kernel size는 2 x 2 X 2 (1st layer 제외), stride=1
    - 입력 신호에 대비 8개의 factor가 소거됨
- 1st pooling layer의 kernel size는 1 X 2 X 2
    - 너무 초기에 temporal signal을 합쳐버리는 것을 방지
    - 16 frame의 clip length를 만족시키기 위해
    - `e.g. we can temporally pool with factor 2 at most 4 times before compoletely collapsing the temporal signal`
    - 음... pool에서 factor가 의미하는 바가 뭘까?
- 두 FC layer는 2,048개의 output을 가짐
- 30 clips의 mini-batches로 훈련을 시켰고 learning rate는 3e-3
- learning rate는 4 epochs마다 10으로 나눠줌
- 총 16 epochs를 돌려서 학습을 끝마침

#### Varing network architectures
- 논문의 목적은 deep network를 통해 termporal information을 얼마나 catch하느냐
- 최상의 3D ConvNet architecture를 찾기 위해, 위에서 기술한 모든 조건을 고정시키고 temporal depth d_i만을 변화시킴
- 논문에선 아래 두 architecture를 실험
    1. Homogeneouis temporal depth
        - 모든 conv layer는 동일한 kernel temporal depth를 가짐
        - d를 1, 3, 5, 7로 설정, 4개의 network로 실험
        - 이 network를 **depth-d**라고 명명
        - d는 homogeneous termporal depth
    2. Varying temporal depth
        - layer를 지남에 따라 kernel temporal depth가 달라진다
        - Increasing, Decreasing의 두 network로 실험 (1st->5th)
            - Increasing: 3-3-5-5-7
            - Decreasing: 7-5-5-3-3
- Note that: 마지막 pooling layer를 지나면 위 모든 구조는 결국 같은 size의 output을 가짐
    - FC layer의 parameter의 수는 모두 똑같음
- parameters의 수는 conv layer에서만 달라진다
- architecture별로 달라지는 parameter의 수는 FC layer의 paremeter수에 비해 굉장히 미미하다.
- depth-1이 17K, depth-7이 51K의 parameter를 가진데 비해 전체 파라미터는 17.5 millions로 앞선 conv layer의 param은 전체의 0.3%에 불과하다.
    - 실제로 그런지 torch로 확인하자!
- 이는 network의 learning capacity(용량, 능력)이 비슷하다는 것을 의미하며 architecture별 params의 차이는 큰 의미가 없다는 것을 의미!

### 3.2 Exploring kernel temporal depth
- 저자는 UCF101의 train split 1으로 학습을 시켰다. (kernel temporal depth)<br>
- 해당 결과는 아래의 plot과 같다.

<img src="https://icbcbicc.github.io/img/1.JPG" width=70% height=70%>

- 왼쪽의 plot은 동일한(homogeneous) temporl depth를 가지는 networks의 결과를,
- 오른쪽의 plot은 kernel temporal depth를 변화시킨 networks의 결과를 시각화한 것이다.
    1. *homo_depth-3*의 performance가 제일 좋았다.
    2. *homo_depth-1*는 각각의 frame에 2D convolution을 적용한 것과 동일하다고 말했다.
        위의 motion modeling의 부재 때문에(sequential 고려 X) 성능이 좋게 나오지 않았다고 밝힌다.
- 또한 저자는 bigger spatial receptive field(e.g. 5X5) 와/혹은 전체 해상도(full resolution, 240X320 fram inputs)으로 실험하고 유사 동작을 관찰했다.
- 이에 대한 결과로 저자는 3X3X3이 3DConvNets의 최상의 kernel이라고 제시하고 비디오 분류 문제에서 2DConv보다 더 우월한 성능을 제공한다고 말한다.
- 또한 I380K라 불러닌 large-scale internal dataset에서 3DConvNet이 2DConvNet보다 더 우월한 성능을 보임을 검증했다.

### 3.3 Spatiotemporal feature learning

#### Network architecture
- 3D ConvNet에서 3 X 3 X 3의 homogeneous setting이 최상의 결과를 보였음 ([37] 2D Convnet의 결과와 유사)
- Large-Scale dataset에서 homo_depth-3의 kernel로 3D CovnNet을 memory와 계산 능력에 따라 학습시키는 것이 가능
- large dataset에 대해 8개의 convnet과 5개의 pooling layer, 2개의 FC layer와 softmax output layer를 사용

![c3d_fig3](https://user-images.githubusercontent.com/37775784/75989384-8441d100-5f36-11ea-9669-a8c379617362.PNG)

- 위의 Network를 C3D라고 명명
- 모든 3D conv filter는 3 x 3 x 3 with stride 1 X 1 X 1
- 모든 3D pooling layer는 2 X @2 X 2 with stride 2 X 2 X 2
    - 첫번째 pooling layer만 제외! kernel size 1 X 2 X 2 and stride 1 X 2 x 2
    - 초기에 시간 정보를 압축하는 현상을 방지하기 위해

#### Dataset
- Spatiotemporal feature를 학습하기 위해 Sports-1M dataset[18]에 C3D를 학습
    - 현재 가장 큰 video classification benchmark
- 이 dataset은 110만개의 sports video를 포함
- 각 비디오는 487개의 sport category 중 하나를 따름
- UCF101과 비교했을 때 Sports-1M은 5배 가량 class의 숫자가 많고 video의 숫자는 100배정도 더 많음

#### Training 
- Training은 Sports-1M train split에서 진행
- Sports-1M의 video의 길이가 길기 때문에 모든 training video의 5개의 2초 길이의 clip을 임의로 추출
- Clip들은 128 X 171로 resize
- 학습 중, 시공간 jittering을 위해 input clip을 16 X 112 X 112로 cropping 실시
- 50%의 확률로 수평적으로 flippint(뒤집기) 실시 (y축 대칭)
- 학습은 mini-batch size 30으로 SGD 알고리즘으로 진행
- 초기 learning rate는 3e-3, 150K iteration마다 2로 나눠줌
- 











