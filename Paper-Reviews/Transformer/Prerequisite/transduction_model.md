# Transduction (in Machine Learning)

induction과 대조적인 개념. 연역적 추론? 음... deductive라는 용어가 있잖아.
- 아 deductive맞네
- 아니네, 완전히 다르네.

보통 교육학에서는 `전환적 학습`이라는 용어를 `transformative learning`의 번역으로 사용하고 `전이 학습`이라는 용어는 일반적으로 `transfer learning`의 번역, `전도`는 `conduction`의 번역으로 사용되고 `transduction`은 `변환` 정도로 해석이 되면 좋음.

## 1. Wikipedia: Transduction (machine learning)
논리, 통계적 추론 및 지도학습에서 `Transduction`은 관찰된 특정 (학습) 데이터에서 특정 (테스트) 데이터를 추론하는 것.
`Induction`은 말그대로 귀납적으로 학습 데이터에서 테스트 데이터에 적용할 일반적인 규칙을 추론.
이 차이는 `Transduction`의 추론이 `Induction`의 추론으로는 얻어질 수 없는 경우 매우 흥미롭게 작용함.
Note that: `Transductive inference`는 서로 다른 test set에 대해 일치하지 않는 예측값을 제공함

`Transduction`은 1990년대 _Vladimir Vapnik_ 에 의해 소개됨.
그에 따르면 `Induction`은 특정 문제를 풀기 전에 더 일반적인 문제를 풀어야 함.
때문에 `Transduction`가 더 유용하다고 주장.

> 관심있는 문제를 풀 때 중간 단계로 일반적인 문제를 풀지 말라. 일반적인 답이 아닌 너가 정말 필요로 하는 정답을 얻으려 노력하라.

_Bertrand Russell_ 도 이와 비슷한 관찰을 했다.

> 우리는 '모든 사람은 죽는다'고 생각하고 연역적인 추론을 사용하는 것보다 순전히 귀납적인 주장을 한다면 더 확실하게 '소크라테스도 죽는다'는 결론에 도달할 수 있을 것이다.
(Russell 1912, chap VII)

귀납적이지 않은 학습의 예시는 입력값을 두 개의 클러스터로 나눠야하는 이진 분류이다.
대량의 테스트 데이터 셋은 해당 클러스터를 찾는데 도움을 줄 것이고 레이블링을 하는데 더 유용한 정보를 제공할 것이다. 학습 데이터에만 의존하여 함수를 유도(induce)하는 모델로 부터는 같은 예측값을 얻을 수 없다(ex> SVM). 몇몇 사람들은 `Vapnik`의 동기와 조금 다르기 때문에 이를 `semi-supervised learning`이라고 부르기도 한다. 이 항목의 한 가지 예시는 `Transductive Support Vector Machine(TSVM)`이다.

`Transduction`으로 이어지는 세 번째 동기는 `approximation`에 대한 필요로 발생한다. 만약 정확한 추론이 계산적으로 불가능하다면, 어떤 이는 적어도 근사치가 테스트 셋에 적합하다는 것을 확인하려고 시도할 수 있다. 이 경우 테스트 셋은 준지도학습에선 허용되지 않는 임의의 분포(학습 데이터의 분포와 반드시 관련이 있는 것은 아닌)에서 왔을 수 있다. 이 범주에 속하는 알고리즘은 `Bayesian Committe Machine(BCM)`이 있다.

### Example problem
아래 예제는 귀납법(`induction`)과 `transduction`의 성질의 차이를 보여줌

![img](https://upload.wikimedia.org/wikipedia/en/1/19/Labels.png)

A, B, C라고 labling된 어떠한 점과 ?로 unlabeled된 점들의 모임이 주어졌다. 목표는 모든 labling되지 않은 point에 대해 적절한 예측값을 얻는 것.

`Induction` 접근 방법은 `supervised learning` 알고리즘으로 labeled된 점을 학습하고 모든 unlabeled 점에 대한 예측 label을 얻는 것. 그러나 문제는 supervised learning algorithm으로 풀면 오직 5개의 data point밖에 얻지 못한다. 이는 전체적인 데이터 구조를 파악할 모델을 구축할 때 확실히 어려움을 겪을 것이다. 예를 들어 만약 nearest neighbor 알고리즘을 사용한다면 한 가운데 있는 점은 "B"가 아닌 A"와 "C" 중 하나로 예측될 것이다.

`Transduction`은 label된, 혹은 label되지 않은 모든 점들을 고려할 수 있다. 이 경우, `Transduction` 알고리즘은 자연스럽게 속하는 클러스터에 따라 label이 지정되지 않은 점을 레이블링한다. 그러므로 한 중앙에 있는 점은 "B"쪽 클러스터에 가깝게 위치하고 있기 때문에 "B"라고 labeling될 가능성이 높다.

`Transduction`의 장점은 라벨이 없는 점에서 발견되는 natural breaks를 사용하기 때문에 라벨이 지정된 점이 적을수록 더 나은 예측을 할 수 있다.
그러나 `Transduction`의 단점은 모델을 구축하지 않는다는 점이다. 데이터가 stream을 통해 점진적으로 증가되면 cost가 많이 들 수 있다.

### Transduction algorithms
two categories
- label이 지정되지 않은 점에 discrete label을 할당
- label이 지정되지 않은 점에 continuous label을 회귀

전자는 clustering algorithms에 부분적인 지도학습을 추가하여 파생되기도 함
이들은 두 가지 범주로 더 세분화될 수 있음
- Partitioning(분할)
- Agglomerating(응집)

후자는 `manifold learning` 알고리즘에 부분적인 지도학습을 추가하여 파생

#### Partitioning Transduction
Top-Down transduction, 파티션 기반 클러스터링의 준 지도학습
```
모든 점의 집합을 하나의 큰 파티션으로 간주함
while 어떤 파티션 P가 레이블이 충돌하는 두 개의 포인트를 가짐:
    파티션 P를 더 작은 파티션으로 분할
for 각 파티션 P:
    동일한 레이블을 P의 모든 포인트에 할당
```
`Max flow min cut` 등의 partitioning 기법도 위와 함께 사용될 수 있음

#### Agglomerative Transduction
Bottom-Up transduction, 응집 클러스터링의 준 지도학습법 확장
```
모든 점들 사이의 pair-wise 거리 D를 계산
D를 오름차순으로 정렬
각 포인트들을 크기가 1인 클러스터로 간주
for 각 포인트 쌍 {a, b} in D:
    if (a가 레이블이 지정되지 않음) or (b가 레이블이 지정되지 않음) or (a와 b가 동일한 레이블을 가짐):
        a와 b를 포함하는 두 클러스터를 병합
        병합된 클러스터의 모든 점에 동일한 레이블을 할당
```

#### Maniford Transduction
여전히 매우 신생의 연구 분야 (Transformer도 이쪽이지!)

## 2. Gentle Introduction to Transduction in Machine Learning

### What is Transduction?

- 기본 사전적인 정의
>변환 : (에너지 또는 메시지와 같은) 다른 형태로 전환하는 것은 본질적으로 감각 기관으로 물리적 에너지를 신경 신호로 변환합니다.
Merriam-Webster Dictionary (online), 2017

- 전자 및 신호 처리 분야, `Transducer`는 소리를 에너지 또는 그 반대로 변환하는 구성 요소 또는 모듈의 일반적인 이름인 전자 및 신호 처리 분야에서 널리 사용되는 용어
>모든 신호 처리는 입력 변환기로 시작합니다. 입력 변환기는 입력 신호를 가져 와서 전기 신호로 변환합니다. 신호 처리 응용 프로그램에서 변환기는 여러 가지 형태를 취할 수 있습니다. 입력 변환기의 일반적인 예는 마이크입니다.
Digital Signal Processing Demystified, 1997

- 생물학, 유전학에서 `Transduction`은 유전 물질을 한 미생물에서 다른 미생물로 옮기는 프로세스를 의미
>변환 : 변환의 작용 또는 과정; 특히 : 바이러스 성 물질 (예 : 박테리오파지)에 의해한 미생물에서 다른 미생물로 유전 물질을 전달하는 것
Merriam-Webster Dictionary (online), 2017

자, `Transduction`은 일반적으로 신호를 다른 형태로 변환하는 것을 의미.
신호 처리 설명은 음파가 시스템 내에서 일부 용도로 전기 에너지로 전환되는 가장 두드러진 부분
각 사운드는 일부 선택된 샘플링 수준에서 전자적 특성으로 표시
![img](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Example-of-Signal-Processing-Transducer.png)

### Transductive Learning
transduction 또는 transductive learning은 통계 학습 이론 분야에서 도메인 내의 주어진 특정 예제(example)를 이용해 다른 특정 예제를 예측하는 것을 설명하기 위해 사용됩니다. 이는 귀납적 학습 및 연역적 학습과 등의 다른 유형의 학습과 대조됩니다.

>귀납(induction), 주어진 데이터에서 함수를 유도합니다. 연역(deduction), 관심있는 점에 대해 주어진 함수의 값을 유도합니다. 변환(transduction), 주어진 데이터에서 관심있는 점에 대해 알려지지 않은 함수의 값을 유도합니다.
Page 169, The Nature of Statistical Learning Theory》, 1995

![img](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Relationship-between-Induction-Deduction-and-Transduction.png)

이는 “데이터로부터 매핑 함수를 근사하고 예측을 위해 그것을 사용하는” 고전적인 문제가 필요 이상으로 어려워 보이는 지도 학습(supervised learning)에 반하는 흥미로운 프레임.
대신 특정 예측은 도메인의 실제 샘플에서 직접 수행된다.
여기서는 함수 근사(function approximation)가 필요하지 않다.

>주어진 관심 지점에서 함수의 값을 추정하는 모델은 추론의 새로운 개념을 설명합니다. 즉, 특정 지점에서 특정 지점으로 이동합니다. 우리는 이런 종류의 추론을 전도성 추론(transductive inference)이라고 부릅니다. 제한된 양의 정보에서 가장 좋은 결과를 얻고 싶을 때 이러한 추론 개념이 나타난다는 사실에 주목하세요.
Page 169, The Nature of Statistical Learning Theory》, 1995

`Transductive` 알고리즘의 고전적인 예는 학습 데이터를 모델링하지 않고 예측이 필요할 때마다 이를 직접 사용하는 k-Nearest Neighbors(k-NN) 알고리즘입니다.

>transduction은 인스턴스 기반 또는 사례 기반 학습으로 알려진 일련의 알고리즘과 자연스럽게 관련됩니다. 아마도 이 부류에서 가장 잘 알려진 알고리즘은 k-NN 알고리즘일 것입니다.
Learning by Transduction, 1998

### Transduction in Linguistics
고전적으로, 언어학 분야와 같이 자연어(natural language)에 대해 이야기 할 때 transduction이 사용되어 왔습니다. 예를 들어 한 언어를 다른 언어로 변환하기 위한 일련의 규칙을 나타내는 “변환 문법(transduction grammar)”이라는 개념이 있습니다.

>변환 문법(transduction grammar)은 구조적으로 상관된 언어 쌍을 설명합니다. 이는 단일 문장이 아닌 문장 쌍(pair)을 생성합니다. 1번 언어의 문장은 (의도에 따르면) 2번 언어 문장의 번역입니다.
Page 460, Handbook of Natural Language Processing, 2000.

또한 한 세트의 기호를 다른 기호에 매핑하기 위한 번역 task에 대해 이야기 할 때 언급되곤 하는 계산 이론에서의 “유한 상태 변환기”(FST; Finite State Transducer)라는 개념이 있습니다. 중요한 것은, 각각의 입력이 하나의 출력을 생성한다는 것입니다.

>유한 상태 변환기(finite state transducer)는 여러 개의 state로 구성됩니다. 상태 간 전환시 입력 기호가 소비되고 출력 기호가 방출됩니다.
Page 294, Statistical Machine Translation, 2010.

이론과 고전 기계 번역에 대해 이야기할 때의 transduction의 사용은 NLP task에 RNN을 이용하는 오늘날 시퀀스 예측에서의 쓰임과 의미가 많이 달라, 이 용어의 사용에 부정적인 영향을 미칠 수 있습니다.

### Transduction in Sequence Prediction
Yoav Goldberg는 언어처리를 위한 신경망에 관한 그의 교과서에서 트랜스듀서(변환기)를 NLP task를 위한 특정 네트워크 모델로 정의했습니다. 트랜스듀서는 좁게 정의한다면 제공된 각 input time step에 대한 하나의 time step(output)을 출력하는 모델이라 할 수 있습니다. 이것은 특히 유한 상태 변환기(finite state transducer)와 함께 언어적 사용법까지 연결됩니다.

>또다른 옵션은 RNN을, 읽어들이는 각 입력(input)에 대한 출력(output)을 생성하는 변환기로써 바라보는 것입니다.
Page 168, Neural Network Methods in Natural Language Processing, 2017.

그는 언어 모델링뿐만 아니라 시퀀스 태깅을 위한 이러한 유형의 모델을 제안하였는데, 또한 인코더-디코더 아키텍처와 같은 조건부 생성(conditioned generation)이 RNN 트랜스듀서의 특별한 케이스로 간주 될 수 있음을 지적합니다. 이 마지막 부분은 인코더-디코더 모델 아키텍처의 디코더가 주어진 입력 시퀀스에 대해 다양한 개수의 출력을 허용하여 기존 정의에서의 “입력 당 하나의 출력”을 깬다는 점에서 놀랍습니다.
![img](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Transducer-RNN-Training-Graph.png)

보다 일반적으로 transduction은 NLP 시퀀스 예측 작업, 특히 번역에 사용됩니다. 이 정의는 Goldberg와 FST의 엄격한 “입력 당 하나의 출력”보다 더 받아들이기 수월합니다. 예를 들어 Ed Grefenstette, et al.은 transduction을 입력 문자열을 출력 문자열에 매핑하는 것으로 설명합니다.

>많은 자연어 처리 (NLP) task는 하나의 문자열을 다른 문자열로 변환하는 것을 배우는 transduction 문제로 볼 수 있습니다. 기계 번역은 transduction의 전형적인 예이며, 최근의 연구 결과들은 Deep RNN이 긴 원본 문자열을 인코딩하여 일관성 있는 번역문을 생성할 수 있음을 보여줍니다.
Learning to Transduce with Unbounded Memory, 2015.

그들은 이 광범위한 정의를 구체적으로 만드는 데 도움이 되는 특정 NLP task 목록을 제공합니다.

>문자열 transduction은 이름 음차 및 철자 수정에서 굴절 형태학 및 기계 번역에 이르기까지 NLP의 많은 응용 분야에서의 핵심입니다.

또한 Alex Graves는 ‘transduction’을 ‘transformation’의 동의어로 사용하며, 이러한 정의를 충족시키는 유용한 예제 NLP task 목록을 제공합니다.

>몇 가지 예를 들자면 많은 기계 학습 task들은 다음과 같은 입력 시퀀스로부터 출력 시퀀스로의 transformation - 또는 transduction - 으로 표현 될 수 있습니다: 음성인식, 기계 번역, 단백질 2차 구조 예측 및 TTS 등
Sequence Transduction with Recurrent Neural Networks, 2012.

요약하면 다음과 같이 transductive 자연어처리 task 목록을 다시 작성할 수 있습니다.

- 음역(transliteration): 소스 형식의 예제에 따라 대상 형식으로 단어를 생성
- 철자 수정(spelling correction): 주어진 잘못된 단어 철자에서 올바른 단어 철자를 생성
- 굴절 형태학(inflectional morphology): 소스 시퀀스와 컨텍스트가 주어진 새로운 시퀀스를 생성
- 기계번역: 소스 언어로된 예제에서 대상 언어로 단어 시퀀스를 생성
- 음성인식: 주어진 오디오 시퀀스로 텍스트 시퀀스 생성
- 단백질 2차 구조 예측: 아미노산의 입력 서열(NLP가 아닌)이 주어진 3D 구조를 예측
- TTS(Test-To-Speech) 또는 음성 합성으로 오디오 주어진 텍스트 시퀀스를 생성

마지막으로, 광범위한 NLP 문제와 RNN 시퀀스 예측 모델을 언급하는 transduction 개념 외에도 일부 새로운 방법론들이 명시적으로 명명되고 있습니다. Navdeep Jaitly, et al.은 기술적으로 sequence-to-sequence 예측을 위한 RNN이 될 “뉴럴 트랜스듀서”로서 새로운 RNN sequence-to-sequence 예측 방법을 언급합니다.

>우리는 seq-to-seq 학습 모델보다 일반적인 종류인 ‘뉴럴 트랜스듀서’를 제시합니다. 뉴럴 트랜스듀서는 입력 블록이 도착하면 출력 chunk(길이가 0일 수 있는)를 생성 할 수 있으므로 “온라인”상태를 만족시킵니다. 이 모델은 seq-to-seq 모델을 구현하는 트랜스듀서 RNN을 사용하여 각 블록에 대한 출력을 생성합니다.
A Neural Transducer, 2016

### Further Reading

#### Definition
- [Merriam-Webster Dictionary definition of transduce](https://www.merriam-webster.com/dictionary/transduce)
- [Digital Signal Processing Demystified, 1997](https://www.amazon.com/Digital-Signal-Processing-Demystified-Engineering/dp/1878707167/ref=as_li_ss_tl?ie=UTF8&qid=1501051382&sr=8-1&keywords=Digital+Signal+Processing+Demystified&linkCode=sl1&tag=inspiredalgor-20&linkId=e01b9d0a6e351f2de9ad271aefa428ab)
- [Transduction in Genetics on Wikipedia](https://en.wikipedia.org/wiki/Transduction_(genetics))

#### Learning Theory
- [The Nature of Statistical Learning Theory, 1995](https://www.amazon.com/Statistical-Learning-Theory-Vladimir-Vapnik/dp/0471030031/ref=as_li_ss_tl?ie=UTF8&qid=1501052308&sr=8-1&keywords=Statistical+Learning+Theory&linkCode=sl1&tag=inspiredalgor-20&linkId=b323b9c69c842d517ae7339469261e1b)
- [Learning by Transduction, 1998](https://arxiv.org/abs/1301.7375)
- [Transduction (machine learning) on Wikipedia](https://en.wikipedia.org/wiki/Transduction_(machine_learning))

#### Linguistics
- [Handbook of Natural Language Processing, 2000.](https://www.amazon.com/Handbook-Language-Processing-Chemical-Industries/dp/0824790006/ref=as_li_ss_tl?ie=UTF8&qid=1501108861&sr=8-3&keywords=%22Handbook+of+Natural+Language+Processing%22&linkCode=sl1&tag=inspiredalgor-20&linkId=dd3886f75bad4dd2a4f7b948703d9986)
- [Finite-state transducer on Wikipedia](https://en.wikipedia.org/wiki/Finite-state_transducer)
- [Statistical Machine Translation, 2010](https://www.amazon.com/Statistical-Machine-Translation-Philipp-Koehn/dp/0521874157/ref=as_li_ss_tl?ie=UTF8&qid=1501109077&sr=8-1&keywords=%22Statistical+Machine+Translation%22&linkCode=sl1&tag=inspiredalgor-20&linkId=2055ba413dd072a48142ff3cb748cf2c)

#### Sequence Prediction
- [Neural Network Methods in Natural Language Processing, 2017.](https://www.amazon.com/Language-Processing-Synthesis-Lectures-Technologies/dp/1627052984/ref=as_li_ss_tl?ie=UTF8&qid=1501109351&sr=8-1&keywords=Neural+Network+Methods+in+Natural+Language+Processing&linkCode=sl1&tag=inspiredalgor-20&linkId=5beb166aeec7455c18d441e0e6d7b62d)
- [Learning to Transduce with Unbounded Memory, 2015.](https://arxiv.org/abs/1506.02516)
- [Sequence Transduction with Recurrent Neural Networks, 2012.](https://arxiv.org/abs/1211.3711)
- [A Neural Transducer, 2016](https://arxiv.org/abs/1511.04868)


출처:
- https://en.wikipedia.org/wiki/Transduction_(machine_learning)
- https://dos-tacos.github.io/translation/transductive-learning/
- https://machinelearningmastery.com/transduction-in-machine-learning/
