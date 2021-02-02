# ShallowMinded
딥마인드 블로그 포스팅 읽기 [google sheet](https://docs.google.com/spreadsheets/d/1tw2tE6Rwag38DPyCdrKq3GcbZaJ9O9Na0k5hN6A0amY/edit#gid=0)

## Read posts

### [20.10.05] Open-sourcing DeepMind Lab (16.12.03)
- 딥마인드의 오픈소스 랩에 대한 설명

### [20.10.05] Our collaborations with academia to advance the field of AI (17.01.23)
- 학계와 협력하는 최고의 스타트업. 하사비스의 철학 또한 다시 엿볼 수 있었음

### [20.10.14] Understanding Agent Cooperation (17.02.09)
- 이기적인 multi-agent 채용, game theorey, prisoner's dillemma
- [rational agent](https://en.wikipedia.org/wiki/Rational_agent): 이기적인 agent. 변수 혹은 변수들의 함수의 기댓값(uncertainty)과 같은 명백한 목표를 가지고 있으며 자신이 취할 수 있는 행동들 (feasible actions) 중 최상의 결과만을 취할 행동을 택한다.
  - 주로 인지 과학, 윤리, 실용주의 철학 등에서 연구된다고 한다.
  - 합리적인 판단 -> selfish하게 비춰질 수 있으나, 명확한 목적을 가지고 행동하는 봇이라고 이해하면 될 듯 하다
- Deep Multi-Agent Reinforcement Learning


**영어 표현**
- agents' cognitive capacity
- why should this be the case
  - https://english.stackexchange.com/questions/257538/what-does-this-mean-why-would-that-be-the-case
  - is same to
    - `why would that be the case?`
    - `why would that by true?`
    - `what leads you to believe that that is true?`
- entice: 꾀다
- testify: 증언하다
- no matter what: ~할 지라도
- roam: 배회하다

### [20.11.11] DeepMind papers at ICML 2017 (part one) (17.08.04)

[**Sharp Minima Can Generalize For Deep Nets**](https://arxiv.org/abs/1703.04933)
- 실험적으로 Batch 방법론보다 SGD가 일반화를 더 잘하는 것처럼 보임
  - 한가지 가설: SGD의 noise 가 sharpe한 minima보다 narrow(flat)한 minima를 더 잘 찾는다.
- 위 가설에 대한 연구를 진행, Neural Network(특히 rectified)의 구조 때문에 Sharp한지 Flat한지 여부로 일반화 여부를 결정짓긴 어려움
  - Sharp한지 Flat한지에 대한 정의는 Schmidhuber의 Flat Minima 논문 기타 등을 참조함
- 즉, Batch size와 Generalization 사이에 인과관계는 없음

### [20.11.10]

### [20.11.17] The hippocampus as a predictive map (17.10.02)

### [20.11.30] [Population based training of neural networks](https://github.com/jinmang2/Awesome-Papers/blob/master/ShallowMinded/201130_PBT.md) (17.11.27)
- Random Search와 Hand-Tuning의 hybrid 모델
- 비동기적으로 Exploitation와 Exploration을 수행하는 worker를 도입하여 빠르게 최적의 hyperparamter를 찾는다.
- DM Lab, Atari, 스타2, 기계 번역, GAN 등에서 SOTA를 뛰어넘었다.

### [20.12.01] [Specifying AI safety problems in simple environments](https://github.com/jinmang2/Awesome-Papers/blob/master/ShallowMinded/201201_SafetyAI.md) (17.11.28)
- AI의 발전과 더불어 Safety AI의 발전도 가속화되나 기존엔 Unsafe한 행동의 본질과 원인에 대한 이론적인 이해에 치중했었음
- 본 논문에서 소개하는 9가지 `gridworlds` 환경은 이러한 Safety AI를 test할 기본적인 환경이고 이 중 3가지 환경에 대해 A2C, Rainbow-DQN으로 실험해봤으나 design 문제로 잘 동작하지 않았음
- 위 실패를 밑거름으로 이러한 실용적인 환경 개발과 더불어 Safety AI 개발에 더욱 착수할 

### [20.12.04] Collaborating with patients for better outcomes (17.12.19)
- NHS에서 10년 동안 doctor로 근무하시고 해당 주에 DeepMind Health에서 근무하며 쓴 포스팅
- 치료는 환자에게 행하는 것이 아니라 관련된 모든이들이 형성하는 것이다 라고 말하며 DeepMind에서의 경험이 놀랍다고 포스팅을 진행한다.
- 전문가들과 일하고 환자들을 초청하여 피드백을 듣고 다른 기관(C4CC)과 협력하며 일하는 등 좋은 글들을 써놓고 있지만...
- 링크의 유튜브 댓글에 보면 open source를 공개하라! suleyman은 sold out됐다, 니 개인정보 다 팔아먹어도 괜찮냐, DeepMind는 돈 있는 사람들만 살린다 등 긍정적으로만 보는 반응은 (당시에) 좀 드문 모양인 것 같다
- NHS 관련은 모르는 도메인이라 이해하기가 좀 어렵네 포스팅 읽으면서 관련 기사 찾아보자

### [20.12.07] [2017: DeepMind's year in review](https://github.com/jinmang2/Awesome-Papers/blob/master/ShallowMinded/201207_DeepMind's2017.md)
- 17년도 DeepMind의 활동 

### [21.01.07] Measuring abstract reasoning in neural networks (18.07.11)
- Abstract Reasoning은 General Intelligence에 굉장히 중요! (아르키메데스의 유레카를 떠올려봐라!)
- 바둑에서 세계 챔피온을 이겼지만, 삼각형만 계산하도록 특별히 훈련된 경우 사각형 혹은 이전에 발견되지 않은 다른 물체는 계산하지 못할 수 잇음
- 따라서 신경망이 더 우수한 지능적인 시스템을 구축할 수 있게 인간 IQ Test에서 추상적인 추론을 측정하는 데 영감을 얻음
- Paper는 ICML 2018 [Measuring abstract reasoning in neural networks](http://proceedings.mlr.press/v80/santoro18a/santoro18a.pdf)
- 표준 인간 IQ test는 응시자(test-taker)가 일상 경험을 통해 배운 원칙을 적용하여 지각적으로 단순한 시각적 장면을 해석하도록 요구 e.g., 식물이 자라는 것을 관찰, 덧셈을 공부, 은행 잔고를 추적
  - 이 후 퍼즐에 이 개념을 적용, 모양의 수, 크기 또는 색상의 강도가 순서에 따라 증가할 것이라고 추론
- 아직은 real world에서 visual reasoning test에 지식을 어떻게 전이시키는가 (적용하는가)를 측정하는 것은 어려움
  - real world -> visual reasoning problem (as in human tesing)이 어려우니
  - one controlled set of visual reasoning problems to another로 실험 setting
- 위를 위해 아래를 수행, 만일 test set에서 일반화가 잘 된다면 모델이 추상적인 개념을 추론하고 적용할 수 있는 능력을 갖출 수 있다고 판단 (과연?)
  1. Question Generator를 구축! (abstract factors: progression & attributes: colour, size 등을 포함하는 matrix problem을 생성하는)
  2. 위 generator가 접근할 수 있는 factor와 attribute의 조합을 제한함! (To measure how well models can generalise to held-out test sets)
- 논문을 보면 여러 모델을 사용했음! (CNN-MLP, ResNet, LSTM) 그리고 가장 좋았던 모델은 Wild Relation Network (WReN)라고 함!
- 최근 연구는 NN 기반의 용량 혹은 일반화 실패와 같은 강점과 약점을 파악하는데 주력
- 본 연구에선 일반화에 대한 보편적인 결론을 도출하는 것이 도움이 되지 않을 수 있음을 보여줌
- 각 모델은 특정 일반화 영역에선 잘 수행되었으나 다른 영역에선 열악했음
  - 이 특정한 성공은 사용도니 모델의 아키텍처와 모델이 대답 선택에 대한 해석 가능한 "이유"를 제공하도록 훈련되었는지 등 여러 요인에 의해 결정
- 실험 결과에서 보듯이 경험한 것을 넘어서는 input값을 extrapolate하거나 익숙하지 않은 속성을 처리해야할 때 제대로 수행되지 않았음
- Artificial General Intelligence의 길은 멀고도 험하다

### [21.01.19] [Predicting eye disease with Moorfields Eye Hospital](https://deepmind.com/blog/article/predicting-eye-disease-moorfields) (18.11.05)
- 증상이 발현되기 전 안구 질환 예측을 돕는 방안을 소개하는 포스팅
- AI System을 적용하여 미리 증상을 예견하는게 어떤 이점을 가져오는지 설명
- Moorfields, NHS와의 협력으로 이끌어낼 것이라고 

### [21.01.20] [Scaling Streams with Google](https://deepmind.com/blog/announcements/scaling-streams-google) (18.11.13)
- 의사와 간호사들을 지원해주는 mobile app, Streams 팀이 Google에 합류함을 알리는 글
- DeepMind가 2014년 Google과 힘을 합친 이유는 구글의 힘을 빌려 더 넓은 세계에 혁신을 가져오고자 함이었고, data centre efficiency, Androi bettery life, tts 등을 선보이고 이제 Stream Team도 합류했다고 한다.
- 안구 질환 예측, 초단위의 암 방사선 치료, 전자 기록으로 환자 악화상태 진단 등을 지원
- Stream 팀은 전 NHS의 외과 의사이자 연구자인 Dominic King의 지도 하에 런던에서 연구를 진행한다고 함
- 향후 몇 년동안 DeepMind는 AI는 단백질 폴딩에서 이미지 분석에 이르기까지 의료 진단, 약물 발견 등을 잠재적으로 개선할 것이라고 함

### [21.02.02] [How evolutionary selection can train more capable self-driving cars](https://deepmind.com/blog/article/how-evolutionary-selection-can-train-more-capable-self-driving-cars) (19.07.25)
- waymo의 자율 주행차의 neural networks는 물체를 탐지하거나 다른 차들이 어떻게 움직일 지 예측하는 등 많은 task를 수행
- 각기 모델을 fine-tuning하는 것은 보통 몇 주가 걸리고 연산량도 많이 필요함
- DeepMind와 협력함으로써 waymo는 다윈의 진화론적 시점을 빌려 더 효과적으로 학습을 가능케 만듦
