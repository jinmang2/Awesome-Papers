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

