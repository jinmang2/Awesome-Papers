# ShallowMinded
딥마인드 블로그 포스팅 읽기

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
