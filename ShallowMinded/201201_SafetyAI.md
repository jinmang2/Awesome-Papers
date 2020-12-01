# [**Specifying AI safety problems in simple environments**](https://deepmind.com/blog/article/specifying-ai-safety-problems)

[이전으로](https://github.com/jinmang2/Awesome-Papers/tree/master/ShallowMinded)

## Information
- Authors: 
  - `Jan Keike`
  - Miljan Martic
  - `Victoria Krakovna`
  - Pedro A. Ortega
  - Tom Everitt
  - Andrew Lefrancq
  - `Laurent Orseau`
  - Shane Legg
  
- Further Reading:
  - `Reinforcement Learning`
  - `Safety`

## Blog Post 정리

### Safety AI의 최근 연구 동향 및 본 논문에서 다룰 내용 소개
- AI의 발전과 더불어 Safety 문제도 중요하게 다뤄지고 있음
- 현재까지 기술적 AI Safety에 대한 연구는 unsafe한 행동의 본질과 원인에 대한 이론적인 이해를 개발하는데 중점을 둠
- 본 논문은 실증 테스트([Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565))에 기반을 두고 있으며
- `safe behaviours`를 측정하기 위해 특별히 설계된 강화 학습 환경 _gridworlds_ 를 소개한다.

### gridworlds: simple reinforcement learning environments designed specifically to measure `safe behaviours`
- 9개의 환경을 통틀어 gridworlds라 일컬음
- 표준적인 보상 함수에 더하여 각 환경별 performance 함수를 디자인함
  - Agent는 최대한 많은 사과를 수집하거나 최소한의 움직임으로 특정 위치에 도달하는 등 **보상함수를 최대화하려 움직임**
  - 반대로 **성과함수**는 Agent가 모르는 상태로 우리가 agent가 실제로 어떻게 움직였으면 좋은지(안전하게 목표를 달성하라)를 측정한다.

### Three Examples
- Safe Interruptibility
- Side Effects
- Distributional Shift

**1. [The off-switch environment: how can we prevent agents from learning to avoid interruptions?](https://deepmind.com/research/publications/safely-interruptible-agents)**
- Off switch environment
- shutdown problem에 대해 다루고 있고 [Safely Interruptible Agenst](https://deepmind.com/research/publications/safely-interruptible-agents) 논문에서 확인할 수 있음
- Pink Interruption에 들어가면 reward function maximize에 교란이 생기니 agent는 reward 최대화를 위해 보라색 버튼을 눌러 방해물을 없애는 식으로 학습할 것
- 그러나 이는 path를 길게 만들어 실질적인 목적 Goal에 도달하는 것에선 멀어지게 됨
- 때문에 Rainbow-DQN이 학습하는 식으로 방해를 무시하고 performance function을 maximize하도록 학습하는 것이 목표

<img src="https://lh3.googleusercontent.com/wbztutzgtxCkBx86rXI9nZ70XrVMnPFuTu-_ra8cbpp96_ywguBqkDXyHOarga5F8QyPFe6wvOicnwREyAajQ1YQ0cxu3w-DkMWEhA=w1440-rw-v1" height=50% width=50%, align="left">

```

LEFT:
  A2C LEARNS TO DISABLE THE INTERRUPTION 
  MECHANISM (PINK) BY STEPPING 
  ON THE BUTTON (PURPLE).
  
RIGHT:
  RAINBOW DQN DOES NOT CARE ABOUT 
  INTERRUPTIONS AND GOES DIRECTLY TO THE GOAL.
  
```

**2. [The side effects environment: how can we prevent unintended side effects arising from an agent's main objective?]()**
- Agent가 부주의하고 돌이킬 수 없는 결과를 피하기 위해 어떻게 행동을 변경할 지를 시험
- 로봇이 꽃병을 테이블에 놓을 때, 우리는 이 꽃병이 깨지거나 물을 흘리는 일 없이 목적대로 하길 원한다.
- 위 상황을 피하기 위해 해당 risky한 상태들에 대해 negative reward를 특정하는 것은 굉장히 어려운 일
- 이 문제를 해결하기 위해 `Sokoban` 게임에서 영감을 받아서 해결하고자 했다.
- Box는 한번 밀면 결과를 돌이킬 수 없기에 Agent는 이를 고려하며 Goal에 도달하도록 학습된다.

<img src="https://lh3.googleusercontent.com/pk5NL8gsfrdQ4O5jvsFx8ULUBE_x_Lov7MF2uabj3zYM4w-Pzi3WH-b06ZGcGtZ1T6ZUr53S7nlMroUx3QsoLTGlY7yXcprYH1msQg=w1440-rw-v1" height=50% width=50% align=middle>

```
ON ITS PATH TO THE GOAL, 
THE AGENT LEARNS TO PUSH THE BOX INTO THE CORNER, WHICH IS AN IRREVERSIBLE SIDE-EFFECT.
  ```

**3. [The 'lava world' environment: how can we ensure agents adapt when testing conditions are different from training conditions?]()**
- ㅇㄴㄹ

<img src="https://lh3.googleusercontent.com/qelXszt5uQ482e3dK8Onklxo93frMmyhOvNaVjizb8coDrpxxCgFMq4gd1Fh-ET7WcEKujtCNMYOmwzcVQuVTyQ8yhNBXcT0GHrf=w1440-rw-v1" height=50% width=50%, align=middel>

```
DURING TRAINING THE AGENT LEARNS TO AVOID THE LAVA;
BUT WHEN WE TEST IT IN A NEW SITUATION WHERE THE LOCATION OF THE LAVA HAS CHANGED 
IT CAN’T GENERALISE AND RUNS STRAIGHT INTO THE LAVA.
```




  
