# [**AlphaFold: Using AI for scientific discovery**](https://deepmind.com/blog/article/alphafold-casp13)

[이전으로](https://github.com/jinmang2/Awesome-Papers/tree/master/ShallowMinded)

## Information
- Authors:
  - Andrew Senior
  - John Jumper
  - Demis Hassabis

## Blog Post 정리

### Intro - DeepMind의 첫번 째 중요한 Milestone 소개
AI가 새로운 과학적 발견을 촉진하고 가속화할 수 있는 방법을 소개한다고 함.
generic sequence에 기반한 단백질의 3D 구조를 예측하는 최첨단 ML + 물리 + 구조 생물학 분야의 전문가들을 모음!

프로젝트의 이름은 **AlphaFold**! 포스팅 기준으로 2년전부터 준비해왔으니 16년 12월부터!

### Protein Folding 문제가 무엇인가?
단백질은 생명 유지에 필수적인 크고 복합적인 분자.
우리 몸이 수행하는 거의 모든 기능 (근육 수측, 빛 감지 또는 음식을 에너지로 전환)은 하나 이상의 단백질이 어떻게 이동하고 변화하는지로 추적 가능.
이러한 단백질 제조법은 우리 DNA에 암호화되어 있음

단백질이 할 수 있는 일은 3D 구조에 따라 다름! (예시를 본문에서 설명함. 나중에 읽어보시길.)

역할은 연구되어 왔지만, 유전적 서열(genetic sequence)로 부터 단백질의 3D 형태를 알아내는 것은 수십년간 도전되어 온 복잡한 작업!
문제는 DNA가 긴 사슬을 형성하는 아미노산 residues라고 하는 단백질 구성 요소의 서열에 대한 정보만 포함한다고 하는데...
이러한 사슬이 단백질의 복잡한 3D 구조로 접히는 방식을 예측하는 문제를 **Protein Folding Problem**이라고 함

단백질이 크면 클수록 더 복잡해지고 어려워짐... 왜냐? 고려해야 할 아미노산 사이에 더 많은 상호 작용이 있기 떄문!
[LevinThal's Paradox](https://en.wikipedia.org/wiki/Levinthal%27s_paradox)에 의하면, 올바른 3D 구조에 도달하기 전에 일반적인 단백질의
가능한 모든 구성을 열거하려면 우주의 나이보다 더 오래 걸릴 것이라고 언급

### 왜 Protein Folding이 중요한가?
단백질 모양을 예측하는 것은 우리 몸 내부의 역할의 이해의 근반이 될 뿐만 아니라 
잘못 접힌 단백질로부터 야기된 질명을 진단하거나 치료하는데 도움이 된다.

DeepMind는 이해도 증진 및 이를 통해 과학자들이 더 질병을 효과적으로 파악할 수 있게 도움을 주는데 관심을 가진다고 함! 신약 개발

인체 뿐만 아니라 플라스틱으로 인한 공해문제도 분해하는 연구로 이를 조율할 수 있을 것! 실제로 이미 박테리아로 연구 중이라고 한다.

### AI의 차별점?
지난 50년간 과학자들은 [cryo-electron microscopy](https://en.wikipedia.org/wiki/Cryogenic_electron_microscopy), [nuclear magnetic resonance](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance) or [X-ray crystallography](https://en.wikipedia.org/wiki/X-ray_crystallography),
