# If Beam Search is the Answer, What was the Question?

### Authors
- Clara Meister, _ETH Zurich_
- Tim Vieira, _Johns Hopkins University_
- Ryan Cotterell, _University of Cambridge_
### Publish
- EMNLP 2020

### Key-Word
_Beam Search_ | _Inductive Bias_ | _Uniform Information Density_ | _Cognitive Science_ | _Surprisal_ | _MAP Search_

## Abstract
- [1]에 따르면, Exact Search는 종종 low quality의 결과를 내고 small k를 가진 Beam Search는 높은 search error rate을 가짐
- 대부분의 language생성 문제에서 SOTA는 Beam Search를 활용한 경우라고 함.
- 논문의 제목처럼, Beam Search가 답이라면 이유가 뭘까?
  - 이를 분석하기 위해 Beam Search를 다른 decoding objective에 대한 exact solution으로 frame
  - 분석 결과로 Beam Search가 text에서 **_uniform information density_** 를 강제한다는 것을 알아냄
    - 이는 인지 과학(cognitive science)에서 아이디어를 얻음
  - 이를 통해 UID를 명시적으로 적용하는 decoding objective를 제안하고
  - 위 objective를 사용하는 exact decoding은 잘못 조정된(calibrated) 언어 생성 모델을 디코딩할 때 마주하는 문제를 완화
- 추가적으로 다양한 decoding 전략을 사용, 생성된 텍스트를 분석, NMT에서 UID와 BLEU가 가지는 강력한 상관관계를 확인
- 코드는 [여기](https://github.com/rycolab/uid-decoding)서 확인하세여

## Introduction
- Beam Search는 NLP의 decoding algorithm의 조상님 [2]
- 모든 Search Space를 탐색하는 것은 불가능 [3], [4], [5], [6]
- Beam Search가 모델 하에서 가장 높은 점수를 받은 후보를 반환하거나 근사할 것이라는 정확한 보장은 없지만
  - 이에 대한 장점을 반복적으로 조사함 [7], [8], [9]
  - 때문에 NLP의 heuristic search로 받아들여지기도 함
- 그러나 NMT에서 Large K를 사용하는 Beam Search 혹은 Exact Search를 사용하는 것보다 Beam Search를 neural text generator로 사용했을 때 더 나은 text를 생성함
- [1]에 의하면 Beam Search의 성공은 exact decoding을 근사할 수 있기 때문이 아니라 알고리즘 내에 포함된 **hidden inductive bias** 때문임을 보임
  - Inductive bias는 text generator에서 원하는 text를 생성하는데 가장 중요한 요소로 보임
- 이러한 현상을 연구한 [10], [11], [1], [12]에서 아무도 Beam Search의 hidden inductive bias가 무엇인지 가설을 세우지 않음
  - 우리가 할 것임

**point!!**
- 본 논문에서, **Beam Search Blessing**에 대해서 연구.
  - Beam search가 반환할 exact solution의 objective에 대해 reverse engineering을 통해 분석
  - 특히, 표준(MAP) decoding objective에 대한 regularizer를 소개
    - 이 reg_의 exact solution은 beam search로부터 찾아지는 solution과 동일
- 위에서 소개한 "Beam Search Regularizer"과 인지과학의 **UID**[13] 이론과의 연관성을 정적 inspection을 통해 밝혀낼 것임
  - UID란 
  

## Reference

#### [1] [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://www.aclweb.org/anthology/D19-1331/)
```
Authors: Felix Stahlberg, Bill Byrne
Publisher: ACL
Volume:EMNLP-IJCNLP
Year/Month: 2019.11
```

#### [2] [Speech understanding systems: summary of results of the five-year research effort at Carnegie-Mellon University](https://kilthub.cmu.edu/articles/Speech_understanding_systems_summary_of_results_of_the_five-year_research_effort_at_Carnegie-Mellon_University_/6609821/1)
```
Raj Reddy
Carnegie mellon 1977
```

#### [3] [Recurrent Convolutional Neural Networks for Discourse Compositionality](https://www.aclweb.org/anthology/W13-3214/)
```
Authors: Nal Kalchbrenner and Phil Blunsom
ACL, Proceedings of the Workshop on Continuous Vector Space Models and their Compositionality
2013.08
```
  
#### [4] [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
```
Ilya Sutskever, Oriol Vinyals, and Quoc V. Le  - .
Nips 2014
```

#### [5] [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf)
```
Authors: Oriol Vinyals and Quoc V. Le. (Google)
ICML 2015
```

#### [6] [Neural Generative Question Answering](https://www.aclweb.org/anthology/W16-0106/)
```
Authors: Jun Yin, Xin Jiang, Zhengdong Lu, Lifeng Shang, Hang Li, and Xiaoming Li
ACL, Proceedings of the Workshop on Human-Computer Question Answering
2016.06
```

#### [7] [Multiresolution recurrent neural networks: An application to dialogue response generation](https://dl.acm.org/doi/10.5555/3298023.3298046)
```
Authors:
    - Iulian Serban
    - Tim Klinger
    - Gerald Tesauro
    - Kartik Talamadupula
    - Bowen Zhou
    - Yoshua Bengio
    - Aaron Courville
AAAI'17: 
    Proceedings of the Thirty-First AAAI Conference on Artificial IntelligenceFebruary 2017
    Pages 3288–3294
```

#### [8] [Understanding Back-Translation at Scale](https://www.aclweb.org/anthology/D18-1045/)
```
(FAIR, Google Brain)
Authors: Sergey Edunov, Myle Ott, Michael Auli, David Grangier
ACL, EMNLP 2018.10~11
```

#### [9] [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://proceedings.neurips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf)
```
Authors: (Carnegi Mellon, Google Brain)
    - Zhilin Yang
    - Zihang Dai
    - Yiming Yang
    - Jaime Carbonell
    - Russ R Salakhutdinov
    - Quoc V Le.
neurips 2019
```

#### [10] [Correcting Length Bias in Neural Machine Translation](https://www.aclweb.org/anthology/W18-6322/)
```
Authors: Kenton Murray, David Chiang
ACL, Proceedings of the Third Conference on Machine Translation: Research Papers 2018.10
```

#### [11] [Breaking the Beam Search Curse: A Study of (Re-)Scoring Methods and Stopping Criteria for Neural Machine Translation](https://www.aclweb.org/anthology/D18-1342/)
```
Authors: Yilin Yang, Liang Huang, Mingbo Ma
ACL, EMNLP 2018.10~11
```

#### [12] [Empirical Analysis of Beam Search Performance Degradation in Neural Sequence Models](http://proceedings.mlr.press/v97/cohen19a.html)
```
Authors: Eldan Cohen, Christopher Beck
ICML, PMLR 97:1290-1299, 2019.
```

#### [13] [Speakers optimize information density through syntactic reduction](https://proceedings.neurips.cc/paper/2006/file/c6a01432c8138d46ba39957a8250e027-Paper.pdf)
```
Roger P. Levy and T. F. Jaeger
neurips 2007
```
