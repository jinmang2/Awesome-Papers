# If Beam Search is the Answer, What was the Question?

### Authors
- Clara Meister, _ETH Zurich_
- Tim Vieira, _Johns Hopkins University_
- Ryan Cotterell, _University of Cambridge_
### Publish
- EMNLP 2020

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
- 모든 Search Space를 탐색하는 것은 불가능 [1](#[1]-on-nmt-search-errors-and-model-errors:-cat-got-your-tongue?)

## Reference

#### [1] [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://www.aclweb.org/anthology/D19-1331/)
```
- Authors: Felix Stahlberg, Bill Byrne
- Publisher: ACL
- Volume:EMNLP-IJCNLP
- Year/Month: 2019.11
```
- Exact MAP decoding은 종종 저품질의 결과를 냄
- Exact Search의 50% 이상의 사례에서 빈 문자열을 반환
- Beam Search의 성공은 Exact Decoding을 근사할 수 있기 때문이라기 보다 알고리즘 내에 포함된 **hidden inductive bias** 때문
- Beam Search(for smalle k)는 nueral text 생성에서 decoding algorithm으로 사용될 경우 높은 비율의 search error를 가짐

#### [2] [Speech understanding systems: summary of results of the five-year research effort at Carnegie-Mellon University](https://kilthub.cmu.edu/articles/Speech_understanding_systems_summary_of_results_of_the_five-year_research_effort_at_Carnegie-Mellon_University_/6609821/1)
```
- Raj Reddy
- Carnegie mellon 1977
```
- Beam Search는 1970년대부터 NLP의 기초로 계속 이어져왔다.

#### [3] [Recurrent Convolutional Neural Networks for Discourse Compositionality](https://www.aclweb.org/anthology/W13-3214/)
```
- Authors: Nal Kalchbrenner and Phil Blunsom
- ACL, Proceedings of the Workshop on Continuous Vector Space Models and their Compositionality
- 2013.08
```
- 모든 search space를 탐색하는 것은 불가능 (A)
  
#### [4] [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
```
- Ilya Sutskever, Oriol Vinyals, and Quoc V. Le  - .
- Nips 2014
```
- 모든 search space를 탐색하는 것은 불가능 (A)

#### [5] [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf)
```
- Authors: Oriol Vinyals and Quoc V. Le. (Google)
- ICML 2015
```
- 모든 search space를 탐색하는 것은 불가능 (A)

#### [6] [Neural Generative Question Answering](https://www.aclweb.org/anthology/W16-0106/)
```
- Authors: Jun Yin, Xin Jiang, Zhengdong Lu, Lifeng Shang, Hang Li, and Xiaoming Li
- ACL, Proceedings of the Workshop on Human-Computer Question Answering
- 2016.06
```
- 모든 search space를 탐색하는 것은 불가능 (A)
