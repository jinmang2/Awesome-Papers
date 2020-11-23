Deep contextualized word representations
========================================
- **E**mbeddings from **L**anguage **Mo**del
- 22 Mar 2018
- Allen Institute for Artificial Intelligence & University of Washington

    <img src="https://wikidocs.net/images/page/33930/elmo_DSHQjZD.png" width="80%" height="80%">

## 어떤 논문인가?
- 미국 연구기관 Allen Institute for AI와 미국 워싱턴 대학교 공동연구팀이 발표한 문장 임베딩 기법
- Computer Vision에서 널리 쓰이고 있던 **전이 학습**(Transfer Learning)을 자연어 처리에 사용하여 주목받음
- 이 논문 이후 자연어 처리에서 pre-training and then fine-tuneing하는 양상이 일반화됨

## ELMo의 업적
- 새로운 형태의 _deep contextualized_한 단어 표현방법을 소개
    - 단어 사용의 복잡한 특성을 모델링
        * Syntactic; Grammatical sense
        * Semantic; Meaning representation
    - Linguistic context(언어적 맥락)에 따른 단어 사용법 모델링
        * [polysemy](https://en.wikipedia.org/wiki/Polysemy); The capacity for a word or phrase to have multiple meanings
- 6가지 challenging NLP 문제들에서 SOTA 달성

## ELMo의 특징
- Language Model
- Convolutional Neural Network - Bidirectional LSTM - ELMo layer
- context-dependent representation을 학습
- Highway network로 학습 가속화
- Residual connection 적용, gradient 전파 효과적
- Sentence Embedding
    - 각 token이 전체 input sentence의 함수인 representation에 할당

## 왜 이름이 ELMo인가?
- Large text corpus로 학습시킨 두 Language Model 모델(bi-LSTM)에서 파생된 vector를 사용
- LM으로 부터 학습시킨 Embedding(representation) 모델이기 때문에 이를 아래와 같이 명명
- **ELMo** (Embeddings from Language Models) representations

## ELMo는 Deep하다
- 기존 learning contextualized word vectors 접근법과는 다르게,
    - [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108)
    - [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)
- ELMo는 Deep함. biLM의 모든 internal layer들의 함수라는 소리!
- 각종 downstream task(e.g, qa/sa/te/etc.)들의 각 input word에 대해,
- 해당 input word들을 쌓은 vector들의 linear combination을 학습 (<img src="https://latex.codecogs.com/svg.latex?\Large&space;C_1W_1+C2W_2"/>)
- 위와 같이 학습시키면 단순하게 top LSTM layer를 사용하는 것보다 성능을 크게(markedly) 상승시킴
- 이러한 방법으로 내부 상태를 결합시키는 것은 풍부한(rich) 단어 표현을 가능케함

## ELMo의 단어 표현은 풍부(rich)하다!
- 내부 평가를 통해 본 논문에서 보인 것은
    - `intrinsic evaluation`: 모델 내에서 자신의 성능을 수치화하여 결과를 내놓는 내부 평가 (ex, perplexity)
- 저수준 state model은 문법적인 측면에 초점을 맞춘것에 비해(syntax),
- 고수준 LSTM state는 단어 의미의 맥락 의존적인 측면을 포착한다(semantic).
- lower-level states model들은 WSD task를 풀 때 POS(part-of-speech) tagging 등의 syntax로 단어를 파악, 적용한다면
    - [`WDS(word sense disambiguation)`](http://www.scholarpedia.org/article/Word_sense_disambiguation): 사람이 무의식적(unconscious)으로 수행하는 맥락에 따른 단어의 뜻(sense)를 결정하는 문제
- higher-level states model(본 논문 ELMo와 같은)은 WSD task등에서 어떠한 수정없이 vector를 사용 가능
- **단어가 더 풍부한(rich) 표현을 가지게 된다는 의미**
- `머리`가 어떤 형태로 입력됬는지 `머리/[문법]`으로 안넣어주고 ELMo에선 `머리`가 어떤 맥락에서 어떠한 의미겠구나라는 것을 알기 때문에
- 그냥 `머리`를 입력으로 넣어줘도 된다는 의미!

## ELMo를 추가하여 specific downstream 문제를 풀자! (Fine-tune)
- 위와 같이 layer별 맥락적인 signal을 동시에 노출하기 때문에
- 학습된 model이 각 downstream task에 알맞게 적용되어 다양한 문제를 푸는 것이 가능해짐
- ELMo는 모든 task를 [relative error reductions](https://math.stackexchange.com/questions/17190/relative-error-reduction) 20%에서 SOTA를 갱신!
- ELMo는 [CoVe](https://arxiv.org/abs/1708.00107)를 뛰어넘음 ㅎ
    - `CoVe`: neural machine translation encoder를 사용한 contextualized representation을 계산한 embedding
- ELMo와 CoVe를 비교한 결과 LSTM의 최상위 layer에서 파생된 표현보다 deep-contextualized 표현이 더 뛰어남

## ELMo는 요컨대,
- 기존 representation과 다르게 ELMo는 깊음(Deep)
- 대량의 corpus로 biLM을 학습시킴
    - ELMo라는 이름은 학습시킨 LM모델로 부터 나온 Embedding이란 뜻 (Embeddings from Language Model)
- ELMo는 고수준 LSTM State model로 단어 의미의 contextual(맥락적인) 부분을 포착한다.
- 이로 인해 따로 POS tagging등을 통해 input word를 수정해주는 작업이 불필요하게 되고
    - pre-training시켜 단어가 이미 rich한 표현을 가지고 있다!
- 이는 downstream task에 적용하는 fine-tuning을 가능케한다.
    - semi-supervision의 type을 선택할 수 있음

## ELMo vs previous work

#### Previous pre-trained vector is standard, but it is context-independent representations
- [Word Representations: A Simple and General Method for Semi-Supervised Learning, Turian et al., 2010](https://www.aclweb.org/anthology/P10-1040/)
- [Distributed Representations of Words and Phrases and their Compositionality, Mikolov et al., 2013](https://arxiv.org/abs/1310.4546)
- [Glove: Global vectors for word representation, Pennington et al., 2014](https://www.aclweb.org/anthology/D14-1162/)
- Pre-training하면 labeling되지 않은 text 단어들의 syntactic, semantic 정보들을 포착할 수 있음
- 때문에 pre-trained word representation/vector는 NLU, SOTA NLP architecture의 표준이었음
- 그러나, 이는 low-quality feature
- context-independent한 표현만을 제공
- high-quality feature를 학습하기 위해서는
    - (1) 단어 사용의 복잡한 특성 및
    - (2) 언어적 맥락에 따라 단어의 사용법이 어떻게 달라지는지 모델링해야함

#### Our apporoach has subword & multi-sense information
- Enriching word vector with subword information
    - [Charagram: Embedding Words and Sentences via Character n-grams](https://www.aclweb.org/anthology/D16-1157/)
    - [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
- Learning separate vectors for each word sense
    - [Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space](https://arxiv.org/abs/1504.06654)
- 위 방법들로 기존의 word vector의 단점들을 어느정도는 극복해옴
- 하지만 ELMo는 char-CNN을 사용하여 subword unit들로부터 이점을 동일하게 얻을 수 있으며
- 표현 class를 미리 정의/예측하는 일 없이 multi-sense information(다양한 표현 정보)를 완벽하게 downstream task로 통합시킴.

#### Deep contextual representation
- context-dependent 표현에 초점을 맞춘 연구도 존재했음
- [`context2vec`: Learning Generic Context Embedding with Bidirectional LSTM, Melamud et al., 2016)](https://www.aclweb.org/anthology/K16-1006/)
    - bidirectional LSTM를 사용하여 pivot vector 주변 맥락을 encode
    - [LSTM; Hochreiter and Schmidhuber, 1997](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory/link/5700e75608aea6b7746a0624/download)
- context embedding을 학습하기 위해 표현에 pivot word를 포함시켜 지도/비지도 encoder로 계산한 접근 방식도 존재
    - supervised neural machine translation (MT) system
        - CoVe; [Learned in Translation: Contextualized Word Vectors, McCann et al., 2017](https://arxiv.org/abs/1708.00107)
    - unsupervised language model;
        - 저자의 다른 논문; [Semi-supervised sequence tagging with bidirectional language models, Peters et al., 2017](https://arxiv.org/abs/1705.00108)
- 위 방법은 large-scale에서 이점이 있음 `(CoVe는 parallel corpora에서 제한이 있음)`
- 본 논문에선 풍부한 monolingual(단일언어)에 접근하여 많은 이점을 얻으며 대략 [30 million sentences](https://arxiv.org/abs/1312.3005)의 corpus로 biLM모델을 학습
- 또한 위 접근방법을 광범위한 NLP task에서 잘 동작할 수 있도록하는 deep contextual representation 방법으로 일반화시킴

#### Layer representations
- 이전의 연구는 또한 다른 계층의 deep biRNNs이 다른 유형의 정보를 인코딩한다는 것을 보여줌
    - 예를 들어, deep LSTM의 하위 level에 POS tag와 같은 multi-task syntactic supervision을 도입하면 아래와 같은 고수준 task의 전반적인 성능을 올릴 수 있다.
        - Dependency parsing; [A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks, (Hashimoto et el., 2017)](https://arxiv.org/abs/1611.01587)
        - CCG super tagging; [Deep multi-task learning with low level tasks supervised at lower layers (Søgaard and Goldberg, 2016)](https://www.aclweb.org/anthology/P16-2038/)
- RNN기반 encoder-decoder MT system에서 [Belinkov et al. (2017)](https://arxiv.org/abs/1704.03471)은 2-layer LSTM encoder의 첫 번째 layer에서 학습된 표현이 두 번째 layer보다 POS tag를 예측하는데 더 낫다는 것을 밝혀낸 바 있음
- 최종적으로 context2vec에서 word context를 encoding하기 위한 top layer는 word sense의 표현들을 학습하는 것으로 밝혀짐
- ELMo는 이러한 layer representation이 ELMo modified language model objective에 의해 유도되고
- 이를 downstream task에 적용했을 시 매우 효과적

#### Fix pre-train weights and Add additional task-specific model capacity
- [Semi-supervised Sequence Learning, Dai and Le (2015)](https://arxiv.org/abs/1511.01432)
- [Unsupervised Pretraining for Sequence to Sequence Learning, Ramachandran et al. (2017)](https://arxiv.org/abs/1611.02683)
- 위 두 논문에선 LM, sequence AEs 그리고 task specific supervison에서의 fine tune을 사용하여 encoder-decoder pairs를 pretrain 실시
- 이와는 반대로, unlabeled data로 biLM을 pretrain한 이후에 본 논문의 저자는 weight를 고정하고 task-specific model capacity를 추가함으로써
- data와 model이 작은 경우에 downstream task에서 더 크고 풍부하고 보편적인 biLM 표현을 사용할 수 있게 됨

---
## ELMo architecture
![elmo2](https://user-images.githubusercontent.com/37775784/76699461-fd88b300-66f0-11ea-9b98-1884d8baf92d.PNG)

### Char-CNN layer
- ELMo의 입력은 해당 문자에 대응하는 유니코드 ID
- `밥` -> `235,176,165`
- ELMo 입력 ID Sequence 만들기
```python
import numpy as np

max_word_length = 20
bow_char = 258
eow_char = 259
pad_char = 260

word = '밥'
code = np.ones([max_word_length], dtype=np.int32) * pad_char
word_encoded = word.encode('utf-8', 'ignore')[:(max_word_length-2)]

word_encoded
>>> b'\xeb\xb0\xa5'

list(word_encoded)
>>> [235, 176, 165]

code
>>> array([260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260,
>>>        260, 260, 260, 260, 260, 260, 260])

code[0] = bow_char
for k, chr_id in enumerate(word_encoded, start=1):
    code[k] = chr_id
code[len(word_encoded)+1] = eow_char

code
>>> array([258, 235, 176, 165, 259, 260, 260, 260, 260, 260, 260, 260, 260,
>>>        260, 260, 260, 260, 260, 260, 260])
```

#### Dictionary형 구현체
- 위와 다르게 사전형으로 word/character를 index로 mapping하는 구현체도 존재
    - ratsgo님도 사전형 구현체를 구축해놓음
- https://github.com/DancingSoul/ELMo Pytorch버전 구현체로 보는 Char-CNN으로 biLM layer 입력 만들기!
    - [Click Here!!](https://nbviewer.jupyter.org/github/jinmang2/ELMo/blob/master/Analysis_ELMo_source_code.ipynb.ipynb)

### Bidirectional Language Models
- 저자의 다른 논문 [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108)과 유사
- 차이는 공유하는 parameter가 있냐 없냐의 차이
- 논문 분석 시작!

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Given\;a\;sequence\;N\;tokens,\;(t_1,t_2,\dots,t_N),"/>

`참고: 위의 token은 Char-CNN으로 만들어진 각 단어/문자별 padding word vector!!`

```python
# example

# <bos> : 17679
# <eos> : 17680
# <oov> : 17681
# <pad> : 17682
# <bow> : 17683
# <eow> : 17684

sentence = ['<bos>', '발', '없는', '말이', '천리', '간다', '<eos>']

max_chars = 10
batch_size = 1
max_len = 7

batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)
for i, x_i in enumerate(x_b):
    print(f"{i+1}번째 문장:")
    for j, x_ij in enumerate(x_i):
        batch_c[i][j][0] = bow_id
        if x_ij in ['<bos>', '<eos>']:
            batch_c[i][j][1] = char2id.get(x_ij)
            batch_c[i][j][2] = eow_id
        else:
            for k, c in enumerate(x_ij):
                batch_c[i][j][k+1] = char2id.get(c, oov_id)
            batch_c[i][j][len(x_ij)+1] = eow_id
    print(batch_c[i])

>>> tensor([[17684, 17679, 17683, 17682, 17682, 17682, 17682],   # <bos>
>>>         [17684,   217, 17683, 17682, 17682, 17682, 17682],   # 발
>>>         [17684,   186,    31, 17683, 17682, 17682, 17682],   # 없는
>>>         [17684,   134,    53, 17683, 17682, 17682, 17682],   # 말이
>>>         [17684,   419,    40, 17683, 17682, 17682, 17682],   # 천리
>>>         [17684,    65,    92, 17683, 17682, 17682, 17682],   # 간다
>>>         [17684, 17680, 17683, 17682, 17682, 17682, 17682]])  # <eos>
```

* 각 문자가 token!!
* ELMo의 char-cnn layer의 input은 character
* ELMo의 Bidirectional LM layer의 input도 character!!
* 단지 하나의 forward LSTM과 backward LSTM 두 쪽에 입력될 뿐!!

#### forward LM
- forward language model은 history $(t_1,\dots,t_{k-1})$가 주어졌을 때 token $t_k$을 아래와 같이 확률을 모델링하여 sequence의 확률을 계산(Baysian Theorem, 조건부확률의 곱)

    $$p(t_1,t_2,\dots,t_N)=\prod_{k=1}^{N}p(t_k|t_1,t_2,\dots,t_{k-1})$$

- 16, 17년도의 SOTA neural LM 모델들은 token embedding 혹은 char-CNN을 통해 context-independent(맥락 독립적) token representation $x_k^{LM}$을 계산하고 forward LSTM의 $L$ layer를 통과시킴
    - ```
      p208, 해당 단어 내의 문자들 사이의 의미적, 문법적 관계가 함축돼 있다. (context-independent)
      ```
    - [Exploring the limits of language modeling](https://arxiv.org/abs/1602.02410)
    - [On the state of the art of evaluation in neural language models](https://arxiv.org/abs/1707.05589)
- 각 position $k$에서 각 LSTM layer는 context-dependent한 표현 $\overrightarrow{h}_{k,j}^{LM},\;where\;j=1,\dots,L$을 산출
- LSTM의 top layer의 output $\overrightarrow{h}_{k,L}^{LM}$은 sotfmax layer를 거친 다음, 다음 토큰 $t_{k+1}$을 예측하기 위해 사용

#### backward LM
- backward LM은 전체 seqence를 거꾸로 계산하는 것 외에 forward LM과 유사
- future context가 주어지면 이전 token을 예측

    $$p(t_1,t_2,\dots,t_N)=\prod_{k=1}^{N}p(t_k|t_{k+1},t_{k+2},\dots,t_{N})$$

- $(t_{k+1},\dots,t_N)$가 주어졌을 때 token $t_k$을 표현해야함
- 위 토큰에 대한 표현 $\overleftarrow{h}_{k,j}^{LM}$은 $L$개의 backward LSTM layer j로 이루어진 deep model로 얻을 수 있고
- 이에 대한 구현은 위에서 기술한 점만 유념하여 forward LM과 유사하게(analogous) 구현 가능.

#### biLM
- biLM은 위의 forward LM와 backward LM을 결합
- Objective Function은 forward와 backward 방향의 log likelihood를 jointly maximize한다!

    $$\sum_{k=1}^{N}(\overbrace{\log\;p(t_k|t_1,\dots,t_{k-1};\Theta_x,\overrightarrow{\Theta}_{LSTM},\Theta_s)}^{forward\;LM}+\overbrace{\log\;p(t_k|t_{k+1},\dots,t_N;\Theta_x,\overleftarrow{\Theta}_{LSTM},\Theta_s)}^{backward\;LM})$$

- 각 방향의 LSTM parameter는 다르게 ($\overrightarrow{\Theta}_{LSTM}$, $\overleftarrow{\Theta}_{LSTM}$),
- token representation($\overrightarrow{\Theta}_{x}$)과 softmax layer($\overrightarrow{\Theta}_{s}$)의 paremeter는 동일하게 tie
- 위 formulation은 [Peters et al, (2017)](https://arxiv.org/abs/1705.00108)의 접근법과 유사
    - 차이점은 token representation과 softmax layer의 paremeter를 공유하냐 vs 공유하지 않냐
- ```
  p211, 이렇게 거대한 말뭉치를 단어 하나씩 슬라이딩해 가면서 그 다음 단어가 무엇인지
  맞추는 과정을 반복하다 보면 문장 내에 속한 단어들 사이의 의미적, 문법적 관계들을
  ELMo 모델이 이해할 수 있게 된다. (context-dependent)
  ```

### ELMo layer
- **biLM layer들의 linear combination으로 word representation을 학습시키는 접근법 제시**
- ELMo에서 정말 중요한 부분!
- 각 토큰 $t_k$에 대해, $L$ layer biLM은 $2L+1$개의 representation set을 계산

    <img src="https://user-images.githubusercontent.com/37775784/76698918-4d647b80-66eb-11ea-9e93-bbe92dd80f0f.PNG" width="50%" height="50%">

    - where <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_{k,0}^{LM}"/>: token layer
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;h_{k,j}^{LM}=\bigg[\overrightarrow{h_{k,j}^{LM}},\overleftarrow{h_{k,j}^{LM}}\bigg]"/>: each biLSTM layer
- downstream model에 포함시키기 위해 ELMo는 <img src="https://latex.codecogs.com/svg.latex?\Large&space;R"/>의 모든 layer들을 single vector <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{ELMo}_k=E(R_k;\Theta_e)"/>로 축소(collapse)시킴.
- 간단한 경우를 살펴보면, ELMo는 단지 TagLM과 CoVe와 같이 top layer <img src="https://latex.codecogs.com/svg.latex?\Large&space;E(R_k)=h_{k,L}^{LM}"/>을 선택한다.
    - [TagLM(Peters et al., 2017)](https://arxiv.org/abs/1705.00108)
    - [CoVe(McCann et al., 2017)](https://arxiv.org/abs/1708.00107)
- Generally, 모든 biLM layer의 task specific weighting을 다음과 같이 계산(linear combination)

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{ELMo}_k^{task}=E(R_k;\Theta^{task})=\gamma^{task}\sum_{j=0}^{L}s_{j}^{task}h_{k,j}^{LM}\;\cdots\;(1)"/>

    - In (1), <img src="https://latex.codecogs.com/svg.latex?\Large&space;s^{task}"/>: softmax-normalized weights
        - j번째 layer가 해당 태스크 수행에 얼마나 중요한지를 가리키는 스칼라 값
    - scalar parameter <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma^{task}"/>는 task model이 전체 ELMo vector의 scale을 조정
        - 해당 task가 얼마나 중요한지 뜻하는 스칼라 값
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma"/>는 최적화 과정을 돕는 실질적으로 중요한 과정
        - Suplemental material for details 참고
- 고려할 부분
    - 각 biLM layer의 activation은 각기 다른 분포를 가짐
    - 몇몇 경우에 이는 weighting하기 전에 각 biLM layer에 [layer normalization](https://arxiv.org/abs/1607.06450)를 적용하는데 도움이 됨.

## ELMo의 biLM을 사용하여 supervised NLP task를 푸는 방법
- First consider; biLM이 없는 지도학습 모델의 최하단 layer
    - 대부분의 supervised NLP 모델은 최하단 layer에서 architecture를 공유하기에 ELMo를 일관되고 통합된 방식으로 추가할 수 있게 해줌
    - 주어진 tokens <img src="https://latex.codecogs.com/svg.latex?(t_1,\dots,t_N)"/>의 sequence에 대해,
    - pre-training한 word embedding 혹은 선택적으로 char 기반 표현을 사용하여 각 토큰 위치에 대해 context-independent 토큰 표현 <img src="https://latex.codecogs.com/svg.latex?x_k"/>를 형성
    - 그 다음, 모델은 일반적으로 bidirectional RNN, CNN, DNN을 사용하여 context-sentitive 표현인 <img src="https://latex.codecogs.com/svg.latex?h_k"/>를 형성
- Second; ELMo를 supervised model에 추가하기 위해서
    - 첫 번째로 biLM의 가중치를 freeze시키고
    - ELMo vector <img src="https://latex.codecogs.com/svg.latex?ELMo_k^{task}"/>와 <img src="https://latex.codecogs.com/svg.latex?x_k"/>를 연결(concatenate)하고 task RNN에 ELMo의 향상된 표현인 <img src="https://latex.codecogs.com/svg.latex?[x_k;ELMo_k^{task}]"/>를 전달
    - SNLI, SQuad와 같은 몇몇 task의 경우 아래 과정을 거쳐 추가적인 개선이 존재하는지 관찰
        - output specific linear weights의 다른 set을 도입 및 <img src="https://latex.codecogs.com/svg.latex?h_k"/>를 <img src="https://latex.codecogs.com/svg.latex?[h_k;ELMo_k^{task}]"/>로 대체하여
        - task RNN의 output에 ELMo를 포함
- Finally; ELMo에 아래 행위를 취하면 더 효과적이라는 것을 발견
    - 일정량의 [`dropout`](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)을 추가
    - Loss function에 <img src="https://latex.codecogs.com/svg.latex?\gamma||w||_2^2"/>를 더하여 ELMo의 가중치를 정규화
    - 위 내용은 모든 biLM layer들의 평균에 가깝도록 ELMo weights에 [inductive bias](https://en.wikipedia.org/wiki/Inductive_bias)를 부과함

## Pre-trained biLM architecture
- 본 논문의 pre-trained biLM은 아래 두 논문과 구조가 유사
    - [Exploring the Limits of Language Modeling, Jozefowicz et al. (2016)](https://arxiv.org/abs/1602.02410)
    - [Character-aware neural language models, Kim et al. (2015)](https://arxiv.org/abs/1508.06615)
- 다른 점은 **(1) 양 방향으로 joint하여 학습시키는 점과 (2) LSTM layer들 사이에 `residual connection`을 추가한 점**
- Large scale biLMs에 초점을 맞춰서 작업을 진행
    - 저자는 본인의 다른 논문 [Peters et al. (2017)](https://arxiv.org/abs/1705.00108)에서 forward-only LMs보단 biLMs, 그리고 large scale training을 사용하는 것이 중요하다고 강조
- [Jozefowicz et al. (2016)](https://arxiv.org/abs/1602.02410)의 single best model `CNN-BIG-LSTM`의 모든 embedding과 hiddem 차원을 반으로 줄임. 왜?
    - 순전히 문자 기반 입력 표현을 유지하면서
    - 전반적인 언어 모델 복잡성과 다운스트림 작업에 대한 모델 크기 및 계산 요구 사항의 균형을 맞추기 위해!
- 최종 모델은 4,096 unit과 512D projection 그리고 1->2로 가는 residual connection을 가진 <img src="https://latex.codecogs.com/svg.latex?L=2"/> biLSTM layers를 사용
- Context insensitive type representation은 아래 두 filter/projection을 사용
    - 2,048 character n-gram convolution filters followed by two highway layers ([Srivastava et al., 2015](https://arxiv.org/abs/1507.06228))
    - linear projection down to a 512 representation
- 그 결과로 biLM은 순수 문자 입력(다른 POS tagging등이 필요X)으로 인해 train set 이외의 token을 포함한 각 input token의 표현의 three-layer를 제공
- 대조적으로, 기존의 word embedding 방법들은 해당 token의 1-layer적인 표현밖에 제공하지 못함 (3-layer가 더 rich함)
- [1B Word Benchmark (Chelba et al., 2014)](https://arxiv.org/abs/1312.3005)에서 10 epoch 학습시킨 후에,
- forward `CNN-BIG-LSTM`의 perplexity가 30.0인데 비해 forward LM과 backward LM의 평균 perplexity는 39.7이었다. (응? 왜 더 나쁘지;;)
- 일반적으로, backward value가 조금 낮긴 하지만, forward와 backward perplexity가 근사적으로 같다는 것을 확인
- Pre-training 후, biLM은 각 task에 대해 표현을 계산
- 몇몇 경우에 domain 특화된 data에 biLM을 fine-tuning시키는 것은 perplexity를 상당히 낮춰주고 downstream task 성능을 향상시킴
- 이는 biLM을 위한 domain transfer의 한 유형이라고 볼 수 있음
- 대부분의 경우에 downstream task에 fine-tuning된 biLM을 사용

## Challenging 6 NLP tasks Evaluation
![elmo3](https://user-images.githubusercontent.com/37775784/76721805-8ac91780-6784-11ea-9cf5-e0a9c3ac5256.PNG)

- Table 1은 6가지 benchmark NLP task에서의 ELMo performance를 보여줌
- 모든 task에서 간단히 ELMo를 더하기만해도 SOTA 달성 with relative error reductions 6-20%

### Question answering
- The Stanford Question Answering Dataset (SQuAD) ([Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250))
- 위키피디아 paragraph와 이에 대한 답변으로 구성된 100K+개의 대중 QA쌍으로 구성
- Baseline model: [Simple and effective multi-paragraph reading comprehension, Clark and Gardner, 2017](https://arxiv.org/abs/1710.10723)
    - [Bidirectional attention flow for machine comprehension, Seo et al.(BiDAF; 2017)](https://arxiv.org/abs/1611.01603)
    - Baseline model은 위 논문의 향상된 버전
- bidirectional attention component뒤에 self-attention layer를 추가하고
- 일부 pooling 계산을 단순화하고
- LSTM을 gated recurrent units (GRUs; [Cho et al., 2014](https://arxiv.org/abs/1409.1259))로 대체
- baseline에 ELMo를 추가한 이후에
    - test set F1은 81.1%에서 85.8%로 4.7% 상승 with 24.9% relative error reduction
    - SOTA기준 1.4% 향상
- 11 member ensemble F1은 87.4%로 리더보드에 submit할 당시 전체 1위!
- baseline model에 CoVe를 추가하여 1.8% 상승을 이룩한데 비해 ELMo를 추가해 4.7%의 향상을 얻어낸 것은 괄목할만함!

### Textual entailment
- Textual entailment는 "premise(전제)"가 주어졌을 때 "hypothesis(가설)"이 참인지를 결정하는 task
- The Stanford Natural Language Inference(SNLI) corpus ([Bowman et al.,2015](https://arxiv.org/abs/1508.05326))
- 대략 550K의 hypothesis/premise pair를 제공
- Baseline model: ESIM sequence model from [Chen et al.,2017](https://arxiv.org/abs/1609.06038)
    - premise와 hypothesis를 encode하기 위해 **biLSTM**을 사용
    - matrix attention layer,
    - local inference layer,
    - pooling operation before output layer로 구성
- 5개의 random seeds에서 ESIM에 ELMo를 더하니 accuracy를 평균 0.7% 향상시킴
- 5 member ensemble의 전체 accuracy는 89.3%로 이전 ensemble best 88.9%([Gong et al.,2018](https://arxiv.org/abs/1709.04348))의 성능을 뛰어넘음

### Semantic role labeling
- Semantic role labeling (SRL) system은 문장의 [predicate-](https://ko.wikipedia.org/wiki/%EC%84%9C%EC%88%A0%EC%96%B4)[argument](https://ko.wikipedia.org/wiki/%EB%85%BC%ED%95%AD)(서술어-논황) structure를 모델링.
- 이는 종종 `"Who did what to whom"`으로 표현되기도 함
- [Deep Semantic Role Labeling: what Works and What's Next, He et al.,2017](https://kentonl.com/pub/hllz-acl.2017.pdf)은 SRL은 BIO tagging 문제로 모델링했고 8-layer deep biLSTM을 사용
    - forward와 backward 방향을 [interleaved](https://www.scienceall.com/%EC%9D%B8%ED%84%B0%EB%A6%AC%EB%B8%8Cinterleave/)하게 배열
        - 맞는 설명인진 모르겠다.
    - [End-to-end learning of semantic role labeling using recurrent neural networks, Zhou and Xu et al.,2015](https://www.aclweb.org/anthology/P15-1109/)
- [He et al.,2017](https://kentonl.com/pub/hllz-acl.2017.pdf)의 재구현체에 ELMo를 더했을 때, single model test set F1은 81.4%에서 84.6%로 3.2% jump
    - [OntoNotes benchmark (Pradhan et al.,2013)](https://www.aclweb.org/anthology/W13-3516/)에서 새로운 SOTA를 달성
- 심지어 이전 best ensemble result보다 1.2%나 더 향상시킴

### Coreference resolution
- 텍스트에서 언급된 내용을 real world의 entity로 clustering하는 task
- Baseline model: E2E span-based NN of [End-to-end Neural Coreference Resolution, Lee et al.,2017](https://arxiv.org/abs/1707.07045)
    - 첫 span 표현을 계산하는데 biLSTM과 attention mechanism을 사용
    - coreference chain을 찾는데 softmax mention ranking model을 적용
- OntoNotes coreference annotation from CoNLL 2012 [(Pradhan et al.,2012)](https://www.aclweb.org/anthology/W12-4501/)
- ELMo을 더하여 67.2%에서 70.4%로 F1 3.2% 상승!
- 이전 best ensemble result보다 1.6% 향상시키며 SOTA 달성!

### Named entity extraction
- The CoNLL 2003 NER task ([Sang and Meulder, 2003](https://arxiv.org/abs/cs/0306050))
- 네 가지의 다른 entity types (PER, LOC, ORG, MISC)로 tag된 Reuters RCV1 corpus의 newswire로 구성
- Baseline model: (1) pre-trained word embeddings (2) char-CNN rep (3) two biLSTM layers (4) conditional random field (CRF) loss ([Lafferty et al.,2001](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers))
    - [Neural Architectures for Named Entity Recognition, Lample et al.,2016](https://arxiv.org/abs/1603.01360)
    - [Semi-supervised sequence tagging with bidirectional language models, Peters et al.,2017](https://arxiv.org/abs/1705.00108)
    - similar to [Collobert et al.,2011](https://arxiv.org/abs/1103.0398)
- ELMo는 biLSTM-CRF가 92.22% F1을 얻도록 강화시킴
- 우리 system과 이전 SOTA인 [Peters et al.,2017](https://arxiv.org/abs/1705.00108)의 결정적인 차이점은 **모든 biLM layer의 weighted average를 계산했느냐, top biLM layer만 사용했느냐 다.**
- 1-layer < 3-layer 우수하다!

### Sentiment Analysis
- Stanford Sentiment Tree-bank (SST-5; [Socher et al.,2013](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf))
- 5 labels (from very nagative to very positive)
- movie review로부터 sentence를 위 label 중 하나로 선택
- 문장은 idiom(관용어)와 같은 다양한 linguistic phenomena(언어적 현상)과
- 모델이 학습하기 어려운 부정어구와 같은 복잡한 문장 구조를 포함
- Baseline model: Biattentive Classification Network(BCN) from [McCann et al.,2017](https://arxiv.org/abs/1708.00107)
    - CoVe embedding으로 augment시킨 이전 SOTA 모델
- CoVe를 ELMo로 대체시켰더니 1.0% acc 향상 및 새로운 SOTA 달성!

## Analysis ELMo

### Alternate layer weighting schemes
- biLM layer를 결합하는데 있어 Eq(1)은 많은 alternative가 존재
- Contextual representation에 대한 이전 연구는 biLM ([Peters et al.,2017](https://arxiv.org/abs/1705.00108))든 MT encoder (CoVe; [McCann et al.,2017](https://arxiv.org/abs/1708.00107))든 마지막 layer만 사용했다.
- Regularization parameter <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda"/>를 선택하는 것은 굉장히 중요하다!
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda=1"/>은 weighting function을 간단한 전반적인 layer들의 평균으로 줄여버린다.
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda=1e-3"/>은 layer weights를 다양하게 만든다.

![elmotable2](https://user-images.githubusercontent.com/37775784/76726226-ebac1c00-6793-11ea-8bb5-3611d07d2483.PNG)

- 위 table2에서 SQuAD, SNLI, SRL에서 이러한 alternative를 비교해보자.
- baseline < last only < all layers repr
- small <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda"/>가 대부분의 ELMo에서 좋았음
- smaller training set에서는 <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda"/>에 민감하지 않았음

### Where to include ELMo?
- 본 논문의 Challenging 6 NLP task는 biRNN의 최하단 layer의 input으로만 word embedding을 포함함
- 그러나, 저자는 task-specific architecture에서 ELMo를 biRNN의 output에 포함시켰을 경우 몇몇 task에서 성능이 향상되는 것을 발견함.

![elmotable3](https://user-images.githubusercontent.com/37775784/76726679-3d08db00-6795-11ea-945a-bc3f3bbae49b.PNG)

- SRL제외 output에 이를 단순하게 input layer에만 ELMo를 추가한 것 보다 output layer와 input layer 동시에 ELMo를 포함시켰더니 성능이 뛰었다.
- SRL과 coreference resolution은 input layer에만 넣었을 때 성능이 좋았음
- 왜 그럴까 저자는 이유로 아래와 같이 설명하고 있음
    - SNLI와 SQuAD architecture는 biRNN 다음에 attention layer를 사용
    - ELMo를 소개시키는 것은 모델로 하여금 biLM의 내부 표현에 집중케 만듦
    - STL의 경우, biLM으로 나온 것보다 task-specific context representation이 더 중요함

### What information is captured by the biLM's representations?
- EMLo를 추가하면 word vector보다 성능이 향상되기 때문에, biLM의 contextual 표현은 반드시 word vector에서는 포착되지 않는 NLP task에 유용한 일반적인 정보로 encode되야한다.
- 그 맥락을 사용할 때 biLM은 직관적으로 단어의 의미를 명확하게 해야한다는 얘기이다.
- `play`라는 매우 polysemous한 단어를 생각해보자.

 ![elmotable4](https://user-images.githubusercontent.com/37775784/76727406-4bf08d00-6797-11ea-9b22-bd2f08973ba9.PNG)

- GloVe로 `play`와 가장 가까운 단어를 추출하면 이는 몇몇 POS(e.g., `played`, `playing` as verbs, `player`, `game` as nouns)에 걸쳐 펴져있으나 `play`의 sports적인 관계 어휘에 집중되어있다.
- 이와는 반대로 아래 두 열은 biLM의 `play`에 대한 맥락적 표현을 사용해 SemCor dataset에서 가장 가까운 문장을 보여주고있다.
- 이 경우, biLM은 주어진 문장에서 pos와 word sense 둘 다 모호하지 않은 것을 확인할 수 있다
- 이러한 발견은 [What do Neural Machine Translation Models Learn about Morphology?, Belinkov et al., 2017](https://arxiv.org/abs/1704.03471)와 유사한 맥락적 표현의 내부 평가로 정량화될 수 있다.
- biLM으로부터 encoding된 정보를 분리하기 위해 해당 표현은 세밀한(fine-grained) WDS와 POS tagging task의 예측을 만들기 위해 사용된다.
- 이러한 접근법을 통하여 CoVe 및 개별 layers와 비교하는 것이 가능하다.

## Conclusion
- biLM으로부터 얻은 고품질 deep context-dependent 표현을 학습하기 위한 일반적인 접근 방법을 소개
- 광범위한 NLP task에 ELMo를 적용했을 때 성능을 크게 향상시키고 SOTA달성한 것을 보임
- 여러 시험을 통해(Through ablations and other controlled experiments,) biLM layers가 맥락 속의 단어에 대한 syntatic/semantic 정보의 다양한 유형을 효율적으로 encoding할 수 있다는 것을 확인
- 모든 layer를 사용하는 것이 전반적인 task 성능을 올릴 수 있었음
