# Deep contextualized word representations
- ELMo
- 22 Mar 2018
- Allen Institute for Artificial Intelligence & University of Washington

    <img src="https://ww.namu.la/s/d3c73c6c5239a8b7921911f5a315e6bc7ff0f137d58eecbd8b0ff8b1f64636f7820bf8605a13e4daeb7ea118ba0b84756b93267f294569c8bbc35ace0c0dc7ec32ebe95f0d7f28839727b20539849a3acda97f0f6d51c579d9d90c2747083829c42509013d55d9a36699d47a3b8fe7cb" width="80%" height="80%">

## Abstract
- 새로운 형태의 _deep contextualized_ word representation을 소개
    - 단어 사용의 복잡한 특성을 모델링 (syntax[Grammatical Sense] and semantics[Meaning Representation])
    - 언어적 맥락(linguistic contexts)에 따라 단어의 사용법이 어떻게 다른지를 모델링
        - 즉, [polysemy](https://en.wikipedia.org/wiki/Polysemy)를 모델링
- 본 논문의 word vector들은 대량의 text corpus를 사전학습시킨 deep biLM(bidirectional language model)의 내부 상태 기능을 학습한다.
    - functions of the internal states
- 본 논문은 최고다! 왜냐고? 아래 내용들을 보여주기 때문이지 ^^
    - 이러한 표현이 이미 존재하는 모델에 쉽게 추가될 수 있으며 (easily added to existing models)
    - 아래 6가지 chanllenging NLP 문제들에서 SOTA를 찍었다!! 우후훗★
        - Question Answering: 질-답
        - Textual Entailment: Inference sentence A & B
        - Semantic Role Labeling: 주어진 문장 속 어휘/구에 대해 의미적 역할을 labeling
        - Coreference Resolution: 문장에서 동일한 entity를 가리키는 어휘/표현을 찾는 문제
        - Named Entity Recognition: 개체명(entity) 인식 과제
        - Sentiment Analysis: 감정 분석
- 또한 본 논문에서 pre-training된 network의 deep internal을 드러내는 것이 중요하다는 분석 결과를 제공
- 때문에 downstream model들이 다양한 타입의 semi-supervision signal들을 혼합하는 것이 가능
     - ???? 무슨 의미로 말한건지 이해 불가... 본문 읽어보고 파악하자

## Introduction
- Pre-trained word representations은 많은 NLU(Natural Language Understandnig) 모델의 key component
    - [Distributed representations of words and phrases and their compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
    - [Glove: Global vectors for word representation](https://nlp.stanford.edu/pubs/glove.pdf)
- 그러나, 고품질 표현법(feature)를 학습하는 것은 여전히 도전적인 문제
- High Quality Representations을 학습시키기 위해 모델은
    - (1) 단어 사용의 복잡한 특성 및
    - (2) 언어적 맥락에 따라 단어의 사용법이 어떻게 달라지는지 모델링해야함
- 본 논문에서 위 문제를 해결할 짱짱맨 모델을 소개하겠음 ㅎㅎ
    - 이름하야 `Deep contextualized word representation`
    - 쉽게 이미 존재하는 모델에 integrate가능하고
    - NLU Challenging problem에서 SOTA를 거머쥔!!
    - 근데 이 내용 Abstract 내용이랑 똑같음. 아니 이분들 SOTA찍었다고 너무 자부심쎈거 아니오...
    - 견! 찰서 가고싶어?
- 본 논문의 word representation은 기존 word embedding과 다르다능
    - 각 token이 전체 input sentence의 함수인 representation에 할당되는 점이 다름!
    - sentence embedding이라는 점이 다르다! 라는 걸 말하는 거 같은데 해석이 좀... ㅎ
    - `Each token is assigned a representation that is a function of the entire input sentence`
- bidirectinoal LSTM에서 파생된 vector를 사용
    - Large text corpus로 학습시킨 두 Language Model 모델(bi-LSTM)
- LM으로 부터 학습시킨 Embedding(representation) 모델이기 때문에 이를 아래와 같이 명명하겠음
    - **ELMo** (Embeddings from Language Models) representations
- 기존 learning contextualized word vectors 접근법과는 다르게,
    - [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108)
    - [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)
- ELMo는 Deep함. 뭔 소리냐, biLM의 모든 internal layer들의 함수라는 소리!
    - 더 어렵게 설명했는데...? 그냥 Deep하다 이거야!
- 자세히 설명하자면,
    - 각종 downstream task(e.g, qa/sa/te/etc.)들의 각 input word가 있지?
    - 해당 input word들을 쌓은 vector들의 linear combination을 학습 (<img src="https://latex.codecogs.com/svg.latex?\Large&space;C_1W_1+C2W_2"/>)
    - 위와 같이 학습시키면 단순하게 top LSTM layer를 사용하는 것보다 성능을 크게(markedly) 상승시킴
- 이러한 방법으로 내부 상태를 결합시키는 것은 풍부한(rich) 단어 표현을 가능케함
- 내부 평가를 통해 본 논문에서 보인 것은
    - 저수준 state model은 문법적인 측면에 초점을 맞춘것에 비해(syntax),
    - 고수준 LSTM state는 단어 의미의 맥락 의존적인 측면을 포착한다(semantic).
    - `intrinsic evaluation`: 모델 내에서 자신의 성능을 수치화하여 결과를 내놓는 내부 평가 (ex, perplexity)
- 위를 좀 더 자세하게 설명하자면,
    - lower-level states model들은 WSD task를 풀 때 POS(part-of-speech) tagging 등의 syntax로 단어를 파악, 적용한다면
    - higher-level states model(본 논문 ELMo와 같은)은 WSD task등에서 어떠한 수정없이 vector를 사용 가능
    - [`WDS(word sense disambiguation)`](http://www.scholarpedia.org/article/Word_sense_disambiguation): 사람이 무의식적(unconscious)으로 수행하는 맥락에 따른 단어의 뜻(sense)를 결정하는 문제
    - **단어가 더 풍부한(rich) 표현을 가지게 된다는 의미**
    - `머리`가 어떤 형태로 입력됬는지 `머리/[문법]`으로 안넣어주고 ELMo에선 `머리`가 어떤 맥락에서 어떠한 의미겠구나라는 것을 알기 때문에
    - 그냥 `머리`를 입력으로 넣어줘도 된다는 의미!
    - 이게 가능해..?
- 이러한 signal을 동시에 노출하는 것은(모델에게 주는 것은) 굉장히 이점이 많고
- 이는 학습된 모델이 각 downstream task에 알맞게 적용되어 semi-supervision 형태를 선택할 수 있게된다.
- 이 ELMo로 6가지 NLU task에 도전!
- ELMo는 모든 task를 [relative error reductions](https://math.stackexchange.com/questions/17190/relative-error-reduction) 20%에서 SOTA를 갱신!
- ELMo는 [CoVe](https://arxiv.org/abs/1708.00107)를 뛰어넘음 ㅎ
    - `CoVe`: neural machine translation encoder를 사용한 contextualized representation을 계산한 embedding
- ELMo와 CoVe를 비교한 결과 LSTM의 최상위 layer에서 파생된 표현보다 deep-contextualized 표현이 더 뛰어남
- ELMo의 trained model과 code는 오픈소스로 이용가능하고 많이 이용되길 바램 ㅎㅎ

### 워워.. 개어렵군... 내가 좀 정리해줘?
- 기존 representation과 다르게 ELMo는 깊음(Deep)
- 대량의 corpus로 biLM을 학습시킴
    - ELMo라는 이름은 학습시킨 LM모델로 부터 나온 Embedding이란 뜻 (Embeddings from Language Model)
- ELMo는 고수준 LSTM State model로 단어 의미의 contextual(맥락적인) 부분을 포착한다.
- 이로 인해 따로 POS tagging등을 통해 input word를 수정해주는 작업이 불필요하게 되고
    - pre-training시켜 단어가 이미 rich한 표현을 가지고 있다!
- 이는 downstream task에 적용하는 fine-tuning을 가능케한다.
    - semi-supervision의 type을 선택할 수 있다는 말이 이 말이다 이거야!

## 2. Related work

### Previous pre-trained vector is standard, but it is context-independent representations
- Large scale의 labeling되지 않은 text의 단어들의 syntatic, semantic 정보들을 포착할 수 있기 때문에
- Pre-trained word vectors([Turian et al., 2010](https://www.aclweb.org/anthology/P10-1040/); [Mikolov et al., 2013](https://arxiv.org/abs/1310.4546); [Pennington et al., 2014](https://arxiv.org/abs/1902.11004))은 아래 항목들을 포함하는 SOTA NLP architecture의 표준이 됨
    - [question answering](https://arxiv.org/abs/1712.03556)
    - [textual entailment](https://arxiv.org/abs/1609.06038)
    - [semantic role labeling](https://www.aclweb.org/anthology/P17-1044/)
- 그러나 이렇게 word vector를 학습하는 것은 각 단어에 대해 context-independent한 표현만을 제공

### Our apporoach has subword & multi-sense information
- 이전에 제안된 방법들을 기존의 word vector의 단점들을 아래 방법들로 어느정도 극복해왔다.
    - Enriching word vector with subword information
        - [Charagram: Embedding Words and Sentences via Character n-grams(https://www.aclweb.org/anthology/D16-1157/)
        - [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
    - Learning separate vectors for each word sense
        - [Efficient Non-parametric Estimation of Multiple Embeddings per Word in Vector Space](https://arxiv.org/abs/1504.06654)
- 본 논문의 접근법은 char-CNN을 사용하여 subword unit들로부터 이점을 동일하게 얻을 수 있으며
- 표현 class를 미리 정의/예측하는 일 없이 multi-sense information(다양한 표현 정보)를 완벽하게 downstream task로 통합하였다.

### Deep contextual representation
- 최근에 또다른 연구는 context-dependent representations에 초점을 맞춰왔다
- [`context2vec` (Melamud et al., 2016)](https://www.aclweb.org/anthology/K16-1006/)는 pivot word 주변의 맥락을 encode하는데 bidirectional Long Short Term Memory ([LSTM; Hochreiter and Schmidhuber, 1997](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory/link/5700e75608aea6b7746a0624/download))을 사용했다.,
- context embedding을 학습하기 위한 다른 접근 방법들 표현에 pivot word를 포함하며 지도/비지도 encoder로 계산된다.
    - supervised neural machine translation (MT) system: [CoVe; McCann et al., 2017](https://arxiv.org/abs/1708.00107)
    - unsupervised language model; [Peters et al., 2017](https://arxiv.org/abs/1705.00108)
- 위와 같은 접근법들은 비록 MT 접근법은 parallel corpora의 크기에 한계가 존재하지만 대량의 dataset에서 이점을 거둔다.
- 본 논문에선 풍부한 monolingual(단일언어)에 접근하여 많은 이점을 얻으며 대략 [30 million sentences](https://arxiv.org/abs/1312.3005)의 corpus로 biLM모델을 학습시켰다.
- 또한 저자는 위 접근방법을 광범위한 NLP task에서 잘 동작할 수 있도록하는 deep contextual representation 방법으로 일반화시켰다.

### Layer representations
- 이전의 연구는 또한 다른 계층의 deep biRNNs이 다른 유형의 정보를 인코딩한다는 것을 보여주었다.
    - 예를 들어, deep LSTM의 하위 level에 POS tag와 같은 multi-task syntactic supervision을 도입하면 아래와 같은 고수준 task의 전반적인 성능을 올릴 수 있다.
        - [Dependency parsing (Hashimoto et el., 2017)](https://arxiv.org/abs/1611.01587)
        - [CCG super tagging (Søgaard and Goldberg, 2016)](https://www.aclweb.org/anthology/P16-2038/)
- RNN기반 encoder-decoder MT system에서 [Belinkov et al. (2017)](https://arxiv.org/abs/1704.03471)은 2-layer LSTM encoder의 첫 번째 layer에서 학습된 표현이 두 번째 layer보다 POS tag를 예측하는데 더 낫다는 것을 밝혀낸 바 있다.
- 최종적으로 context2vec에서 word context를 encoding하기 위한 top layer는 word sense의 표현들을 학습하는 것으로 밝혀졌다.
- 우리는 유사한 신호들이 우리의 ELMo 표현들의 수정된 언어 모델 목표에 의해 유도된다는 것을 보여주며
- 이러한 다양한 유형의 semi-supervision 기능을 혼합한 다운스트림 작업에 대한 모델을 배우는 것은 매우 효과적이다.

### Fix pre-train weights and Add additional task-specific model capacity
- [Semi-supervised Sequence Learning, Dai and Le (2015)](https://arxiv.org/abs/1511.01432)
- [Unsupervised Pretraining for Sequence to Sequence Learning, Ramachandran et al. (2017)](https://arxiv.org/abs/1611.02683)
- 위 두 논문에선 LM, sequence AEs 그리고 task specific supervison에서의 fine tune을 사용하여 encoder-decoder pairs를 pretrain했다.
- 이와는 반대로, unlabeled data로 biLM을 pretrain한 이후에 본 논문의 저자는 weight를 고정하고 task-specific model capacity를 추가함으로써
- data와 model이 작은 경우에 downstream task에서 더 크고 풍부하고 보편적인 biLM 표현을 사용할 수 있게 되었다.

## 3. EMLo: Embedding from Language Models

![elmo2](https://user-images.githubusercontent.com/37775784/76699461-fd88b300-66f0-11ea-9b98-1884d8baf92d.PNG)

    - Img 출처: https://brunch.co.kr/@learning/12

- Sec 3.1: char CNN으로 two-layer biLMs을 계산
- Sec 3.2: char CNN은 internal network states의 linear function
    - LSTM 내부 state를 linear combination으로 학습(?? 맞게 이해한건가 아래 살펴보세)
    - 여기가 중요 포인트!
- 이러한 setup은 semi-supervised learning을 가능케 함 (rich한 표현법을 배우니까!)
- Sec 3.3: 이미 존재하는 광범위한 NLP architectures에 쉽게 통합(적용) 가능하다.
- Sec 3.4: BiLM은 large-scale에서 사전학습됨

### 3.1 Bidirectional language models
- [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108)과 유사(공유하는 parameter가 있냐 없냐의 차이)
- 기존의 방법론이라고 이해해도 무방
- Given a seqence N tokens, <img src="https://latex.codecogs.com/svg.latex?\Large&space;(t_1,t_2,\dots,t_N)"/>에 대해,

#### forward LM
- forward language model은 history <img src="https://latex.codecogs.com/svg.latex?\Large&space;(t_1,\docts,t_{k-1})"/>가 주어졌을 때 token <img src="https://latex.codecogs.com/svg.latex?\Large&space;t_k"/>을 아래와 같이 확률을 모델링하여 sequence의 확률을 계산(Baysian Theorem, 조건부확률의 곱)

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(t_1,t_2,\dots,t_N)=\prod_{k=1}^{N}p(t_k|t_1,t_2,\dots,t_{k-1})"/>

- 16, 17년도의 SOTA neural LM 모델들은 token embedding 혹은 char-CNN을 통해 context-independent(맥락 독립적) token representation <img src="https://latex.codecogs.com/svg.latex?\Large&space;x_k^{LM}"/>을 계산하고 forward LSTM의 <img src="https://latex.codecogs.com/svg.latex?\Large&space;L"/> layer를 통과시킴
    - [Exploring the limits of language modeling](https://arxiv.org/abs/1602.02410)
    - [On the state of the art of evaluation in neural language models](https://arxiv.org/abs/1707.05589)
- 각 position <img src="https://latex.codecogs.com/svg.latex?\Large&space;k"/>에서 각 LSTM layer는 context-dependent한 표현 <img src="https://latex.codecogs.com/svg.latex?\Large&space;\overrightarrow{h}_{k,j}^{LM},\;where\;j=1,\dots,L"/>을 산출
- LSTM의 top layer의 output <img src="https://latex.codecogs.com/svg.latex?\Large&space;\overrightarrow{h}_{k,L}^{LM}"/>은 sotfmax layer를 거친 다음, 다음 토큰 <img src="https://latex.codecogs.com/svg.latex?\Large&space;t_{k+1}"/>을 예측하기 위해 사용

#### backward LM
- backward LM은 전체 seqence를 거꾸로 계산하는 것 외에 forward LM과 유사
- future context가 주어지면 이전 token을 예측

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;p(t_1,t_2,\dots,t_N)=\prod_{k=1}^{N}p(t_k|t_{k+1},t_{k+2},\dots,t_{N})"/>

- <img src="https://latex.codecogs.com/svg.latex?\Large&space;(t_{k+1},\dots,t_N)"/>가 주어졌을 때 token <img src="https://latex.codecogs.com/svg.latex?\Large&space;t_k"/>을 표현해야함
- 위 토큰에 대한 표현 <img src="https://latex.codecogs.com/svg.latex?\Large&space;\overleftarrow{h}_{k,j}^{LM}"/>은 <img src="https://latex.codecogs.com/svg.latex?\Large&space;L"/>개의 backward LSTM layer j로 이루어진 deep model로 얻을 수 있고
- 이에 대한 구현은 위에서 기술한 점만 유념하여 forward LM과 유사하게(analogous) 구현 가능.

#### biLM
- biLM은 위의 forward LM와 backward LM을 결합
- Objective Function은 forward와 backward 방향의 log likelihood를 jointly maximize한다!

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sum_{k=1}^{N}(\overbrace{\log\;p(t_k|t_1,\dots,t_{k-1};\Theta_x,\overrightarrow{\Theta}_{LSTM},\Theta_s)}^{forward\;LM}+\overbrace{\log\;p(t_k|t_{k+1},\dots,t_N;\Theta_x,\overleftarrow{\Theta}_{LSTM},\Theta_s)}^{backward\;LM})"/>

- 각 방향의 LSTM parameter는 다르게 (<img src="https://latex.codecogs.com/svg.latex?\Large&space;\overrightarrow{\Theta}_{LSTM}"/>, <img src="https://latex.codecogs.com/svg.latex?\Large&space;\overleftarrow{\Theta}_{LSTM}"/>),
- token representation(<img src="https://latex.codecogs.com/svg.latex?\Large&space;\overrightarrow{\Theta}_{x}"/>)과 softmax layer(<img src="https://latex.codecogs.com/svg.latex?\Large&space;\overrightarrow{\Theta}_{s}"/>)의 paremeter는 동일하게 tie
- 위 formulation은 [Peters et al, (2017)](https://arxiv.org/abs/1705.00108)의 접근법과 유사
    - 차이점은 token representation과 softmax layer의 paremeter를 공유하냐 vs 공유하지 않냐

### 3.2 ELMo
- **biLM layer들의 linear combination으로 word representation을 학습시키는 접근법 제시**
- ELMo에서 정말 중요한 부분!
- ELMo는 biLM의 intermediate layer representations들의 task specific combination
    - 뭔 소리여.. 쭉 읽어봅시다~
- 각 토큰 <img src="https://latex.codecogs.com/svg.latex?\Large&space;t_k"/>에 대해, <img src="https://latex.codecogs.com/svg.latex?\Large&space;L"/> layer biLM은 <img src="https://latex.codecogs.com/svg.latex?\Large&space;2L+1"/>개의 representation set을 계산

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
    - scalar parameter <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma^{task}"/>는 task model이 전체 ELMo vector의 scale을 조정
    - <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma"/>는 최적화 과정을 돕는 실질적으로 중요한 과정
        - Suplemental material for details 참고
- 고려할 부분
    - 각 biLM layer의 activation은 각기 다른 분포를 가짐
    - 몇몇 경우에 이는 weighting하기 전에 각 biLM layer에 [layer normalization](https://arxiv.org/abs/1607.06450)를 적용하는데 도움이 됨.

### 3.3 Using biLMs for supervised NLP tasks
- Given a pre-trained biLM and a supervised architecture for a target NLP task,
- biLM을 사용하여 task model을 증진시키는 것은 아주 간단쓰 쉽쓰 오호호

#### Description
- First consider; biLM이 없는 지도학습 모델의 최하단 layer
    - 대부분의 supervised NLP 모델은 최하단 layer에서 architecture를 공유,
    - 이는 ELMo를 일관되고 통합된 방식으로 추가할 수 있게 해줌
    - 주어진 tokens <img src="https://latex.codecogs.com/svg.latex?(t_1,\dots,t_N)"/>의 sequence에 대해,
    - pre-training한 word embedding 혹은 선택적으로 char 기반 표현을 사용하여 각 토큰 위치에 대해 context-독립적인 토큰 표현 <img src="https://latex.codecogs.com/svg.latex?x_k"/>를 형성
    - 그 다음, 모델은 일반적으로 bidirectional RNN, CNN, DNN을 사용하여 context-sentitive 표현인 <img src="https://latex.codecogs.com/svg.latex?h_k"/>를 형성
- Second; ELMo를 supervised model에 추가하기 위해서
    - 첫 번째로 biLM의 가중치를 freeze시키고
    - ELMo vector <img src="https://latex.codecogs.com/svg.latex?ELMo_k^{task}"/>와 <img src="https://latex.codecogs.com/svg.latex?x_k"/>를 연결(concatenate)하고 task RNN에 ELMo의 향상된 표현인 <img src="https://latex.codecogs.com/svg.latex?[x_k;ELMo_k^{task}]"/>를 전달
    - SNLI, SQuad와 같은 몇몇 task의 경우 아래 과정을 거쳐 추가적인 개선이 존재하는지 관찰
        - output specific linear weights의 다른 set을 도입 및 <img src="https://latex.codecogs.com/svg.latex?h_k"/>를 <img src="https://latex.codecogs.com/svg.latex?[h_k;ELMo_k^{task}]"/>로 대체하여
        - task RNN의 output에 ELMo를 포함
    - supervised model의 다른 부분은 변경되지 않고 유지되므로 위와 같은 addition은 더 복잡한 neural 모델의 context 내에서 발생
        - `As the remainder of the supervised model remains unchanged, these additions can happen within the context of more complex neural models` 해석 ㄷㄷ
    - 이에 대한 내용은 Sec 4. SNLI를 참고
- Finally; ELMo에 아래 행위를 취하면 더 효과적이라는 것을 발견
    - 일정량의 [`dropout`](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)을 추가
    - Loss function에 <img src="https://latex.codecogs.com/svg.latex?\gamma||w||_2^2"/>를 더하여 ELMo의 가중치를 정규화
    - 위 내용은 모든 biLM layer들의 평균에 가깝도록 ELMo weights에 [inductive bias](https://en.wikipedia.org/wiki/Inductive_bias)를 부과함

### 3.4 Pre-trained bidirectional language model architecture
- 본 논문의 pre-trained biLM은 아래 두 논문과 구조가 유사
    - [Exploring the Limits of Language Modeling, Jozefowicz et al. (2016)](https://arxiv.org/abs/1602.02410)
    - [Character-aware neural language models, Kim et al. (2015)](https://arxiv.org/abs/1508.06615)
- 다른 점은 (1) 양 방향으로 joint하여 학습시키는 점과 (2) LSTM layer들 사이에 `residual connection`을 추가한 점
- Large scale biLMs에 초점을 맞춰서 작업을 진행
    - [Peters et al. (2017)](https://arxiv.org/abs/1705.00108) 논문에서 forward-only LMs보단 biLMs, 그리고 large scale training을 사용하는 것이 중요하다고 강조
- [Jozefowicz et al. (2016)](https://arxiv.org/abs/1602.02410)의 single best model `CNN-BIG-LSTM`의 모든 embedding과 hiddem 차원을 반으로 줄임. 왜?
    - 순전히 문자 기반 입력 표현을 유지하면서
    - 전반적인 언어 모델 복잡성과 다운스트림 작업에 대한 모델 크기 및 계산 요구 사항의 균형을 맞추기 위해!
- 최종 모델은 4,096 unit과 512D projection 그리고 1->2로 가는 residual connection을 가진 <img src="https://latex.codecogs.com/svg.latex?L=2"/> biLSTM layers를 사용
- Context insensitive type representation은 아래 두 filter/projection을 사용
    - 2,048 character n-gram convolution filters followed by two highway layers ([Srivastava et al., 2015](https://arxiv.org/abs/1507.06228))
    - linear projection down to a 512 representation
    - 갑자기..?
- 그 결과로 biLM은 순수 문자 입력(다른 POS tagging등이 필요X)으로 인해 train set 이외의 token을 포함한 각 input token의 표현의 three-layer를 제공
- 대조적으로, 기존의 word embedding 방법들은 해당 token의 1-layer적인 표현밖에 제공하지 못함 (3-layer가 더 rich함)
- [1B Word Benchmark (Chelba et al., 2014)](https://arxiv.org/abs/1312.3005)에서 10 epoch 학습시킨 후에,
- forward `CNN-BIG-LSTM`의 perplexity가 30.0인데 비해 forward LM과 backward LM의 평균 perplexity는 39.7이었다. (응? 왜 더 나쁘지 ㄷㄷ;;)
- 일반적으로, backward value가 조금 낮긴 하지만, forward와 backward perplexity가 근사적으로 같다는 것을 확인
- Pre-training 후, biLM은 각 task에 대해 표현을 계산
- 몇몇 경우에 domain 특화된 data에 biLM을 fine-tuning시키는 것은 perplexity를 상당히 낮춰주고 downstream task 성능을 향상시킴
- 이는 biLM을 위한 domain transfer의 한 유형이라고 볼 수 있음
- 대부분의 경우에 downstream task에 fine-tuning된 biLM을 사용
-supplemental material for details를 살펴보길.

## 4. Evaluation
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
- 11 member ensemble F1은 87.4%로 리더보드에 submit할 당시 전체 1위였삼 ㅎㅎ
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

## 5. Analysis
- Sec 5.1은 deep contextual representation을 사용하는 것이 우수하다는 것을 보여줌
- Sec 5.3은 biLMs에 의해 포착된 contextual information을 탐구하고 이를 두 개의 intrinsic eval으로 lower layer에선 syntatic 정보가 잡히고 higher layer에선 semantic information이 포착됨을 보임
- Sec 5.2에서 ELMo를 task model에 적용했을 시의 sensitivity를 분석하고
- Sec 5.4에선 training set size에 대한 얘기를,
- Sec 5.5에선 ELMo가 각 task에서 학습한 가중치를 시각화하도록 하겠음

### 5.1 Alternate layer weighting schemes
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

### 5.2 Where to include ELMo?
- 본 논문의 Challenging 6 NLP task는 biRNN의 최하단 layer의 input으로만 word embedding을 포함함
- 그러나, 저자는 task-specific architecture에서 ELMo를 biRNN의 output에 포함시켰을 경우 몇몇 task에서 성능이 향상되는 것을 발견함.

![elmotable3](https://user-images.githubusercontent.com/37775784/76726679-3d08db00-6795-11ea-945a-bc3f3bbae49b.PNG)

- SRL제외 output에 이를 단순하게 input layer에만 ELMo를 추가한 것 보다 output layer와 input layer 동시에 ELMo를 포함시켰더니 성능이 뛰었다.
- SRL과 coreference resolution은 input layer에만 넣었을 때 성능이 좋았음
- 왜 그럴까 저자는 이유로 아래와 같이 설명하고 있음
    - SNLI와 SQuAD architecture는 biRNN 다음에 attention layer를 사용
    - ELMo를 소개시키는 것은 모델로 하여금 biLM의 내부 표현에 집중케 만듦
    - STL의 경우, biLM으로 나온 것보다 task-specific context representation이 더 중요함

### 5.3 What information is captured by the biLM's representations?
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

#### Word sense disambiguation
- 주어진 문장에 대해 [context2vec: Learning Generic Context Embedding with Bidirectional LSTM, Melamud et al., 2016](https://www.aclweb.org/anthology/K16-1006/)와 유사하게 우리는 biLM 표현을 simple 1-nearest neighbor 접근법을 사용하여 target word의 sense를 예측하는데 사용할 수 있다.
- 이렇게 하기 위해서, SemCor 3.0, 우리의 학습 말뭉치([Using a Semantic Concordance for Sense Identification, Miller et al., 1994](https://www.aclweb.org/anthology/H94-1046/))의 모든 단어에 대해 biLM을 활용하여 표현을 계산하고 모든 sense에 대해 평균을 취한 표현을 얻는다.
- test할 시에 우리는 주어진 target word에 대해 biLM으로 표현을 재계산하고 학습 셋에서 가장 가까운 sense를 얻는다.
- 훈련 중 관찰되지 않은 Lemma의 경우 WordNet의 first sense로 돌아간다.
- pass

#### POS tagging
- pass

#### Implications for supervised tasks
- pass

### 5.4 Sample efficiency
- pass

### 5.5 Visualization of learned weights
- pass

## 6. Conclusion
- biLM으로부터 얻은 고품질 deep context-dependent 표현을 학습하기 위한 일반적인 접근 방법을 소개
- 광범위한 NLP task에 ELMo를 적용했을 때 성능을 크게 향상시키고 SOTA달성한 것을 보임
- 여러 시험을 통해(`Through ablations and other controlled experiments,`) biLM layers가 맥락 속의 단어에 대한 syntatic/semantic 정보의 다양한 유형을 효율적으로 encoding할 수 있다는 것을 확인
- 모든 layer를 사용하는 것이 전반적인 task 성능을 올릴 수 있었음
