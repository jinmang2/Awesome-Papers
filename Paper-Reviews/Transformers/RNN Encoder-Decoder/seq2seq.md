# RNN Encoder-Decoder for SMT
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://www.aclweb.org/anthology/D14-1179/)

- :smile: 이 논문이 어떤 논문일까요? 소개해드릴게요!
- Authored by: [MyungHoon Jin](https://www.github.com/jinmang2)

---
## Information

### :family: Authors
- **Kyunghyun Cho**, Bart van Merrienboer, Caglar Gulcehre
  - Université de Montréal
- Dzmitry Bahdanau
  - Jacobs University, Germany
- Fethi Bougares Holger Schwenk
  - Universite du Maine, France
- Yoshua Bengio
  - Université de Montréal, CIFAR Senior Fellow

### :rocket: Publish
- **Arxiv** Submission history
  From: KyungHyun Cho
  [v1] Tue, 3 Jun 2014 17:47:08 UTC (875 KB)
  [v2] Thu, 24 Jul 2014 20:07:13 UTC (460 KB)
  [v3] Wed, 3 Sep 2014 00:25:02 UTC (551 KB)

- Volume: **EMNLP 2014**
  Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing
  [Publishing](https://www.aclweb.org/anthology/volumes/D14-1/)
  [emnlp2014.org](https://emnlp2014.org/)

- Publisher: **Association for Computational Linguistics**

### :dart: 논문의 목적
- **Seq2Seq** 모델의 제안
- **Gated Recurrent Unit (GRU)** 도입

---
## Introduction

#### :one: DNN의 성공적인 사례들, SMT에서도 효과적!
Deep Neural Network는 object dectection / speech recognition에서 큰 성과를 거둠
게다가 NLP의 여러 task에서도 성공적으로 사용되고 있음
- Language Modeling
- Paraphrase Detection
- Word Embeddign Extraction

DNN을 SMT에 적용시켰을 때의 성능은 이미 보장되어있음 ^^

#### :two: Seq2Seq과 GRU를 소개
**Seq2Seq**
_Sequence_ to _Sequence_ 의 약자
phrase 기반의 **Novel Neural Network architecture**
`RNN Encoder-Decoder`: 2 개의 RNN이 Encoder와 Decoder part로 구성
- Encoder: Map `variable-length source sequence` to `fixed-length vector`
- Decoder: Map `vector representation` to `variable-length target sequence`
위 두 network는 $p(s_{target}|s_{source})$를 최대로 만들며 **동시에** 학습됨

**GRU(Gated Recurrent Unit)**
`Ch2.3 Hidden Unit that adaptively Remembers and Forgets`에서 소개된 모델!
LSTM과 비슷한, 그러나 더 쉽게 계산 및 구현이 가능!
새로운 형태의 `hidden unit`
`memory capacity`와 `ease of training`를 향상시키기 위해 고안된 **정교한** `hidden unit`

#### :three: 변역을 위한 Scoring Approach
Translate English to French
모델이 English phrase에 대응되는 French phrase의 확률을 학습하도록 훈련
위 모델을 phrase 기반의 표준 SMT system으로 사용
- `phrase table`의 각 `phrase pair`에 점수를 매겨서
- `(input_seq, output_seq)` pair의 `p(output_seq|innput_seq)`을 사용!

이렇게 적용했을 때 SMT 번역 성능이 상승했음!

#### :four: 정성적 분석 & 정성적 분석(간접적)
기존의 번역 모델과 본 논문의 학습된 `RNN Encoder-Decoder` 모델의 phrase score를 비교, 정석적인 분석을 실시
정성적인 분석은 `Encoder-Decoder`가 phrase table의 `linguistic regularities`를 잘 포착하고
간접적으로 번역 성능의 정량적인 향상을 설명 가능케 함
또한 `RNN Encoder-Decoder`는 semantic, syntactic한 phrase의 구조의 `continuous space representation`를 학습한다는 것을 밝힘

---
## RNN Encoder-Decoder

#### :one: RNN (Recurrent Neural Network)
RNN은 $h$, $y$, $x=(x_1,x_2,\dots,x_T)$로 이루어진 neural network
- $h$: hidden state
- $y$: optional output (one or many)
- $x$: variable-length sequence

각 time step $t$마다 $x_t$와 비선형 활성화 함수 $f$를 사용하여
$$h_{<t>}=f(h_{<t-1>},x_t)\cdots(1)$$
로 업데이트 ($f$는 `logistic sigmoid function` 혹은 `LSTM`)

RNN은 다음 symbol을 예측하게 학습, 전체 sequence의 확률 분포를 배운다.
이 경우, 각 timestep $t$에서 output은 아래와 같은 조건부 확률
$$p(x_t|x_{t-1},\dots,x_1)$$
예를 들어 `multinomial distribution`(1-of-K coding)은 `softmax` 함수를 사용해 출력 가능
$$p(x_{t,j}=1|x_{t-1},\dots,x_1)=\cfrac{exp(w_j h_{<t>})}{\sum_{j^\prime=1}^{K}exp(w_j^\prime h_{<t>})}\cdots(2)$$
$$where\,j=[1,K]: \text{possible symbols}\text{ \& }w_j: \text{rows of a Weight Matrix }W$$

위 확률을 결합하여 sequence $x$의 확률을 계산 가능
$$p(x)=\prod_{t=1}^{T}p(x_t|x_{t-1},\dots,x_1)\cdots(3)$$

위와 같이 분포를 학습하여 각 time step별 symbol을 반복적으로 추출, 새로운 sequence를 생성

#### :two: Seq2Seq
확률적인 관점에서 seq2seq은 주어진 sequence에 대해 target sequence가 등장할 아래 조건부 확률을 학습
$$p(y_1,\dots,y_T|x_1,\dots,x_{T^\prime})$$
$note\;that:\;T\text{ is not always equal to }T^\prime$

- _**Encoder**_
Encoder는 `RNN`으로 input sequence $x$의 symbol을 sequentially하게 읽음
각 symbol을 불러올 때마다 `RNN의` hidden state $h$는 $Eq(1)$과 같이 updated
sequence를 전부 읽은 후의 `RNN의` hidden state는 전체 input sequence의 요약 $c$

- _**Decoder**_
제안된 모델의 Decoder는 hidden state $h$가 주어졌을 때 다음 symbol $y_t$를 예측하여 output sequence를 생성하는 `RNN`
그러나 Encoder의 `RNN`과 다르게 $y_t$와 $h$는 이전 심볼 $y_{t-1}$과 input sequence의 요약 $c$의 영향을 받음
때문에 각 time step $t$에서 decoder의 hidden state는 아래와 같이 update
$$h_{<t>}=f(h_{<t-1>},y_{t-1},c)$$
유사하게 다음 symbol의 조건부 확률을 아래와 같이 계산
$$p(y_t|y_{t-1},y_{t-2},\dots,y_1,c)=g(h_{<t>},y_{t-1},c)$$

![img](https://user-images.githubusercontent.com/37775784/77247746-430c2980-6c77-11ea-8779-8481c64fa2c1.PNG)

- _**Objective**_
제안된 seq2seq의 두 RNN은 아래 조건부 로그우도함수를 최대로 만들며 **동시에** 학습
$$\max_{\theta}\frac{1}{N}\sum_{n=1}^{N}\log{p\theta(y_n|x_n)}\cdots(4)$$
$$where\;\theta:\text{set of the model parameters}\;\&\;(x_n,y_n):\text{(input seq, output seq) pair}$$
input seq에서 output seq을 출력하는 모든 연산 과정은 모두 **미분 가능(differentiable)** 하기 때문에 model parameter를 추정하는데 `gradient-based algorithm`을 사용

**Usage**

`seq2seq`을 학습시킨 후에 모델을 아래 두 가지 방법으로 활용
- input sequence가 주어졌을 때, 모델을 활용하여 target sequence를 생성
- $Eq(3)$과 $Eq(4)$의 확률값 $p\theta(y|x)$로 input과 output sequence pair에 score 부여

#### :three: GRU; Hidden Unit that adaptively Remembers and Forgets
LSTM과 비슷한, 그러나 더 쉽게 계산 및 구현이 가능한 새로운 형태의 `hidden unit`을 제안 ($Eq(1)$의 $f$)

<img src="https://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/10/Screen-Shot-2015-10-23-at-10.36.51-AM.png" width="80%" height="40%">

**reset gate**

`reset gate` $r_j$는 아래와 같이 계산됨
$$r_j=\sigma({[W_r{x}]}_j+{[U_r{h_{t-1}}]}_j)$$
- $\sigma$: `logistic sigmoid function`
- ${[.]}_j$: vector의 j번째 요소
- $x$: input
- $h_{t-1}$: previous hidden state

**update gate**

`update gate` $z_j$도 `reset gate`와 유사하게 계산됨
$$z_j=\sigma({[W_z{x}]}_j+{[U_z{h_{t-1}}]}_j)$$

**hidden unit**

제안된 hidden unit $h_j$의 실제 activation은 아래와 같이 계산됨
$$h_j^{<t>}=z_j h_j^{<t-1>}+(1-z_j) \tilde{h}_j^{<t>}$$
$$\tilde{h}_j^{<t>}=\phi({[Wx]}_j+{[U(r{\odot}h_{t-1})]}_j)$$

**_Explane formulation_**

`reset gate`가 0에 가까워지면 hidden state는 이전 상태를 무시하고 입력값으로 reset
- 이는 hidden state가 이 후 symbol과 관련이 없는 정보를 drop하게 만들고 더욱 compact한 표현을 만듦

반대로 `update gate`는 이전 hidden state를 현재 hidden state에 얼마나 반영할지를 결정
- 이는 LSTM의 memory cell과 유사하며 RNN이 장기 정보를 기억하도록 만듦
- 이는 Bengio 교수 논문 [Advances in Optimizing Recurrent Networks](https://arxiv.org/abs/1212.0901)의 `leaky integraion unit`의 변형된 형태로 고려할 수 있음
    - LSTM에서 long term dependency를 조절하는 개념으로 추정

**_Long-Short Dependencies related on reset and update gate_**

각 hidden unit이 분리된 `reset gate`와 `update gate`를 가지고 있음
때문에 각 hidden unit은 다른 time scale의 dependencies를 모델이 이해하도록 학습 가능
특성에 따라 길게 볼지 짧게 볼지 선택 가능하다는 얘기

이렇게 **short dependencies** 를 배운 unit은 `reset gate`가 활성화되는 경향을 보이고
반대로 **longer dependencies** 를 포착하는 unit은 주로 `update gate`가 활성화됨

실험적으로 저자는 위의 새로운 unit을 사용하는 것이 중요하고
흔히 자주 사용하는 `tanh`로는 유의미한 성능을 얻지 못했다고 함

---
## Experiments and Qualitative & Quantitative Analysis
생략!! SMT와 Word and Phrase Representations에 대한 실험 및 결과를 제시
뭐 좋았다 잘 배운다 잘 표현한다 끝입니다.

---
## Understand with codes!
본 논문에서는 `English`를 `French`로 번역하는 task를 풀었지만
본 예제는 `Deutsch`를 `English`로 번역하는 task를 풀며 구현을 할 것임.

### Preparing Data
- Import Library
```python
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time
```
- Set a `random seed` for deterministic results/reproducability
```python
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```
- Instantiate English and German spaCy models
```python
# with permissions,
# python -m spacy download de
# python -m spacy download en

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
```
- Define tokenizer and Create fields to process data
```python
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',    # start of sentence
            eos_token='<eos>',     # end of sentence
            lower=True)

TRG = Field(tokenize = tokenize_en,
            init_token='<sos>',    # start of sentence
            eos_token='<eos>',     # end of sentence
            lower=True)
```
- Load data and Print out example data
```python
train_data, valid_data, test_data = Multi30k.splits(
    exts = ('.de', '.en'), fields = (SRC, TRG))
print('train length: {0}, valid length: {1}, test length: {2}'
    .format(len(train_data), len(valid_data), len(test_data)))

sample_pair = vars(train_data.examples[0])
print(f"src : {sample_pair['src']}")
print(f"trg : {sample_pair['trg']}")
```
```
train length: 29000, valid length: 1014, test length: 1000
src : ['zwei', 'junge', 'weiße', 'männer', 'sind', 'im', 'freien', 'in', 'der', 'nähe', 'vieler', 'büsche', '.']
trg : ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']
```
- Create vocab
```python
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
```
- Define `device` and create data iterator
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)
```

### Building the Seq2Seq model
$\text{note that}:$ it only requires and returns a hidden state.
there is no cell state like in the LSTM

$h_t\qquad=GRU(e(x_t), h_{t-1})$
$(h_t,c_t)=LSTM(e(x_t), h_{t-1}, c_{t-1})$
$h_t\qquad=RNN(e(x_t), h_{t-1})$

#### Encoder
위에서 설명한 GRU로 바로 구현하면 끝.

$h_t=EncoderGRU(e(x_t), h_{t-1})$

![GRUEncoder](https://github.com/bentrevett/pytorch-seq2seq/raw/e8209a7b0207cde55871be352819cac3dd5c05ce/assets/seq2seq5.png)

```python
# The encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        # layers
        self.embedding = nn.Embedding(input_dim, emb_dim) # no dropout as only one layer!
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src's shape (src_len, batch_size)
        embedded = self.dropout(self.embedding(src))
        # embedded's shape (src_len, batch_size, emb_dim)
        outputs, hidden = self.rnn(embedded) # no cell state!
        # outputs' shape (src_len, batch_size, hid_dim*n_directions)
        # hidden's shape (n_layers*n_directions, batch_size, hid_dim)

        # outputs are always from the top hidden layer
        return hidden
```

#### Decoder
$y_t$와 $h$는 이전 심볼 $y_{t-1}$과 input sequence의 요약 $c$ (context vector)의 영향을 받음.
여기서 $c$를 $z$, Decoder의 hidden state를 $h$대신 $s$, embedding을 $d$로 표기하여 아래와 같이 업데이트

$s_t=DecoderGRU(d(y_t), s_{t-1}, z)$

$\text{note that: }z\text{(context vector) does not have a }t\text{ substript.}$
- $\text{that is, }$Encoder에서 나온 **context vector** 를 재사용한다는 의미.
- 이게 사실상 굉장히 중요한데, 이는 **모델의 병목 현상을** 유발하여 성능을 저하시킴.
- 때문에 이는 **_Attention_** 의 태동을 알리는 계기가 되었음

다음 토큰의 조건부 확률을 아래의 $linear\;layer\;f$로 계산

$\hat{y}_{t+1}=f(d(y_t),s_t,z)$

![GRUDecoder](https://github.com/bentrevett/pytorch-seq2seq/raw/e8209a7b0207cde55871be352819cac3dd5c05ce/assets/seq2seq6.png)

Within the implementation,
- $d(y_t)$와 $z$를 `concatenate`하여 `GRU`의 input으로 넣어줌
- 즉, `GRU`의 input dimension은 `emb_dim + hid_dim` (`hid_dim` is context vector's dim)
- `linear layer`에서 $d(y_t)$, $s_t$ 그리고 $z$를 `concatenate`하여 input으로 넣어줌
- 즉, `linear layer`의 input dimension은 `emb_dim + hid_dim * 2`

```python
# The decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        # layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input's shape (batch_size)
        # hidden's shape (n_layers*n_directions, batch_size, hid_dim)
        # context's shape (n_layers*n_directions, batch_size, hid_dim)
        # n_layers and n_directions in the decoder will both always be 1.

        input = input.unsqueeze(0)
        # input's shape (1, batch_size)
        embedded = self.dropout(self.embedding(input))
        # embedded's shape (1, batch_size, emb_dim)
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con's shape (1, batch_size, emb_dim+hid_dim)
        output, hidden = self.rnn(emb_con, hidden)
        # output's shape (seq_len, batch_size, hid_dim * n_directions)
        # hidden's shape (n_layers*n_directions, batch_size, hid_dim)
        # since n_layers = n_directions = 1,
        # output's shape is (1, batch_size, hid_dim)
        # hidden's shape is (1, batch_size, hid_dim)
        output = torch.cat((embedded.squeeze(0),
                            hidden.squeeze(0),
                            context.squeeze(0)),
                           dim=1)
        # output's shape (batch_size, emb_dim+hid_dim+hid_dim)
        prediction = self.fc_out(output)
        # prediction's shape (batch_size, output_dim)
        return prediction, hidden
```

#### Seq2Seq
Encoder와 Decoder를 연결!
![seq2seq](https://github.com/bentrevett/pytorch-seq2seq/raw/e8209a7b0207cde55871be352819cac3dd5c05ce/assets/seq2seq7.png)

**Brief as follow:**
- 모든 예측값을 저장할 `output tensor` $\hat{Y}$ 생성
- `source sequence` $X$는 `Encoder`에서 `context vector`로 변환
- 초기 `Decoder`의 `hidden state`는 `context vector`와 동일
    - $i.e.,\;s_0=z=h_T$
- `<sos>` tokens의 batch를 첫 번째 input $y_1$으로 사용
- 아래의 과정을 반복하며 decoder과정을 거침
    - `Decoder`에 `input token` $y_t$, 이전 `hidden state` $s_{t-1}$ 그리고 `context vector` $z$를 넣어줌
    - 예측값 $\hat{y}_{t+1}$과 새로운 `hidden state` $s_t$를 반환
    - `Teacher Forcing`을 수행할 것인지 결정
        - 다음 input으로 `target sequence`의 `ground truth` nextd token vs Highest predicted next token 중 무엇을 사용할 것인지!

```python
# Seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src's shape (src_len, batch_size)
        # trg's shape (trg_len, batch_size)
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacheR_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs, $\hat{Y}$
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Last hidden state of the encoder is the context
        context = self.encoder(src)

        # context also used as the initial hidden state of the decoder
        hidden = context

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)

            # place predictions in a tensor holding predictions for each tokens
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(axis=1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
```

### Training the Seq2Seq Model
- Set the embedding dimensions and the amount of dropout
- hidden dimensions must remain the same
- Initialize `seq2seq` model

```python
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2Seq(enc, dec, device).to(device)
```

- Initialize parameters
- $paremeters\sim\mathcal{N}(0,\;0.01)$
- 모델의 `apply` 메서드를 사용하면 `init_weights`함수가 모든 module과 sub-module에 호출되어 적용된다.

```python
@torch.no_grad()
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init_normal_(param.data, mean=0, std=0.01)

model.apply(init_weights)
```
```
Seq2Seq(
  (encoder): Encoder(
    (embedding): Embedding(7855, 256)
    (rnn): GRU(256, 512)
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (decoder): Decoder(
    (embedding): Embedding(5893, 256)
    (rnn): GRU(768, 512)
    (fc_out): Linear(in_features=1280, out_features=5893, bias=True)
    (dropout): Dropout(p=0.5, inplace=False)
  )
)
```
- Print out the number of parameters
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```
```
The model has 14,220,293 trainable parameters
```
- Initialize `optimizer`
```python
optimizer = optim.Adam(model.parameters())
```
- Initialize the loss function, making sure to ignore the loss on `<pad>` tokens.
```python
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
```
- Define `train` / `evaluate` function
```python
def train(model, iterator, optimizer, criterion, clip):
    # set .training = True with children modules
    model.train(mode=True)
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        # 매 매치마다 model parameter 이전 계산 노드에서 분리 및 0으로 초기화
        optimizer.zero_grad()
        # forward pass
        output = model(src, trg)
        # trg = [trg_len, batch_size]
        # output = trg_len, batch_size, output_dim

        # output의 0th elt는 게속 0. 첫 번째 원소 제거
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # trg = [(trg_len-1)*batch_size]
        # output = [(trg_len-1)*batch_size, output_dim]

        # loss function은 1d target이 주어졌을 때 2d input에 대해서만 작동
        loss = criterion(output, trg)
        # backward pass
        loss.backward()
        # Gradient Clipping: prevent them from exploding
        # RNN에서 흔한 이슈
        # Adam같이 동적인 학습률을 같은 optimizer를 사용할 경우 굳이 적용안해도 ok
        # 그러나 보안장치로 사용하는 것은 추천
        # clip을 넘겼을 때 최댓값보다 큰 만큼의 비율로 gradient를 나눠줌
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Performs a single optimization step.
        optimizer.step()
        # scalar(loss) update
        epoch_loss += loss.item()
    # get loss
    return epoch_loss / len(iterator)        
```
```python
def evaluate(model, iterator, criterion):
    # set .training = `False` with children modules
    # Turn off dropout and batch_normalization
    model.eval()
    epoch_loss = 0
    # No gradients are calculated within the block
    # This reduces memory consumption and speeds things up
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            # 매 매치마다 model parameter 이전 계산 노드에서 분리 및 0으로 초기화
            output = model(src, trg, 0) #turn off teacher forcing
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            # output의 0th elt는 게속 0. 첫 번째 원소 제거
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            # loss function은 1d target이 주어졌을 때 2d input에 대해서만 작동
            loss = criterion(output, trg)
            # optimizer step, clip을 통과시킬 필요 X
            # scalar(loss) update
            epoch_loss += loss.item()
    # get loss
    return epoch_loss / len(iterator)
```
- Define function that calculates how long an epoch tasks
```python
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

- Train our model, saving the parameters that give us the best validation loss.
```python
N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut2-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
```
```
Epoch: 01 | Time: 0m 36s
	Train Loss: 5.100 | Train PPL: 164.008
	 Val. Loss: 5.049 |  Val. PPL: 155.815
Epoch: 02 | Time: 0m 34s
	Train Loss: 4.381 | Train PPL:  79.893
	 Val. Loss: 5.161 |  Val. PPL: 174.301
Epoch: 03 | Time: 0m 34s
	Train Loss: 3.969 | Train PPL:  52.945
	 Val. Loss: 4.507 |  Val. PPL:  90.666
Epoch: 04 | Time: 0m 35s
	Train Loss: 3.522 | Train PPL:  33.851
	 Val. Loss: 4.040 |  Val. PPL:  56.820
Epoch: 05 | Time: 0m 34s
	Train Loss: 3.145 | Train PPL:  23.219
	 Val. Loss: 3.879 |  Val. PPL:  48.379
Epoch: 06 | Time: 0m 34s
	Train Loss: 2.840 | Train PPL:  17.114
	 Val. Loss: 3.746 |  Val. PPL:  42.346
Epoch: 07 | Time: 0m 35s
	Train Loss: 2.589 | Train PPL:  13.313
	 Val. Loss: 3.553 |  Val. PPL:  34.926
Epoch: 08 | Time: 0m 35s
	Train Loss: 2.344 | Train PPL:  10.422
	 Val. Loss: 3.540 |  Val. PPL:  34.459
Epoch: 09 | Time: 0m 35s
	Train Loss: 2.131 | Train PPL:   8.423
	 Val. Loss: 3.514 |  Val. PPL:  33.574
Epoch: 10 | Time: 0m 36s
	Train Loss: 1.986 | Train PPL:   7.288
	 Val. Loss: 3.560 |  Val. PPL:  35.158
```
- Test the model using "best" parameters
```python
model.load_state_dict(torch.load('tut2-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
```
```
| Test Loss: 3.514 | Test PPL:  33.586 |
```
---
## Conclusion

#### Propose Seq2Seq과, GRU
- 각기 길이가 $T,T^\prime$인 sequence들을 mapping하는 새로운 Neural Network Architecture 제안
- `seq2seq`은 sequence의 pair에 점수를 부여하거나 source seq이 주어졌을 때 target seq을 생성할 수 있음
- seqeuence를 읽거나 생성할 때 각 hidden unit이 기억 혹은 망각할지를 adaptively하게 조절할 수 있는 `reset gate`와 `update gate`를 포함하는 novel hidden unit인 `GRU`를 제안

#### Understand linguistic regularities
- phrase table의 각 phrase pair에 점수를 매겨 SMT에서 모델을 평가
- 질적으로, 새로운 모델이 phrase pair의 언어 규칙성을 잘 포착하는 것과 잘 형성된 target phrase(`번역된`으로 이해가능)를 추출하는 것을 보임

#### Orthogonality with another approach
- seq2seq을 사용, BLEU score 기준 SMT의 전반적인 성능이 향상됨
- 우리의 seq2seq은 기존의 접근법들과 직교하여(orthogonal, 유사하지 않음) 예시로 NPLM등의 모델과 같이 사용하여 성능을 더 크게 증가시킬 수 있음을 확인

#### Capture the linguistic regularities in multiple levels
- 학습 모델의 질적인 분석은 다양한 level에서 언어적 규칙성을 포착할 수 있을 것인가를 보여줌
    - 다양한 level라 함은 word level, phrase level 전부 다!
- 때문에 제안된 `seq2seq`을 다양한 nlp applications에 사용하면 성능 향상, 언어 표현력 등 다양한 이점을 얻을 수 있다는 점을 시사

#### Large potential
- `seq2seq`은 추가적인 개선과 분석을 위한 큰 잠재력을 가지고 있다.
    - `proposed architecture`라고 언급. `GRU`도 겠지.
- 한 가지 예를 들자면 `proposed architecture`을 통해 전체, 혹은 phrase table의 일부를 대체하는 것.
- 또한 제안된 모델이 `written language`에만 국한되지 않음을 기억하라.
- 제안된 모델은 `speech transcription`등 다양한 application에 적용될 수 있다.


---
## Reference
- [reniew's blog, Seq2seq](https://reniew.github.io/31/)
- [bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb)

## Related paper
