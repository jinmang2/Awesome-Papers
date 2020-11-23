# Seq2Seq with Attention Mechanism

link: [here!](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

## Introduction
기존 RNNencdec architecture는 모든 단계에서 컨텍스트 벡터 $z$를 디코더에 명시적으로 전달하고 hidden state $s_t$와 함께 컨텍스트 벡터와 내장된 입력 단어 $d(y_t)$를 모두 전달하여 예측을 함으로써 **information compression** 을 줄이는 방식으로 설정되었다.

information compression을 줄였음에도 불구하고, context vector는 여전히 source sentence에 대한 모든 정보를 담을 것을 필요로 한다. attention을 통해 이를 해소하자.

## Encoder
![encoder](https://github.com/bentrevett/pytorch-seq2seq/raw/e8209a7b0207cde55871be352819cac3dd5c05ce/assets/seq2seq8.png)

single hidden unit으로 **GRU** 사용
Bidirectional RNN 사용

$\overrightarrow{h_{t}}=\overrightarrow{EncoderGRU}(e(\overrightarrow{x_t}),\overrightarrow{h_t})$
$\overleftarrow{h_{t}}=\overleftarrow{EncoderGRU}(e(\overleftarrow{x_t}),\overleftarrow{h_t})$
$\text{where }\overrightarrow{x_0}=\text{<sos>},\;\overrightarrow{x_1}=\text{guten}\text{ and }\overleftarrow{x_0}=\text{<eos>},\;\overleftarrow{x_1}=\text{morgen}$

그림의 output을 보면 hidden dimension이 4일 때 각 방향만큼 총 8개의 output을 배출한다. 논문에서 보면 해당 양 방향에 대한 hidden state를 concatenate하여 아래와 같이 차원이 hid_dim * n_directions의 Annotation을 얻는다.
$h_j=\big[\overrightarrow{h}_j^\top;\overleftarrow{h}_j^\top\big]^\top$
$H=\{\overrightarrow{h}_j\;for\;i\;in\;[1,T]\}$


```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src shape: [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)
        # outputs shape: [src len, batch_size, hid dim * num directions]
        # hidden shape: [n layers * num directions, batch_size, hid dim]
        # in this case,
        #    n layers = 1
        #    

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer

        #hidden [-2, :, : ] is the last of the forwards RNN
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        last_hidden = torch.cat(
            hidden[-2, :, :],
            hidden[-1, :, :], dim=1)
        #
        hidden = torch.tanh(
                     self.fc(
                         torch.cat(
                             (hidden[-2,:,:],
                              hidden[-1,:,:]),
                             dim = 1)
                            )
        )

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]

        return outputs, hidden
```
