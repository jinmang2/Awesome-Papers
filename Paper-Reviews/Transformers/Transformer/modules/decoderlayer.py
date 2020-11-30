import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttentionLayer
from positionwise_feedforward import PositionwiseFeedforwardLayer


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = \
            MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = \
            MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = \
            PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.layer_norm(trg + self.dropout(_trg))
        return trg, attention
