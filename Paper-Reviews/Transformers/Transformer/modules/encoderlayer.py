import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttentionLayer
from positionwise_feedforward import PositionwiseFeedforwardLayer


class EncoderLayer(nn.Module):
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
        self.positionwise_feedforward = \
            PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.layer_norm(src + self.dropout(_src))
        return src
