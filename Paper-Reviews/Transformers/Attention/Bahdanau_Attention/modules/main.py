from data import prepare_data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import random
import math
import time

from models import Encoder, Decoder, Attention, Seq2Seq

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    train_iterator, valid_iterator, test_iterator, params = prepare_data()
    (INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM,
    ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, DEC_DROPOUT) = params
    # INPUT_DIM = len(SRC.vocab), 7855
    # OUTPUT_DIM = len(TRG.vocab), 5893
    # ENC_EMB_DIM = 256
    # DEC_EMB_DIM = 256
    # ENC_HID_DIM = 512
    # DEC_HID_DIM = 512
    # ENC_DROPOUT = 0.5
    # DEC_DROPOUT = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM,
                  DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    for i, batch in enumerate(train_iterator):
        print(f'ITER: {i}')
        example = batch
        print("Input Length:", example.src.shape, "[src_len, batch_size]")
        output = model.forward(example.src, example.trg)
        print(output.shape)
        print('')
        if i > 3: break

if __name__ == '__main__':
    main()
