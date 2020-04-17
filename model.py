import torch
import torch.nn as nn
import torch.nn.functional as F

import glob
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout=0.5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.dropout = nn.Dropout(dropout)
    
    # xs:  入力シーケンス[xs_len, batch_size, x_size]
    def forward(self, xs):
        xs = self.dropout(xs)
        outputs, (hidden, cell) = self.rnn(xs)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, hidden_dim, n_layers=1, dropout=0.5):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        # x = [batch_size, x_size] => [1, batch_size, x_size]
        x = x.unsqueeze(0)

        x = self.dropout(x)

        output, (hidden, cell) = self.rnn(x, (hidden, cell))

        # output = [1, batch_size, hidden_dim] => [batch_size, hidden_dim]
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device  = device

    # xs: 入力データシーケンス[xs_len, batch_size, x_size]
    # ys: 教師データシーケンス[ys_len, batch_size, y_size]
    def forward(self, xs, ys, teacher_forcing_ration = 0.5):
        batch_size = ys.shape[1]
        ys_len     = ys.shape[0]
        y_size     = ys.shape[2]

        outputs = torch.zeros(ys_len, batch_size, y_size).to(self.device)

        hidden, cell = self.encoder(xs)

        y = ys[0, :, :]

        # 教師データの長さ分だけoutputを計算する
        for t in range(1, ys_len):
            output, hidden, cell = self.decoder(y, hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ration

            y = ys[t] if teacher_force else output

        return outputs


