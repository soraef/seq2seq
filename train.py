import math 
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Seq2Seq, Encoder, Decoder

parser = argparse.ArgumentParser()
parser.add_argument("output_file_name", default="result.png")
parser.add_argument("--hidden_dim",     default=64,  type=int)
parser.add_argument("--n_layers",       default=3,   type=int)
parser.add_argument("--dropout_rate",   default=0.5, type=float)
parser.add_argument("--epochs",         default=100, type=int)
parser.add_argument("--optimizer",      default="adam", choices=["adam", "sgd"])
parser.add_argument("--freq",           default=60,  type=int)
args = parser.parse_args()

# データセット作成用の関数
def sin_list(data_num, offset=0, freq=60):
    return [math.sin(2 * math.pi * i / freq) for i in range(data_num)]

# データは[時系列数, バッチサイズ, 1データの次元数]という形
# 今回は20点のsin波の入力から次の20点のsin波を予測するので[20, batch_size, 1]という形になる
def mk_dataset():
    batch_size = 600
    xs = []
    ys = []
    for i in range(batch_size):
        wave = sin_list(40, i, freq=args.freq)
        x = np.array(wave[:20]).astype(np.float32)[:,np.newaxis, np.newaxis]
        y = np.array([0] + wave[20:]).astype(np.float32)[:,np.newaxis, np.newaxis]

        xs.append(x)
        ys.append(y)

    train_x = np.concatenate(xs, axis=1)
    train_y = np.concatenate(ys, axis=1)

    return train_x, train_y

# gpuの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

# 訓練データを準備
train_x, train_y = mk_dataset()

train_x = torch.from_numpy(train_x).to(device)
train_y = torch.from_numpy(train_y).to(device)

# モデルの準備
INPUT_DIM  = 1
OUTPUT_DIM = 1
N_LAYERS   = args.n_layers
HID_DIM    = args.hidden_dim
DROPOUT_RATE = args.dropout_rate

enc = Encoder(INPUT_DIM, HID_DIM, N_LAYERS, DROPOUT_RATE)
dec = Decoder(OUTPUT_DIM, INPUT_DIM, HID_DIM, N_LAYERS, DROPOUT_RATE)
model = Seq2Seq(enc, dec, device).to(device)

# モデルのパラメータの初期化
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

print(model.apply(init_weights))

# optimizerの設定
if args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=0.01)
else:
    optimizer = optim.Adam(model.parameters())

# 損失関数の設定
loss_fn = torch.nn.MSELoss()


# train
model.train()

for epoch in range(args.epochs):
    optimizer.zero_grad()

    outputs = model(train_x, train_y)

    # train_y, outputsの先頭に0が入っているので取り除く
    train_y1 = train_y[1:]
    outputs  = outputs[1:]

    loss = loss_fn(train_y1, outputs)
    loss.backward()

    optimizer.step()
    if epoch % 10 == 0:
        print(loss.item())

# test
# 今回はtrain_xの1番目のバッチをtestデータとして使う
model.eval() # model.eval()を設定するとdropoutを無効化してくれる

test_x = train_x[:, 1:2, :]
test_y = train_y[:, 1:2, :]

with torch.no_grad():
    outputs = model(test_x, test_y, 0)

# 結果の出力
x = test_x[:, 0, 0].to(cpu).detach().tolist()
y = outputs[:, 0, 0].to(cpu).detach().tolist()[1:]

plt.plot(list(range(20)), x, color="r", label="train_x")
plt.plot(list(range(19, 40)), [x[-1]] + y, color="b", label="predict_y")
plt.savefig("result/" + args.output_file_name)


