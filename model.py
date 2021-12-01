import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, layer_cnt):
        super().__init__()
        self.layer_cnt = layer_cnt
        self.ln1 = nn.Linear(in_features=32, out_features=128)
        self.ln_hint = nn.Sequential(*[nn.Linear(in_features=128, out_features=128) for _ in range(layer_cnt)])
        self.ln2 = nn.Linear(in_features=128, out_features=784)
        self.relu = nn.ReLU()

    def forward(self):
        x = torch.rand(32)
        x = self.relu(self.ln1(x))
        x = self.relu(self.ln_hint(x))
        x = self.relu(self.ln2(x))

        return x.reshape((28, 28))


class  Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ln1 = nn.Linear(in_features=784, out_features=128)
        self.ln2 = nn.Linear(in_features=128, out_features=64)
        self.ln3 = nn.Linear(in_features=64, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(784)
        x = self.relu(self.ln1(x))
        x = self.relu(self.ln2(x))
        x = self.sigmoid(self.relu(self.ln3(x)))

        return x
