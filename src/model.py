import torch
import torch.nn as nn




class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.fc = nn.Linear(32, 16)


    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        return self.fc(x)




class SiameseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = Encoder(input_dim)


    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2