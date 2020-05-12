import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
#         m.bias.data.fill_(0.01)

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TFiLM(nn.Module):
    """
    TFiLM model for text classification
    """

    def __init__(self, block_size, input_len, nchannels=256):
        super(TFiLM, self).__init__()
        self.pool_layer = m = nn.MaxPool1d(block_size, block_size)
        self.lstm = nn.LSTM(nchannels, nchannels)

    def forward(self, x):
        # x - minibatch x channels x len
        # Assume minibatch = 1
        lstm_input = self.pool_layer(x)
        lstm_out, _ = self.lstm(lstm_input.permute(0, 2, 1)).permute(0,2,1)
        out = lstm_out * x

class TFiLMSmallCNN(nn.Module):
    """
    TFiLMSmallCNN model for text classification
    """

    def __init__(self, block_size, out_size, lstm_dim=256):
        super(TFiLMSmallCNN, self).__init__()
        self.out_size = 
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True),
            TFilM(block_size, 0), 
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True),
            TFilM(block_size, 0), 
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=1, groups=1, bias=True),
            TFilM(block_size, 0)
        )
        self.fc = nn.Linear(out_size, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.view(x.shape[0], self.out_size))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    nchannels = 256
    input_len = 80000
    batch_sizes = [i for i in range(1, 2000) if i%100==0]
    times = [];
    net = TFiLMSmallCNN().to(device)
    for b in batch_sizes:
      tic = time.time()
      input = torch.randn(b, n_channels, input_len).to(device);
      out = net.forward(input)
      toc = time.time()
      times.append(toc-tic)
      print('times : ', times)
      throughput = [(i*100)/t for i, t in enumerate(times)]
      plt.figure(figsize=(12,8))
      plt.plot(batch_sizes, throughput)
      plt.title('GPU Performance - Throughput')
      plt.xlabel('Input size (number of queries)')
      plt.ylabel('Throughput (queries per second)')

if __name__ == "__main__":
    main()
