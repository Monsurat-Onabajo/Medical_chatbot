import torch
from torch import nn

class RNN_model(nn.Module):
  def __init__(self):
    super().__init__()

    self.rnn= nn.RNN(input_size=1080, hidden_size=240,num_layers=1, nonlinearity= 'relu', bias= True)
    self.output= nn.Linear(in_features=240, out_features=24)

  def forward(self, x):
    y, hidden= self.rnn(x)
    #print(y.shape)
    #print(hidden.shape)
    x= self.output(y)

    return(x)