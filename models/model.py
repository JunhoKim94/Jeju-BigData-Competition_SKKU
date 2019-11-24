from torch import nn, tensor
import torch
import torch.nn.functional as F

class Regression_Model(nn.Module):

    def __init__(self, inputs):
        super(Regression_Model, self).__init__()
        self.inputs = inputs

        self.linear = nn.Sequential(
            nn.Linear(self.inputs,10),
            nn.ReLU(inplace = True),
            nn.Linear(10,3),
            nn.ReLU(inplace = True),
            nn.Linear(3,1))
        

    def forward(self, x):

        y_pred = self.linear(x)

        return y_pred