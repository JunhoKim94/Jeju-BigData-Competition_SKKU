from torch import nn, tensor
import torch
import torch.nn.functional as F

class Regression_Model(nn.Module):

    def __init__(self, inputs, hidden_layer = 256, drop_out = 0.5):
        super(Regression_Model, self).__init__()
        self.inputs = inputs

        self.linear = nn.Sequential(
            nn.Linear(self.inputs,hidden_layer),
            nn.Dropout(drop_out),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_layer,hidden_layer),
            nn.Dropout(drop_out),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_layer,hidden_layer),
            nn.Dropout(drop_out),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_layer,hidden_layer),
            nn.Dropout(drop_out),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_layer,1)
            )
        

    def forward(self, x):

        y_pred = self.linear(x)

        return y_pred