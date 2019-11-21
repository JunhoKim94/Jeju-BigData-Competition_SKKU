import numpy as np
import pandas as pd
from torch import nn, tensor, from_numpy
import matplotlib.pyplot as plt
import torch
from utils.preprocessing import processing
from models.model import Regression_Model

path = "./data/train.csv"

x, y = processing(path)
print(x.head())
batch_size = 1000
learning_rate= 0.01

model = Regression_Model()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(100):
    # 1) Forward pass: Compute predicted y by passing x to the model
    for i in range(len(x) // batch_size):
        seed = np.random.choice(batch_size)
        x_data = tensor(np.array(x.iloc[seed], dtype = np.float32))
        y_data = tensor(np.array(y.iloc[seed], dtype = np.float32))
        
        y_pred = model(x_data)

        # 2) Compute and print loss
        loss = criterion(y_pred, y_data)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')
