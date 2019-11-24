import numpy as np
import pandas as pd
from torch import nn, tensor, from_numpy
import matplotlib.pyplot as plt
import torch
from utils.preprocessing import processing
from models.model import Regression_Model

train_path = "./data/train.csv"
test_path = "./data/test.csv"

input_var = ['in_out','latitude','longitude','6~7_ride', '7~8_ride', '8~9_ride', '9~10_ride',
       '10~11_ride', '11~12_ride', '6~7_takeoff', '7~8_takeoff', '8~9_takeoff',
       '9~10_takeoff', '10~11_takeoff', '11~12_takeoff','weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
       'weekday_5', 'weekday_6', 'dis_jejusi', 'dis_seoquipo']


train_x, train_y = processing(train_path, input_var = input_var)
test_x= processing(test_path,training = False, input_var = input_var)

batch_size = 1000
learning_rate= 0.01

model = Regression_Model(len(train_x.columns))
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(100):
    # 1) Forward pass: Compute predicted y by passing x to the model
    for i in range(len(train_x) // batch_size):
        seed = np.random.choice(batch_size)
        x_data = tensor(np.array(train_x.iloc[seed], dtype = np.float32))
        y_data = tensor(np.array(train_y.iloc[seed], dtype = np.float32))
        
        y_pred = model(x_data)

        # 2) Compute and print loss
        loss = criterion(y_pred, y_data)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch} | Loss: {loss.item()} ')
