import torch
import numpy as np
from IPython.core.debugger import set_trace
import imageio
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import importlib.util

from TwoCUM_copy import TwoCUMfittingConc
from TwoCUM_copy import TwoCUM



def loss_fn(outputs, targets):
    return 0


AIF = np.load("MRI_other/TK_Modelling/AIF.npy")
t = np.arange(0,366,2.45)

# i will not be standardising as i want to be able to interpret the PK parameters afterwards
# define basic net
x, y = generate_xy(1000)

net = torch.nn.Sequential(
    torch.nn.Linear(150, 150),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(150, 3),
)

# loss and optimise
optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)
loss_func = torch.nn.MSELoss()

x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
inputs = Variable(x)
outputs = Variable(y)
torch_dataset = Data.TensorDataset(x, y)  # wrapper to join x and y into one dataloader

dataloader = Data.DataLoader(torch_dataset, batch_size=50,
                             shuffle=True)  # dataloader for batching and shuffle every epoch

enum = 100
for epoch in range(enum):
    for i, (mini_x, mini_y) in enumerate(dataloader):  # take out a batch for each step

        mini_x = Variable(mini_x)
        mini_y = Variable(mini_y)

        prediction = net(mini_x)  # input x and predict based on x

        loss = loss_func(prediction, mini_y)
        if i == 0:
            print(epoch, i, loss)

        optimizer.zero_grad()  # clear gradients so it doesn't stack up over the loops
        loss.backward()  # backprop
        optimizer.step()




