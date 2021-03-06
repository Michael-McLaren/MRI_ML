
import matplotlib.pyplot as plt
import torch.utils.data as Data
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import importlib.util
'''
Example for running on terminal with the args
python Model.py -i ${input_dir} -o ${output_dir} --epochs 2 --seed 1337

'''

from TwoCUM_copy import TwoCUMfittingConc
from TwoCUM_copy import TwoCUM



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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(150, 200)  # hidden layer
        self.hidden2 = torch.nn.Linear(200, 100)  # hidden layer
        self.hidden3 = torch.nn.Linear(100, 50)
        self.hidden4 = torch.nn.Linear(50, 300)
        self.hidden5 = torch.nn.Linear(300, 200)
        self.predict = torch.nn.Linear(200, 3)

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.hidden5(x))
        x = self.predict(x)  # linear output
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # if dropout is implemented
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        prediction = model(data)  # input x and predict based on x
        loss = loss_fn_TOCHANGE(prediction, target)
        train_loss += loss.item()
        optimizer.zero_grad()  # clear gradients so it doesn't stack up over the loops
        loss.backward()  # backprop

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    return train_loss


def test(args, model, device, test_loader):
    model.eval()  # once again - for dropout
    test_loss = 0
    with torch.no_grad():  # deactivate autograd
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn_TOCHANGE(output, target)
            test_loss += loss.item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    return test_loss


# arguments to add in the terminal line
# a convenience for ssh and running from terminal
def construct_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N', help='input batch size for testing '
                                          '(default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')

    parser.add_argument('-i', '--input', required=True, help='Path to the '
                                                             'input data for the model to read')
    parser.add_argument('-o', '--output', required=True, help='Path to the '
                                                              'directory to write output to')
    return parser


def main(args):
    # TODO: add checkpointing
    # This checks if it can run off gpu as well as if it should
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.no_cuda and not use_cuda:
        raise ValueError('You wanted to use cuda but it is not available. '
                         'Check nvidia-smi and your configuration. If you do '
                         'not want to use cuda, pass the --no-cuda flag.')
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using device: {torch.cuda.get_device_name()}')


    config_args = [str(vv) for kk, vv in vars(args).items()
                   if kk in ['batch_size', 'lr', 'gamma']]
    model_name = '_'.join(config_args)

    if not os.path.exists(args.output):
        print(f'{args.output} does not exist, creating...')
        os.makedirs(args.output)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #load the data
    # In the form (batch, data - len 153)
    inputs = np.load(args.input)
    data = inputs[:, :150]
    targets = inputs[:, 150:]
    if targets.shape[1] != 3:
        print('Targets should have 3 values not ', targets.shape[1])
        sys.exit()

    transform_data = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_targets = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data_transformed = transform_data(data)
    train_loader = torch.utils.data.DataLoader(data_transformed,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    targets_transformed = transform_targets(targets)
    test_loader = torch.utils.data.DataLoader(targets_transformed,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    log_fh = open(f'{args.output}/{model_name}.log', 'w')  # possibly change this line?
    print('epoch,trn_loss,vld_loss', file=log_fh)
    # decays learning rate every step size to lr * gamma
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # defo change the the schedule

    # note sure what sys.float_info does - its just the largest value possible
    best_loss = sys.float_info.max
    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        vld_loss = test(args, model, device, test_loader)
        print(f'{epoch},{loss},{vld_loss}', file=log_fh)
        scheduler.step()
        if vld_loss < best_loss:
            best_loss = vld_loss
            torch.save(model.state_dict(),
                       f"{args.output}/{model_name}.best.pt")

    torch.save(model.state_dict(), f"{args.output}/{model_name}.final.pt")

    log_fh.close()


if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    main(args)



