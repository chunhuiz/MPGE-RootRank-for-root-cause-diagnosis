import argparse
import math
import time

import torch
import torch.nn as nn
import MPGE
import numpy as np;
import importlib
import torch.nn.functional as F

from utils import *;
import Optim
import warnings
warnings.filterwarnings("ignore")

def train(data, X, Y, model, criterion, optim, batch_size,lamda):
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        model.zero_grad();
        output = model(X);
        loss1 = criterion(output, Y);
        loss_reg=lamda*torch.sum(torch.abs(model.A0)) + (0.001)*torch.sum((model.A0)**2)
        loss= loss1 + loss_reg
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.data.item();
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples

def evaluate(data, X, Y, model, criterion, batch_size):
    model.eval();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        if X.shape[0]!=args.batch_size:
            break
        output = model(X);
        loss = criterion(output, Y);
    
        total_loss += loss.data.item();
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples
    
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default="Num_case.txt",
                    help='location of the data file')
parser.add_argument('--model', type=str, default='MPGE',
                    help='')
parser.add_argument('--k_size', type=list, default=[5,3],
                    help='CNN kernel sizes')
parser.add_argument('--channel_size1', type=int, default=5,
                    help='channel size of the CNN layers')
parser.add_argument('--channel_size2', type=int, default=10,
                    help='channel size of the CNN layers')
parser.add_argument('--hid1', type=int, default=64,
                    help='hidden size of the MILF layers')
parser.add_argument('--hid2', type=int, default=32,
                    help='hidden size of the MILF layers')
parser.add_argument('--hid3', type=int, default=1,
                    help='hidden size of the MILF layers')
parser.add_argument('--n_e', type=int, default=6,
                    help='number of graph nodes (process variables)')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=54230,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--window', type=int, default=10,
                    help='window size')
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=0)
args = parser.parse_args()

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args.data, 0.95, 0.05, args.cuda, args.horizon, args.window, args.normalize);

model = eval(args.model).Model(args);

if args.cuda:
    model.cuda()
    
nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);

if args.cuda:
    criterion = criterion.cuda()
    
    
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,
)


try:
    print('begin training');
    time_start=time.time()
    for epoch in range(1, args.epochs+1):
        lamda = 0.15
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size,lamda)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss))

    time_end=time.time()
    print('totally cost',time_end-time_start)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


from matplotlib import pyplot as plt
a = torch.abs(model.A0).cpu().data.numpy()
a_sum = np.sum(a,axis=1)
for i in range(a.shape[0]):
    a[i,:] = a[i,:]/a_sum[i]

delta = 5

b = a.copy()
b= torch.from_numpy(b)
b_s1 = F.softmax(delta * b)

b_s2 = F.softmax(delta*b_s1)
a_final = torch.mm(torch.mm(b_s2,b_s1),b).data.numpy()


import pandas as pd
import seaborn as sns
plt.figure(figsize=(5.2,10))
plt.subplot(2,1,1)
p=pd.DataFrame(a_final)
p.columns=['$t_{1}$','$t_{2}$','$x_{1}$','$x_{2}$','$x_{3}$','$x_{4}$']
p.index=['$t_{1}$','$t_{2}$','$x_{1}$','$x_{2}$','$x_{3}$','$x_{4}$']
sns.heatmap(p,cmap='Purples',cbar=False,vmin=0,vmax=0.5, linewidths=0.05, linecolor='white',annot=True)
plt.xlabel('Cause')
plt.ylabel('Effect')


l, v = np.linalg.eig(a_final.T)
score = (v[:,0]/np.sum(v[:,0])).real
score=np.around(score/np.sum(score), decimals=3)
plt.subplot(2,1,2)
t = ['$t_{1}$','$t_{2}$','$x_{1}$','$x_{2}$','$x_{3}$','$x_{4}$']
for x, y in enumerate(score):
    plt.text(x-0.32, y+0.01, "%s" %y)
plt.xticks(range(len(t)), t)
plt.ylabel('Root cause score')
plt.ylim(0,0.45)
plt.bar(range(len(t)), score, color='#6950a1',alpha=0.85)