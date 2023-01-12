import torch
import numpy as np;
from torch.autograd import Variable

class Data_utility(object):

    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 0):
        self.cuda = cuda;
        self.P = window;
        self.h = horizon
        fin = open(file_name);
        self.rawdat = np.loadtxt(fin);
        self.dat = np.zeros(self.rawdat.shape);
        self.n, self.m = self.dat.shape;
        self.normalize = 0
        self.scale = np.ones(self.m);
        self._normalized(normalize);
        self._split(int(train * self.n), int((train+valid) * self.n), self.n);
        
        self.scale = torch.from_numpy(self.scale).float();
            
        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);
    
    def _normalized(self, normalize):
    # If the data has been processed by Z-score normalization, then set normalize = 0.
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            for i in range(self.m):
                self.dat[:,i] = self.rawdat[:,i] / (self.rawdat[:,i].std());
                self.dat[:,i] -= self.dat[:,i].mean()
            
        
    def _split(self, train, valid, test = 0):
        
        train_set = range(self.P+self.h-1, train);
        valid_set = range(train, valid);
        self.train = self._batchify(train_set, self.h);
        self.valid = self._batchify(valid_set, self.h);
        
        
    def _batchify(self, idx_set, horizon):
        
        n = len(idx_set);
        X = torch.zeros((n,self.P,self.m));
        Y = torch.zeros((n,self.m));
        
        for i in range(n):
            end = idx_set[i] - self.h + 1;
            start = end - self.P;
            X[i,:,:] = torch.from_numpy(self.dat[start:end, :]);
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :]);

        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]; Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();  
            yield Variable(X), Variable(Y);
            start_idx += batch_size