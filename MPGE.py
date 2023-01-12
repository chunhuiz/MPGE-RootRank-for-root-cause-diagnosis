import torch
import torch.nn as nn
import torch.nn.functional as F
     
class Model(nn.Module):
    
    def __init__(self, args):
        super(Model,self).__init__()
        
        A0=torch.ones(args.n_e,args.n_e)+torch.eye(args.n_e)
        A0=A0/torch.sum(A0,1)
       
        self.A0=nn.Parameter(A0)
        self.n_e=args.n_e
        self.BATCH_SIZE=args.batch_size
        self.conv1=nn.Conv2d(1, args.channel_size1, kernel_size = (1,args.k_size[0]),stride=1)
        self.conv2=nn.Conv2d(args.channel_size1, args.channel_size2, kernel_size = (1,args.k_size[1]),stride=1)
        t=args.window
        d1= (t -args.k_size[0]+ 1)
        d = (d1 -args.k_size[1]+ 1)*args.channel_size2
        w1=nn.init.kaiming_uniform_(torch.Tensor(d ,args.hid1))
        w2=nn.init.kaiming_uniform_(torch.Tensor(args.hid1, args.hid2))
        w3=nn.init.kaiming_uniform_(torch.Tensor(args.hid2, args.hid3))
        self.w1=nn.Parameter(w1)
        self.w2=nn.Parameter(w2)
        self.w3=nn.Parameter(w3)
        self.d=d

    def forward(self, x ,delta = 5): 
        x=x.permute(0,2,1)
        x=x.unsqueeze(1)
        A=(torch.abs(self.A0))/torch.sum(self.A0,1).repeat(self.BATCH_SIZE,1,1)
       
        softmax_A1=F.softmax(delta*A)
        softmax_A2=F.softmax(delta*softmax_A1)
        
        w1=self.w1.repeat(self.BATCH_SIZE,1,1)
        w2=self.w2.repeat(self.BATCH_SIZE,1,1)
        w3=self.w3.repeat(self.BATCH_SIZE,1,1)

        a1=F.relu(self.conv1(x))
        a2=self.conv2(a1).permute(0,2,1,3).reshape(self.BATCH_SIZE,self.n_e,-1)
        x_conv=F.relu(a2)
      
        x1=F.relu(torch.bmm(A,x_conv).bmm(w1))
        x2=F.relu(torch.bmm(softmax_A1,x1).bmm(w2))
        x3=F.relu(torch.bmm(softmax_A2,x2).bmm(w3))

        return x3.squeeze()    
