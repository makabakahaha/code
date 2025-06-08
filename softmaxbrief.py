from d2l.torch import Accumulator
from torch import nn
import torch
from d2l import torch as d2l
from code import softmax
import matplotlib as plt


batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_wights(m):
    if type(m)== nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_wights)

loss = nn.CrossEntropyLoss(reduction = 'none')

trainer = torch.optim.SGD(net.parameters(),lr=0.1)



def acc_count(y_hat,y):
    if y_hat.dim()>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis =1)
    cmp = y_hat == y
    return float(cmp.sum())

def evaluate_acc(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(acc_count(net(X),y),y.numel())
    return metric[0]/metric[1]

class Animator:
    '''画图绘制数据'''
    def __init__(self,xlabel = None,ylabel = None,legend = None,
                xlim = None,ylim = None,xscale = 'linear',yscale = 'linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)
                 ):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig,self.axes = d2l.plt.subplots(nrows,ncols,figsize=figsize)
        if nrows*ncols==1:
            self.axes = [self.axes,]
        self.config_axes = lambda:d2l.set_axes(self.axes[0],
                xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
        self.X,self.Y,self.fmts=None,None,fmts
        #plt.show(block=False)

    def add(self,x,y):
        if not hasattr(y,"__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x,"__len__"):
            x = [x]*n
        if not self.X:
            self.X = [[]for _ in range(n)]
        if not self.Y:
            self.Y = [[]for _ in range(n)]
        for i,(a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x,y,fmt in zip(self.X,self.Y,self.fmts):
            self.axes[0].plot(x,y,fmt)
        self.config_axes()

        plt.draw()
        plt.pause(0.1)
        print(f"X: {self.X}, Y: {self.Y}")


def train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer):
    net.train()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train_loss', 'train_acc', 'test_acc'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y)
            acc = acc_count(y_hat,y)


            trainer.zero_grad()
            l.mean().backward()
            trainer.step()


        train_metric = float(l.sum())/y.numel(),acc/y.numel()
        test_acc = evaluate_acc(net, test_iter)
        animator.add(epoch + 1, train_metric + (test_acc,))
    train_loss, train_acc = train_metric
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


num_epoch =10
train_ch3(net,train_iter,test_iter,loss,num_epoch,trainer)



