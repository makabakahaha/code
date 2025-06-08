import torch
from torch import nn

from torchvision import datasets,transforms

from code.softmax import batch_size


#数据加载
def get_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    train_iter = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data',train=True,download=False,transform=transform),
        batch_size=batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=False, transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_iter,test_iter


train_iter,test_iter = get_fashion_mnist(batch_size)

class MLP(nn.Module):
    def __init__(self,input_size=784,hidden_size=256,output_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        )

    def forward(self,X):
