import torch.nn as nn
import torch.nn.functional as F
import torch
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        print(self.conv1(x).shape)
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        print(self.conv2(x).shape)
        print(F.max_pool2d(self.conv2(x), 2).shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        print(self.conv3(x).shape)
        print(F.max_pool2d(self.conv3(x),2).shape)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.L1 = nn.Linear(28*28,num_classes)
        self.Act = nn.Sigmoid()

    def forward(self, x):
        return self.Act(self.L1(x.view(8,28*28)))
    
class MLP2(nn.Module):
    def __init__(self, num_classes):
        super(MLP2, self).__init__()
        self.L1 = nn.Linear(28*28,4)
        self.L2 = nn.Linear(4,10)
        self.Act = nn.Sigmoid()

    def forward(self, x):
        k1 = self.Act(self.L1(x.view(8,28*28))) 
        return self.Act(self.L2(k1))
    
class MLPs2(nn.Module):
    def __init__(self, num_classes):
        super(MLPs2, self).__init__()
        self.L1 = nn.Linear(28*28,5)
        self.L2 = nn.Linear(5,10)
        self.Act = nn.Sigmoid()

    def forward(self, x):
        k1 = self.Act(self.L1(x.view(8,28*28))) 
        return self.Act(self.L2(k1))

class MLPs3(nn.Module):
    def __init__(self, num_classes):
        super(MLPs3, self).__init__()
        self.L1 = nn.Linear(28*28,7)
        self.L2 = nn.Linear(7,10)
        self.Act = nn.Sigmoid()

    def forward(self, x):
        k1 = self.Act(self.L1(x.view(8,28*28))) 
        return self.Act(self.L2(k1))
    

class MLP3(nn.Module):
    def __init__(self, num_classes):
        super(MLP3, self).__init__()
        self.L1 = nn.Linear(28*28,10)
        self.L2 = nn.Linear(10,10)
        self.Act = nn.Sigmoid()

    def forward(self, x):
        k1 = self.Act(self.L1(x.view(8,28*28))) 
        return self.Act(self.L2(k1))
    
class MLP4(nn.Module):
    def __init__(self, num_classes):
        super(MLP4, self).__init__()
        self.L1 = nn.Linear(28*28,20)
        self.L2 = nn.Linear(20,10)
        self.Act = nn.Sigmoid()

    def forward(self, x):
        k1 = self.Act(self.L1(x.view(8,28*28))) 
        return self.Act(self.L2(k1))

class MLP5(nn.Module):
    def __init__(self, num_classes):
        super(MLP5, self).__init__()
        self.L1 = nn.Linear(28*28,40)
        self.L2 = nn.Linear(40,10)
        self.Act = nn.Sigmoid()

    def forward(self, x):
        k1 = self.Act(self.L1(x.view(8,28*28))) 
        return self.Act(self.L2(k1))
    
if __name__ == "__main__":
    a = torch.randn((1,1,28,28))
    md = CNN(10)
    out = md(a)
