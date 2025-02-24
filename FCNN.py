import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
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