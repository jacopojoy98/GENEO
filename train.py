import torch
import torchvision
# import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
import config
from FGENEO import GENEO
from funtions import plot_vectors
import os 

## Definizione iperparametri
BATCH_SIZE = config.BATCH_SIZE
TEST_PERCENT = config.TEST_PERCENT
PATTERNS_FILE = config.PATTERNS_FILE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS  = config.EPOCHS
RUN_NAME = "X"

transforms = torchvision.transforms.Compose([
  torchvision.transforms.PILToTensor()]
)

## Import dei dati
dataset = torchvision.datasets.MNIST(root = "Dataset", transform = transforms)
dataset = [(data[0]/255, data[1]) for data in dataset if data[1]==7]
## separazione in train e test
trainset, testset = torch.utils.data.random_split(dataset,[1-TEST_PERCENT,TEST_PERCENT])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
## Definizione di:
 # Modello
patterns = torch.load(PATTERNS_FILE, weights_only=True)/255
Model = GENEO(patterns)
 # Optimizer
Opt = torch.optim.Adam(params = Model.parameters() , lr = LEARNING_RATE )
 # Loss function
Loss_fn = torch.nn.MSELoss()
# if not os.path.isdir(RUN_NAME):
#       os.mkdir(RUN_NAME)
# os.chdir(RUN_NAME)
# os.makedirs("Vector_positions", exist_ok=True)
## Definizione dell'iter di training
for epoch in range(EPOCHS):
    running_loss = 0.
    last_loss = 0
    for i, data in enumerate(train_loader):
 # Data inport
       Model.train()
       inputs, labels = data
 # opt.zero_grad()
       Opt.zero_grad()
 # Forward pass
       output = Model(inputs)
       print(output)
       vectors = Model.vectors
       if i %25 ==0:
        plot_vectors(vectors, patterns)
 # Calcolo la loss
       if labels == 7:
        Loss = Loss_fn(output,torch.ones(output.shape, requires_grad = True))
       else:
        exit()
        Loss = Loss_fn(output,torch.zeros(output.shape, requires_grad = True))
 # Loss.backward()
       Loss.backward()
 # Optimizer.step()
       running_loss += Loss.item()
       print('  batch {} loss: {}'.format(i + 1, Loss.item()))
       Opt.step()
       
    for i, data in enumerate(train_loader):
 # Data inport
       Model.eval()
       inputs, labels = data

 # Forward pass
       output = Model(inputs)
 # Calcolo la loss
       if labels == 7:
        Loss = Loss_fn(output,torch.ones(output.shape, requires_grad = True))
       else:
        exit()
 # Optimizer.step()
       print('  batch {} loss: {}'.format(i + 1, Loss.item()))
## Definizione dell'iter di test
#
#   