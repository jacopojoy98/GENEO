import torch
import torchvision
# import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
import config
from GENEO_ import GENEO

## Definizione iperparametri
BATCH_SIZE = config.BATCH_SIZE
TEST_PERCENT = config.TEST_PERCENT
PATTERNS_FILE = config.PATTERNS_FILE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS  = config.EPOCHS

## Import dei dati
dataset = torchvision.datasets.MNIST(root = "Dataset", transform=torchvision.transforms.PILToTensor())

## separazione in train e test
trainset, testset = torch.utils.data.random_split(dataset,[1-TEST_PERCENT,TEST_PERCENT])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
## Definizione di:
 # Modello
patterns = torch.load(PATTERNS_FILE)
Model = GENEO(patterns)
 # Optimizer
Opt = torch.optim.Adam(params = Model.parameters() , lr = LEARNING_RATE )
 # Loss function
Loss_fn = torch.nn.MSELoss()

## Definizione dell'iter di training
for epoch in range(EPOCHS):
    running_loss = 0.
    last_loss = 0
    for i, data in enumerate(train_loader):
 # Data inport
        inputs, labels = data
 # opt.zero_grad()
        Opt.zero_grad()
 # Forward pass
        output = Model(inputs)
 # Calcolo la loss
        Loss = Loss_fn(output,torch.zeros(output.shape, requires_grad = True))
 # Loss.backward()
        Loss.backward()
 # Optimizer.step()
        running_loss += Loss.item()
        Opt.step
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

## Definizione dell'iter di test
#
#   