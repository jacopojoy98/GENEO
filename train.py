import torch
import torchvision
import matplotlib.pyplot as plt
# import torch.nn as nn
import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
import config
from FGENEO import GENEO
from funtions import plot_vectors, save_patterns, get_points
import os 
import shutil

## Definizione iperparametri
BATCH_SIZE = config.BATCH_SIZE
TEST_PERCENT = config.TEST_PERCENT
PATTERNS_FILE = config.PATTERNS_FILE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS  = config.EPOCHS
TRESHOLD = config.TRESHOLD
RUN_NAME = config.RUN_NAME
SIZE_PATTERNS = config.SIZE_PATTERNS
POINTS = config.POINTS

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

## Spostamento della directory 
if 1:
      if not os.path.isdir(RUN_NAME):
            os.mkdir(RUN_NAME)
      os.chdir(RUN_NAME)
      shutil.copy("../config.py","config.py")
      os.makedirs("Vector_positions", exist_ok=True)

## Definizione di:
 # Patterns
points = get_points(POINTS)
save_patterns(dataset,points,SIZE_PATTERNS, PATTERNS_FILE)
patterns = torch.load(PATTERNS_FILE, weights_only=True)
 # Modello
Model = GENEO(patterns, TRESHOLD)
 # Optimizer
Opt = torch.optim.Adam(params = Model.parameters() , lr = LEARNING_RATE )
 # Loss function
Loss_fn = torch.nn.MSELoss()
 # Array di plot
train_losses = []
test_losses = []
 # Figue di plot
fig, ax = plt.subplots(1)

## Definizione dell'iter di training
for epoch in range(EPOCHS):
    running_loss = 0.
    last_loss = 0
    Model.train()
    for i, data in enumerate(train_loader):
 # Data inport
       inputs, labels = data
 # opt.zero_grad()
       Opt.zero_grad()
 # Forward pass
       output = Model(inputs)
 # Calcolo la loss
       Loss = Loss_fn(output,torch.ones(output.shape, requires_grad = True))
 # Loss.backward()
       Loss.backward()
 # Optimizer.step()
       Opt.step()
       running_loss += Loss.item()
       if i % 1000 == 999:
            vectors = Model.vectors
            last_loss = running_loss / 1000 # loss per batch
            train_losses.append(last_loss)
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
            ax.plot(train_losses, color = "b")
            plt.savefig("Train.png")
            
## Definizione dell'iter di test
    for i, data in enumerate(test_loader):
       Model.eval()
       inputs, labels = data
 # Forward pass
       output = Model(inputs)
 # Calcolo la loss
       Loss = Loss_fn(output,torch.ones(output.shape, requires_grad = True))
       if i == len(test_loader)-1:
            last_loss = running_loss / len(test_loader) # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            plot_vectors(vectors, patterns, epoch)
            test_losses.append(last_loss)
            running_loss = 0.
            ax.plot(test_losses, color = "b")
            plt.savefig("Train.png") 
            

#
#   