import os 
import shutil
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from random import randint
#
import config
from FGENEO import GENEO, GENEO_MLP_Small
from FGENEO_batch import GENEO_MLP_Small_b
from FGENEO_b_r import GENEO_MLP_Small_b_r
from FCNN import CNN
from funtions import plot_vectors, save_patterns, get_points

RUN = 1

## Definizione iperparametri
BATCH_SIZE = config.BATCH_SIZE
TEST_PERCENT = config.TEST_PERCENT
PATTERNS_FILE = config.PATTERNS_FILE
LEARNING_RATE = config.LEARNING_RATE
EPOCHS  = config.EPOCHS
RUN_NAME = config.RUN_NAME
SIZE_PATTERNS = config.SIZE_PATTERNS
POINTS_PER_IMAGE = config.POINTS_PER_IMAGE
NUM_IMAGES = config.NUM_IMAGES

transforms = torchvision.transforms.Compose([
  torchvision.transforms.PILToTensor()]
)
## Import dei dati
dataset = torchvision.datasets.MNIST(root = "Dataset", transform = transforms)
dataset= [(data[0]/255, F.one_hot(torch.tensor(data[1]), num_classes=10).type(torch.float32) ) for data in dataset ]
# datasevens = [(data[0]/255, torch.tensor([0,1], dtype = torch.float32)) for data in dataset if (data[1]==7 )]
# datanonsevens = [(data[0]/255, torch.tensor([1,0], dtype = torch.float32))for data in dataset if (data[1]!=7)]
# dataset = datasevens + datanonsevens[:len(datasevens)]
## separazione in train e test
trainset, testset = torch.utils.data.random_split(dataset,[1-TEST_PERCENT,TEST_PERCENT])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

## Spostamento della directory 
if RUN:
      os.chdir("RUNSTEN")
      if not os.path.isdir(RUN_NAME):
            os.mkdir(RUN_NAME)
      os.chdir(RUN_NAME)
      shutil.copy("../../config.py","config.py")

## Definizione di:
 # Patterns
###
save_patterns(dataset,SIZE_PATTERNS,POINTS_PER_IMAGE, NUM_IMAGES, PATTERNS_FILE)
patterns = torch.load(PATTERNS_FILE, weights_only=True)
###
 # Modello
Model = GENEO_MLP_Small_b(patterns, num_classes=10)
# Model = CNN()
 # Optimizer
Opt = torch.optim.Adam(params = Model.parameters() , lr = LEARNING_RATE )
 # Loss function
Loss_fn = torch.nn.BCELoss()
 # Array di plot
train_losses = []
test_losses = []
 # Figue di plot
fig, ax = plt.subplots(1)
ax.set_yscale('log')
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
       Loss = Loss_fn(output, labels)
 # Loss.backward()
       Loss.backward()
 # Optimizer.step()
       Opt.step()
       running_loss += Loss.item()

       if i % 100 == 99:
            last_loss = running_loss / 1000 # loss per batch
            train_losses.append(last_loss)
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
            ax.plot(np.arange(len(train_losses)), train_losses, color = "b")
            if RUN:
             plt.savefig("Train.png")

## Definizione dell'iter di test
    accuracy = 0 
    total = 0
    for i, data in enumerate(test_loader):
       Model.eval()
       inputs, labels = data
 # Forward pass
       output = Model(inputs)
 # Calcolo la loss
       total += 8
       for label, guess in zip(labels, output):
            if torch.argmax(guess)==torch.argmax(label):
                  accuracy += 1
       Loss = Loss_fn(output,labels)

       running_loss+=Loss.item()
       if i == len(test_loader)-1:
            last_loss = running_loss / len(test_loader) # loss per batch

            print('  batch {} loss: {}, accuracy {} '.format(i + 1, last_loss, accuracy/total))
            test_losses.append(last_loss)
            running_loss = 0.
            ax.plot(np.arange(len(test_losses))*(int(len(trainset)//len(testset))), test_losses, color = "r")
            if RUN:
             plt.savefig("Train.png") 
if RUN:
 torch.save(Model.state_dict(), "model.pt")