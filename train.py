import os 
import shutil
import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import timeit
from math import floor, ceil
#
import config
from FGENEO import GENEO, GENEO_MLP_Small
from FGENEO_batch import GENEO_MLP_Small_b, GENEO_thorus
from FGENEO_b_r import GENEO_MLP_Small_b_r
from FCNN import CNN
from funtions import plot_vectors, save_patterns, get_points,fidelity, plots

# directory_model1 = "/home/jcolombini/GENEO/RUNSTEN/LR_0.003-SzPtt_9-NImg_500-PPImg_2"
# directory_model2 = "/home/jcolombini/GENEO/RUNSTEN/BENCHMARK_CNN"
# patterns = torch.load(os.path.join(directory_model1,"patterns.pt"), weights_only =True)
# model1 = GENEO_MLP_Small_b(patterns=patterns, num_classes=10)
# model1.load_state_dict(torch.load(os.path.join(directory_model1,"model.pt"),weights_only=True))

# model2 = CNN(num_classes=10)
# model2.load_state_dict(torch.load(os.path.join(directory_model2,"model.pt"),weights_only=True))

# transforms = torchvision.transforms.Compose([
# torchvision.transforms.PILToTensor()]
# )
# ## Import dei dati
# dataset = torchvision.datasets.MNIST(root = "Dataset", transform = transforms)
# dataset= [ data[0]/255 for data in dataset ]
# dataoader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
# fidelity(model1, model2, dataoader)
# exit()
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
NUM_CLASSES = config.NUM_CLASSES
transforms = torchvision.transforms.Compose([
  torchvision.transforms.PILToTensor()]
)
## Import dei dati
dataset = torchvision.datasets.MNIST(root = "Dataset", transform = transforms)
# tmpMODEL = CNN(NUM_CLASSES)
# tmpMODEL.load_state_dict(torch.load("/home/jcolombini/GENEO/RUNSTEN/BENCHMARK_CNN/model.pt",weights_only=True))
# dataset= [ ( data[0]/255,F.one_hot(torch.argmax(tmpMODEL(data[0]/255)), num_classes=10).type(torch.float32)) for data in dataset ]

# #F.avg_pool2d(data[0]/255,(2,2),2)
if NUM_CLASSES == 10:
      dataset= [(data[0]/255, F.one_hot(torch.tensor(data[1]), num_classes=10).type(torch.float32) ) for data in dataset ]
else:
      datasevens = [(data[0]/255, torch.tensor([0,1], dtype = torch.float32)) for data in dataset if (data[1]==7 )]
      datanonsevens = [(data[0]/255, torch.tensor([1,0], dtype = torch.float32))for data in dataset if (data[1]!=7)]
      dataset = datasevens + datanonsevens[:len(datasevens)]
## separazione in train e test
trainset, testset = torch.utils.data.random_split(dataset,[1-TEST_PERCENT,TEST_PERCENT])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

## Spostamento della directory 
if NUM_CLASSES == 2:
    RUNS_FOLDER = "RUNS"
else:
    RUNS_FOLDER = "RUNSTEN"

if RUN:
      os.chdir(RUNS_FOLDER)
      if not os.path.isdir(RUN_NAME):
            os.mkdir(RUN_NAME)
      os.chdir(RUN_NAME)
      shutil.copy("../../config.py","config.py")

Report = "reportfile.txt" 
## Definizione di:
 # Patterns
###
if RUN_NAME != "BENCHMARK_CNN":
      save_patterns(dataset,SIZE_PATTERNS,POINTS_PER_IMAGE, NUM_IMAGES, PATTERNS_FILE)
      patterns = torch.load(PATTERNS_FILE, weights_only=True)
###
 # Modello
if RUN_NAME == "BENCHMARK_CNN":
      Model = CNN(NUM_CLASSES)
else:
      Model = GENEO_thorus(patterns, num_classes=NUM_CLASSES)
with open(Report, "a") as f:
     total_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
     f.write("NUmber_of_parameters= "+str(total_params)+" \n ")

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
start = timeit.timeit()
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
       start_one_forward = timeit.timeit()
       output = Model(inputs)
       end_one_forward = timeit.timeit()
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
            print('batch {} loss: {}, accuracy {} '.format(i + 1, last_loss, accuracy/total))
            with open(Report, "a") as f:
                  f.write('epoch {}, accuracy {} \n'.format(epoch, accuracy/total))
            test_losses.append(last_loss)
            running_loss = 0.
            ax.plot(np.arange(len(test_losses))*(int(len(train_losses)//len(test_losses))), test_losses, color = "r")
            if RUN:
             plt.savefig("Train.png") 
if RUN:
      torch.save(Model.state_dict(), "model.pt")
with open(Report, "a") as f:
      f.write('Forward time {} \n'.format(end_one_forward  -start_one_forward))
      f.write('Total time :{}'.format(timeit.timeit()-start))
