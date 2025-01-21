import torchvision
import matplotlib.pyplot as plt
import torch
import os

dataset = torchvision.datasets.MNIST(root = "Dataset",transform=torchvision.transforms.PILToTensor())
sevens = []
for data in dataset:
    if data[1]==7:
        sevens.append(data)
os.mkdir("IMG")
for i in range(12):
    plt.imshow(sevens[i][0].squeeze(), cmap="gray")
    plt.savefig("IMG/"+str(i)+".png")
    plt.close()

