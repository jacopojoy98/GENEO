import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os
import numpy as np

dataset = torchvision.datasets.MNIST(root = "Dataset",transform=torchvision.transforms.PILToTensor())
sevens = []
for data in dataset:
    if data[1]==7:
        sevens.append(data)

filename = "points.txt"
points = []
with open(filename) as f:
    for line in f.readlines():
        line = line.replace("(",""). replace(")","").split(",")
        line =[int(s) for s in line]
        p = [(line[i],line[i+1]) for i in range(len(line)//2)]
        points.append((line[0],p))


for i in range(45):    
    for point in points:
        if point[0]==i:
            fig,ax = plt.subplots(i)
            ax.imshow(sevens[i][0].squeeze(), cmap="gray")
            for p in point[1]:
                rect = patches.Rectangle(p,1,1,linewidth=0,edgecolor=None,color="red")
                ax.add_patch(rect)
            plt.savefig("IMG/"+str(i)+".png")
            plt.close()
           
    
