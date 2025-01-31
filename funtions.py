import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
from FGENEO import i_clip
def get_sevens(dataset):
    sevens = []
    for data in dataset:
        if data[1]==7:
            sevens.append(data)
    return sevens

def get_points(filename):
    points = []
    with open(filename) as f:
        for line in f.readlines():
            line = line.replace("(",""). replace(")","").split(",")
            line =[int(s) for s in line]
            p = [(line[i*2+1],line[i*2+2]) for i in range(len(line)//2)]
            points.append((line[0],p))
    return points

def save_images(images, points):
    for point in points:
        image = images[point[0]][0].squeeze()
        fig, ax = plt.subplots(1)
        ax.imshow(image.squeeze(), cmap="gray")
        for p in point[1]:
            rect = patches.Rectangle((p[0]-0.5,p[1]-0.5), 1, 1, linewidth=0, edgecolor=None, color="red")
            ax.add_patch(rect)
        os.makedirs("IMG/data/"+str(point[0]),exist_ok=True)
        plt.savefig("IMG/data/"+str(point[0])+"/"+str(point[0])+".png")
        plt.close()

def save_images_cuts(images, points):
    for point in points:
        image = images[point[0]][0].squeeze()
        for p in point[1]:
            cut = image[p[1]-3:p[1]+4,p[0]-3:p[0]+4]
            fig,ax = plt.subplots(1)
            ax.imshow(cut, cmap="gray")
            rect = patches.Rectangle((2.5,2.5),1,1, linewidth=0, edgecolor=None, color="red")
            ax.add_patch(rect)
            plt.savefig("IMG/data/"+str(point[0])+"/"+str(p[0])+"_"+str(p[1])+".png")
            plt.close()

def save_cuts(images, points):
    s=0
    patterns = []
    for point in points:
        image = images[point[0]][0].squeeze()
        for p in point[1]:
            cut = image[p[1]-3:p[1]+4,p[0]-3:p[0]+4]
            patterns.append(cut.unsqueeze(0))
    out = torch.cat(patterns, dim = 0)
    torch.save(out,"patterns.pt")

def plot_vectors(vectors,patterns):
    for i in range(len(vectors)):
        fig, ax = plt.subplots(2)
        ax[0].imshow(patterns[i], cmap="gray")
        position = np.zeros((28,28))
        position [i_clip(int(np.floor(vectors[i][0].detach().numpy())))][i_clip(int(np.floor(vectors[i][1].detach().numpy())))] = 1
        ax[1].imshow(position,cmap="gray")
        plt.savefig("vector_positions"+str(i)+".png")
        plt.close()

