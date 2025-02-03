import numpy as np
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os

def clip(value, min_val=0, max_val=27):
    return torch.clamp(value, min=min_val, max=max_val)

def i_clip(value):
    if value >27:
        return 27
    if value<0:
        return 0
    else:
        return value

def plots(matrix_,name): 
    if matrix_.shape == torch.Size([1, 1, 28, 28]):
        matrix = matrix_[0][0]
    else:
        matrix = matrix_
    plt.imshow(matrix, cmap="gray")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i,j]:.2f}",  # Formattazione con due decimali
                     ha="center", va="center",size=4.5, color="white" if matrix[i, j] < torch.mean(matrix) else "black")
    plt.savefig("DEBUG_IMG/"+name+".png")
    plt.close()
    # print("saved "+ name, end ="")
    # input()

def plotsim(matrix_,name,l,m): 
    if matrix_.shape == torch.Size([1, 1, 28, 28]):
        matrix = matrix_[0][0]
    else:
        matrix = matrix_
    plt.imshow(matrix, cmap="gray")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i,j]:.2f}",  # Formattazione con due decimali
                     ha="center", va="center",size=4.5, color="white" if matrix[i, j] < torch.mean(matrix) else "black")
    plt.savefig("DEBUG_IMG/IM/"+str(l)+str(m)+".png")
    plt.close()
    # print("saved "+ name, end ="")
    # input()



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

def save_patterns(images, points, size_cuts, file):
    s=0
    half_size_cuts = size_cuts//2
    patterns = []
    for point in points:
        image = images[point[0]][0].squeeze()
        for p in point[1]:
            cut = image[p[1]-half_size_cuts:p[1]+half_size_cuts+1,\
                        p[0]-half_size_cuts:p[0]+half_size_cuts+1]
            patterns.append(cut.unsqueeze(0).unsqueeze(0))
    out = torch.cat(patterns, dim = 0)
    torch.save(out,file)

def plot_vectors(vectors,patterns, EPOCH):
    for i in range(len(vectors)):
        fig, ax = plt.subplots(2)
        ax[0].imshow(patterns[i][0], cmap="gray")
        position = np.zeros((28,28))
        position [i_clip(int(np.floor(vectors[i][0].detach().numpy())))][i_clip(int(np.floor(vectors[i][1].detach().numpy())))] = 1
        ax[1].imshow(position,cmap="gray")
        plt.savefig("Vector_positions/"+str(i)+"_E"+str(EPOCH)+".png")
        plt.close()

