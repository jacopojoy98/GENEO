import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functions_0 import Circle_indexes, Circle, F_Circle

class GENEO(nn.Module):
    def __init__(self, r_1,r_2,h_3,r_4):
        super().__init__()  

    def forward(self, x):

        return None
    
    
def GENEO_1(pattern,image): 
    (pattern_dimension_x, pattern_dimension_y) = pattern.size()
    image_dimensions = image.size()
    out =  F.pad(image, (pattern_dimension_y,pattern_dimension_y,pattern_dimension_x,pattern_dimension_x) , "constant", 0)
    out_image = torch.zeros(image_dimensions)
    image_window = torch.zeros(pattern.size())
    for i in range(image_dimensions[0]):
        for j in range(image_dimensions[1]):
            for s in range(pattern_dimension_y):
                for t in range(pattern_dimension_x):
                    image_window[s][t] = out[i+s][j+t]
            # print(pattern_dimension_y)
            # print(pattern_dimension_x)
            # print(pattern.shape)
            # print(image_window.shape)
            sum = torch.sum(pattern-image_window)/(pattern_dimension_x*pattern_dimension_y)
            out_image[i][j] = 1 - sum
    return out_image


def GENEO_2(functions):
    return torch.mean(functions, dim = 0)
    

def GENEO_3(vectors,functions):
    image_dimensions = functions[0].size()    
    out = torch.zeros(image_dimensions)
    for i in image_dimensions[0]:
        for j in image_dimensions[1]:
            sum = 0
            for k in range(len(vectors)):
                sum += functions[k][i-vectors[k][0]][j-vectors[k][1]]
            out[i][j] = sum/k
    return out

def GENEO_4(image):
    norm = torch.norm(image, p="inf")
    return 1/norm

image = torch.zeros((100,100))

pattern = torch.randint(0,9,(12,2))
pattern_image=torch.zeros((10,10))
for ii in range(10):
    for jj in range(10):
        pattern_image[ii][jj]=1
centers = torch.randint(0,90,(12,2))
for [c_x,c_y] in centers:
    for [p_x,p_y] in pattern:
        image[c_x+p_x][c_y+p_y] = 1

out = GENEO_1(pattern_image, image)
plt.imshow(out,cmap="gray")
plt.show()
