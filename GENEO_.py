import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GENEO(nn.Module):
    def __init__(self, patterns):
        super().__init__()  
        self.patterns = patterns
        self.vectors = nn.ParameterList([nn.Parameter(torch.randn(2), requires_grad=True) for i in range(len(patterns))])

    def forward(self, x):
        F_k = GENEO_1(self.patterns,x)
        T_k = GENEO_3(self.vectors,F_k)
        # out = GENEO_4(T_k)
        return T_k
    
    
def GENEO_1(patterns,image):
    final_out=[]
    for pattern in patterns:
        (pattern_dimension_x, pattern_dimension_y) = pattern.size()
        image_dimensions = image.size()
        out_p =  F.pad(image, (pattern_dimension_y,pattern_dimension_y,pattern_dimension_x,pattern_dimension_x) , "constant", 0)
        out_image = torch.zeros(image_dimensions)
        image_window = torch.zeros(pattern.size())
        for i in range(image_dimensions[0]):
            for j in range(image_dimensions[1]):
                for s in range(pattern_dimension_y):
                    for t in range(pattern_dimension_x):
                        image_window[s][t] = out_p[0][0][i+s][j+t]
                sum = torch.sum(pattern-image_window)/(pattern_dimension_x*pattern_dimension_y)
                out_image[i][j] = 1 - sum
        final_out.append(out_image.unsqueeze(0))
    out = torch.cat(final_out, dim = 0)
    return out


def GENEO_2(functions):
    return torch.mean(functions, dim = 0)
    
def GENEO_3(vectors,functions):
    image_dimensions = functions[0].size()    
    out = torch.zeros(image_dimensions)
    for i in range(image_dimensions[0]):
        for j in range(image_dimensions[1]):
            sum = 0
            for k in range(len(vectors)):
                sum += functions[k][0][0][i-torch.floor(vectors[k][0]).to(dtype = torch.long)][j-torch.floor(vectors[k][1]).to(dtype = torch.long)]
            out[i][j] = sum/k
    return out

def GENEO_4(image):
    norm = torch.norm(image, p="inf")
    return 1/norm

patterns = torch.load("patterns.pt")
