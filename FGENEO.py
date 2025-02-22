import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from funtions import clip, i_clip, plots, plotsim
######   ######

class GENEO_MLP_Small(nn.Module):
    def __init__(self, patterns, num_classes):
        super().__init__()
        self.patterns = patterns
        self.GENEO2 = nn.Linear(len(patterns), 1)
        self.ACT = nn.Sigmoid()
        self.Finael = nn.Linear(28*28, num_classes)

    def forward(self, x):
        F_k = GENEO_1_optimized(self.patterns, x)
        CWM = Channel_wise_max(F_k)
        T_k = self.GENEO2(CWM.permute(2, 3, 1, 0))
        T_k = self.ACT(T_k)
        out = self.Finael(T_k.flatten())
        return self.ACT(out)
    

class GENEO(nn.Module):
    def __init__(self, patterns,treshold):
        super().__init__()  
        self.treshold = treshold
        self.patterns = patterns
        
        self.vectors = nn.ParameterList([nn.Parameter(torch.rand(2)*27, requires_grad=True) for i in range(len(patterns)-1)])

    def forward(self, x):
        F_k = GENEO_1_optimized(self.patterns, self.treshold, x)
        T_k = GENEO_3(self.vectors,F_k)
        out = torch.dot(torch.softmax((T_k*10).flatten(), dim = -1),  T_k.flatten())
        return out

####
#
# AGGIUNGERE 
# SOFTMAX
#
####

def Channel_wise_max(tensor):
    # Get the shape of the tensor
    patterns_size, b, rows, cols = tensor.shape
    # Flatten the last two dimensions to find the argmax for each matrix
    flat_tensor = tensor.view(patterns_size, -1)
    max_indices = flat_tensor.argmax(dim=1)

    # Create a mask of zeros
    mask = torch.zeros_like(flat_tensor, dtype=torch.float)
    # Set the maximum value positions to 1
    mask[torch.arange(patterns_size), max_indices] = 1

    # Reshape the mask back to the original tensor shape
    mask = mask.view(patterns_size, b, rows, cols)

    # Retain only the maximum values
    result = tensor * mask

    return result


def GENEO_1_optimized(patterns, image):

    # patterns = (K, 1, H_p, W_p)
    # image    = (1, 1, 28, 28)
    # #
    # plots(image, "image")
    #
    K, _, H_p, W_p = patterns.size()
    _, _, H_img, W_img = image.size()

    # Pad the image to handle borders
    padded_image = F.pad(image, (W_p//2, W_p//2, H_p//2, H_p//2), mode="constant", value=0)

    # Extract sliding windows (unfolded image)
    unfolded_image = padded_image.unfold(2, H_p, 1).unfold(3, W_p, 1)  # (1, 1, H_out, W_out, H_p, W_p)
    H_out, W_out = unfolded_image.size(2), unfolded_image.size(3)
    
    # Reshape to (1, 1, H_out * W_out, H_p * W_p)
    unfolded_image = unfolded_image.contiguous().view(1, 1, H_out * W_out, H_p * W_p)
    # print(unfolded_image.shape)
    
    # Reshape patterns for broadcasting: (K, 1, H_p*W_p)
    patterns_reshaped = patterns.view(K, 1, H_p * W_p)
    # Calculate absolute difference and normalize
    difference = torch.abs(patterns_reshaped - unfolded_image)
    # print(difference.shape)
    normalized_diff = torch.mean(difference, dim=-1)  # (K, 1, H_out * W_out)
    # Compute output
    out_image = 1 - normalized_diff
    out_image = out_image.view(K, 1, H_out, W_out)



    return out_image

def GENEO_1_toro(patterns, image):

    # patterns = (K, 1, H_p, W_p)
    # image    = (1, 1, 28, 28)
    # #
    # plots(image, "image")
    #
    K, _, H_p, W_p = patterns.size()
    _, _, H_img, W_img = image.size()

    # Pad the image to handle borders
    padded_image = F.pad(image, (W_p//2, W_p//2, H_p//2, H_p//2), mode="circular")

    # Extract sliding windows (unfolded image)
    unfolded_image = padded_image.unfold(2, H_p, 1).unfold(3, W_p, 1)  # (1, 1, H_out, W_out, H_p, W_p)
    H_out, W_out = unfolded_image.size(2), unfolded_image.size(3)
    
    # Reshape to (1, 1, H_out * W_out, H_p * W_p)
    unfolded_image = unfolded_image.contiguous().view(1, 1, H_out * W_out, H_p * W_p)
    # print(unfolded_image.shape)
    
    # Reshape patterns for broadcasting: (K, 1, H_p*W_p)
    patterns_reshaped = patterns.view(K, 1, H_p * W_p)
    # Calculate absolute difference and normalize
    difference = torch.abs(patterns_reshaped - unfolded_image)
    # print(difference.shape)
    normalized_diff = torch.mean(difference, dim=-1)  # (K, 1, H_out * W_out)
    # Compute output
    out_image = 1 - normalized_diff
    out_image = out_image.view(K, 1, H_out, W_out)
    out_flat = torch.nn.functional.max_pool2d(out_image,(H_out, W_out))
    print(out_flat.shape)
    exit()
    return out_flat

def GENEO_3(vectors,functions):
    image_dimensions = functions.size()    
    i = torch.arange(image_dimensions[2])
    j = torch.arange(image_dimensions[3])
    out = torch.zeros((1,1,28,28))
    for k in range(len(vectors)):
        v_x = torch.floor(vectors[k][0]).to(dtype = torch.long)
        v_y = torch.floor(vectors[k][1]).to(dtype = torch.long)
        v_x_1 = v_x+1
        v_y_1 = v_y+1
        p_x =  vectors[k][0] - v_x
        p_y =  vectors[k][1] - v_y 
        ##
        # print(f"v_x:{v_x} - v_y:{v_y} - p_x:{p_x} - p_y:{p_y}")
        ##
        for s_x,s_y,p in [(v_x,v_y,p_x*p_y),\
                        (v_x,v_y_1,p_x*(1-p_y)),\
                        (v_x_1,v_y,(1-p_x)*p_y),\
                        (v_x_1,v_y_1,(1-p_x)*(1-p_y))]:
            ##
            # print(f"functions = {functions[k][0][clip(i-s_x)][clip(j-s_y)]}")
            # input()
            ##
            out[0][0] += functions[k][0][clip(i-s_x)][clip(j-s_y)]*(p)
            ##
            # IMPLEMENTARE SOFTMAX
            ##

    out[0][0] += functions[len(vectors)][0][i][j]
        ##
        # plots(out.detach(), "out_2//out_2"+str(k))
        ##
    out = out / len(functions)
    ##
    # exit()
    # plots(out[0][0].detach(), "out") 
    # input()
    ##
    return out

#####
## BACKUP FUNCTIONS
#####

# def GENEO_1(patterns, threshold, image):
    
#     # patterns = (K, 1, H_p, W_p)
#     # image    = (1, 1, 28, 28 )

#     final_out=[]
#     for p, pattern in enumerate(patterns):
#         (_, pattern_dimension_x, pattern_dimension_y) = pattern.size()
#         image_dimensions = image.size()
#         out_p =  F.pad(image, (pattern_dimension_y,pattern_dimension_y,pattern_dimension_x,pattern_dimension_x) , "constant", 0)
#         out_image = torch.zeros(image_dimensions)
#         for i in range(image_dimensions[2]):
#             for j in range(image_dimensions[3]):

#                 image_window = out_p[0][0][i:i+pattern_dimension_x, j:j+pattern_dimension_y]
        
#                 difference = torch.abs(pattern-image_window)
        
#                 sum = torch.sum(difference)/(pattern_dimension_x*pattern_dimension_y)
        
#                 out_image[0][0][i][j] = 1 - sum
        
#         F.threshold(out_image, threshold, 0, inplace=True)
#         final_out.append(out_image.unsqueeze(0))
#     out = torch.cat(final_out, dim = 0)
#     return out

# def GENEO_3(vectors,functions):
#     image_dimensions = functions[0].size()    
#     out = torch.zeros(image_dimensions)
#     v_x = torch.floor(vectors[:, 0]).long()
#     v_y = torch.floor(vectors[:, 1]).long()
#     v_x_1 = v_x + 1
#     v_y_1 = v_y + 1
#     p_x = v_x - vectors[:, 0]
#     p_y = v_y - vectors[:, 1]

#     weights = torch.stack([
#         p_x * p_y,
#         p_x * (1 - p_y),
#         (1 - p_x) * p_y,
#         (1 - p_x) * (1 - p_y)
#     ], dim=1)

#     for i in range(image_dimensions[0]):
#         for j in range(image_dimensions[1]):
#             sum = 0
#             for k in range(len(vectors)):
#                 v_x = torch.floor(vectors[k][0]).to(dtype = torch.long)
#                 v_y = torch.floor(vectors[k][1]).to(dtype = torch.long)
#                 v_x_1 = v_x+1
#                 v_y_1 = v_y+1
#                 p_x = v_x - vectors[k][0]
#                 p_y = v_y - vectors[k][1]
#                 for s_x,s_y,p in [(v_x,v_y,p_x*p_y),\
#                                 (v_x,v_y_1,p_x*(1-p_y)),\
#                                 (v_x_1,v_y,(1-p_x)*p_y),\
#                                 (v_x_1,v_y_1,(1-p_x)*(1-p_y))]:
#                     sum += functions[k][0][0][i-s_x][j-s_y]*(p)
#             out[i][j] = sum/k
#     return out

# def GENEO_4(image):
#     norm = torch.norm(image, p="inf")
#     return 1/norm

