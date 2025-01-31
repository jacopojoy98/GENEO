import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

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
    print("saved "+ name, end ="")
    input()

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


class GENEO(nn.Module):
    def __init__(self, patterns):
        super().__init__()  
        self.patterns = patterns
        self.vectors = nn.ParameterList([nn.Parameter(torch.randn(2)*20, requires_grad=True) for i in range(len(patterns))])

    def forward(self, x):
        # start = time.perf_counter()
        F_k = GENEO_1_s(self.patterns,x)
        # middle = time.perf_counter()
        T_k = GENEO_3(self.vectors,F_k)
        # end_m = time.perf_counter()
        # out = GENEO_4(T_k)
        out = torch.max(T_k.squeeze(-1))
        # end = time.perf_counter()

        # print(f'1 Time taken: {start-middle:.6f} seconds')
        # print(f'2 Time taken: {middle-end_m:.6f} seconds')
        # print(f'3Time taken: {end_m- end:.6f} seconds')
        # input()
        return out
    
def GENEO_1(patterns,image):
    DEBUG = 1
    final_out=[]
    for pattern in patterns:
        (pattern_dimension_x, pattern_dimension_y) = pattern.size()
        image_dimensions = image.size()
        if DEBUG:
         plots(image,"image")
         plots(pattern,"pattern")
        out_p =  F.pad(image, (pattern_dimension_y,pattern_dimension_y,pattern_dimension_x,pattern_dimension_x) , "constant", 0)
        out_image = torch.zeros(image_dimensions)
        # image_window = torch.zeros(pattern.size())
        for i in range(image_dimensions[2]):
            for j in range(image_dimensions[3]):
                image_window = out_p[0][0][i:i+pattern_dimension_x, j:j+pattern_dimension_y]
                difference = torch.abs(pattern-image_window)
                if DEBUG and torch.any(image_window):
                #  print(out_p[0][0][i+s][j+t])
                 plots(image_window, "image_window")
                 plots(difference,"difference")
                sum = torch.sum(difference)/(pattern_dimension_x*pattern_dimension_y)
                out_image[0][0][i][j] = 1 - sum
        threshold = out_image[0][0][0][0]
        F.threshold(out_image, threshold, 0, inplace=True)
        final_out.append(out_image.unsqueeze(0))
        if DEBUG:
         plots(out_image, "out_image")
    out = torch.cat(final_out, dim = 0)
    return out

def GENEO_1_s(patterns,image):
    # DEBUG = 0
    final_out=[]
    for pattern in patterns:
        (pattern_dimension_x, pattern_dimension_y) = pattern.size()
        image_dimensions = image.size()
        out_p =  F.pad(image, (pattern_dimension_y,pattern_dimension_y,pattern_dimension_x,pattern_dimension_x) , "constant", 0)
        out_image = torch.zeros(image_dimensions)
        for i in range(image_dimensions[2]):
            for j in range(image_dimensions[3]):
                out_image[0][0][i][j] = 1- torch.sum(torch.abs(pattern-out_p[0][0][i:i+pattern_dimension_x, j:j+pattern_dimension_y]))/(pattern_dimension_x*pattern_dimension_y)
        threshold = out_image[0][0][0][0]
        F.threshold(out_image, threshold, 0, inplace=True)
        final_out.append(out_image.unsqueeze(0))
        # if DEBUG:
        #  plots(out_image, "out_image")
    out = torch.cat(final_out, dim = 0)
    return out

def GENEO_1_s_c(patterns, image):
    final_out = []
    for pattern in patterns:
        pattern_dim_x, pattern_dim_y = pattern.shape[-2:]  # Get pattern size
        padded_image = F.pad(image, (pattern_dim_y, pattern_dim_y, pattern_dim_x, pattern_dim_x), "constant", 0)

        # Extract patches efficiently using unfold
        unfolded = padded_image.unfold(2, pattern_dim_x, 1).unfold(3, pattern_dim_y, 1)
        abs_diff = torch.abs(unfolded - pattern)  # Compute absolute difference
        out_image = 1 - abs_diff.mean(dim=(-1, -2))  # Mean over the last two dimensions

        # Apply thresholding efficiently
        threshold = out_image[0, 0, 0, 0]
        out_image[out_image < threshold] = 0  # Efficient thresholding

        final_out.append(out_image.unsqueeze(0))

    return torch.cat(final_out, dim=0)

def GENEO_2(functions):
    return torch.mean(functions, dim = 0)
    
def GENEO_3(vectors,functions):
    image_dimensions = functions[0].size()    
    out = torch.zeros(image_dimensions)
    i = torch.arange(image_dimensions[2])
    j = torch.arange(image_dimensions[3])

    for k in range(len(vectors)):
        v_x = torch.floor(vectors[k][0]).to(dtype = torch.long)
        v_y = torch.floor(vectors[k][1]).to(dtype = torch.long)
        v_x_1 = v_x+1
        v_y_1 = v_y+1
        p_x = v_x - vectors[k][0]
        p_y = v_y - vectors[k][1]
        for s_x,s_y,p in [(v_x,v_y,p_x*p_y),\
                        (v_x,v_y_1,p_x*(1-p_y)),\
                        (v_x_1,v_y,(1-p_x)*p_y),\
                        (v_x_1,v_y_1,(1-p_x)*(1-p_y))]:
            out[0][0] += functions[k][0][0][clip(i-s_x)][clip(j-s_y)]*(p)
    out /= k
    return out


def GENEO_3_c (vectors, functions):
    print(functions.shape)
    image_dimensions = functions[0].size()
    out = torch.zeros(image_dimensions)
    
    i_indices = torch.arange(image_dimensions[2]).view(1, -1, 1)
    j_indices = torch.arange(image_dimensions[3]).view(1, 1, -1)
    # vectors = ttorch.tensor(vectors)  # Shape: (K, 2)
    v_x = torch.floor(vectors[:][0]).long()
    v_y = torch.floor(vectors[:][1]).long()
    v_x_1 = v_x + 1
    v_y_1 = v_y + 1
    
    p_x = v_x - vectors[:][0]
    p_y = v_y - vectors[:][1]
    
    coeffs = torch.stack([
        p_x * p_y,
        p_x * (1 - p_y),
        (1 - p_x) * p_y,
        (1 - p_x) * (1 - p_y)
    ]).unsqueeze(-1).unsqueeze(-1)  # Shape: (4, K)
    
    shifts = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])  # Shape: (4, 2)
    
    sum_values = torch.zeros_like(out)
    
    for idx, (dx, dy) in enumerate(shifts):
        s_x = clip(i_indices - (v_x + dx).view(-1, 1, 1))
        s_y = clip(j_indices - (v_y + dy).view(-1, 1, 1))
        print(s_x)
        sampled_values = functions[:, 0, 0, s_x, s_y].sum(dim=0, keepdim=True)  # Efficient batched indexing
        sum_values += sampled_values * coeffs[idx]
    
    out[:, :, :, :] = sum_values / len(vectors)  # Normalize by number of vectors
    return out

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

def GENEO_4(image):
    norm = torch.norm(image, p="inf")
    return 1/norm

