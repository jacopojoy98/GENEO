import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import torchvision
from funtions import clip, i_clip, plots, plotsim
######   ######
   
class GENEO_MLP_Small_b(nn.Module):
    def __init__(self, patterns, num_classes):
        super().__init__()
        self.patterns = patterns.permute(1,0,2,3)
        self.GENEO2 = nn.Linear(len(patterns), 1)
        self.ACT = nn.Sigmoid()
        self.Finael = nn.Linear(20*20, num_classes)

    def forward(self, x):
        F_k = GENEO_1(self.patterns, x)
        CWM = Channel_wise_max(F_k)
        T_k = self.GENEO2(CWM.permute(0,2, 3, 1))
        T_k = self.ACT(T_k)
        out = self.Finael(T_k.view(T_k.shape[0],-1))
        return self.ACT(out)

class GENEO_thorus(nn.Module):
    def __init__(self, patterns, num_classes):
        super().__init__()
        # self.patterns = patterns.permute(1,0,2,3)
        self.GENEO2 = nn.Linear(len(patterns), num_classes)
        self.ACT = nn.Sigmoid()
        # self.Finael = nn.Linear(20*20, num_classes)

    def forward(self, x):
        # F_k = GENEO_1_thorus(self.patterns, x)
        CWM = x.squeeze(-1).squeeze(-1)
        T_k = self.GENEO2(CWM)
        # T_k = self.ACT(T_k)
        # out = self.Finael(T_k.view(T_k.shape[0],-1))
        return self.ACT(T_k)

class patterns_preprocess():
    def __init__(self,patterns):
        self.patterns = patterns.permute(1,0,2,3)
    
    def __call__(self,x):
        return GENEO_1_thorus(self.patterns, x)

def Channel_wise_max(tensor):
    # Get the shape of the tensor
    b, patterns_size, rows, cols = tensor.shape
    # Flatten the last two dimensions to find the argmax for each matrix
    flat_tensor = tensor.view(b, patterns_size, -1)
    max_indices = flat_tensor.argmax(dim=2)
    # Create a mask of zeros
    mask = torch.zeros_like(flat_tensor, dtype=torch.float)
    # Set the maximum value positions to 1
    batch_indices = torch.arange(b).unsqueeze(1).expand_as(max_indices)  # Shape (b, patterns_size)
    pattern_indices = torch.arange(patterns_size).unsqueeze(0).expand_as(max_indices)  # Shape (b, patterns_size)

    mask[batch_indices, pattern_indices, max_indices] = 1

    # Reshape the mask back to the original tensor shape
    mask = mask.view(b, patterns_size, rows, cols)

    # Retain only the maximum values
    result = tensor * mask

    return result


def GENEO_1(patterns, image):

    # patterns = (K, 1, H_p, W_p)
    # image    = (B, 1, 28, 28)
    _ , K, H_p, W_p = patterns.size()
    B, _, H_img, W_img = image.size()

    # Pad the image to handle borders
    padded_image = F.pad(image, (W_p//2, W_p//2, H_p//2, H_p//2), mode="constant", value=0)

    # Extract sliding windows (unfolded image)
    unfolded_image = padded_image.unfold(2, H_p, 1).unfold(3, W_p, 1)  # (1, 1, H_out, W_out, H_p, W_p)
    H_out, W_out = unfolded_image.size(2), unfolded_image.size(3)
    
    # Reshape to (1, 1, H_out * W_out, H_p * W_p)
    unfolded_image = unfolded_image.contiguous().view(B, 1, H_out * W_out, H_p * W_p)
    # print(unfolded_image.shape)
    
    # Reshape patterns for broadcasting: (K, 1, H_p*W_p)
    patterns_reshaped = patterns.view(K, 1, H_p * W_p)
    # print(patterns_reshaped.shape)

    # Calculate absolute difference and normalize
    difference = torch.abs(patterns_reshaped - unfolded_image)
    # print(difference.shape)
    normalized_diff = torch.mean(difference, dim=-1)  # (K, 1, H_out * W_out)
    # Compute output
    out_image = 1 - normalized_diff
    out_image = out_image.view(B, K, H_out, W_out)

    return out_image

def GENEO_1_thorus(patterns, image):

    # patterns = (K, 1, H_p, W_p)
    # image    = (B, 1, 28, 28)
    _ , K, H_p, W_p = patterns.size()
    B, _, H_img, W_img = image.size()

    # Pad the image to handle borders
    padded_image = F.pad(image, (W_p//2, W_p//2, H_p//2, H_p//2), mode="circular")

    # Extract sliding windows (unfolded image)
    unfolded_image = padded_image.unfold(2, H_p, 1).unfold(3, W_p, 1)  # (1, 1, H_out, W_out, H_p, W_p)
    H_out, W_out = unfolded_image.size(2), unfolded_image.size(3)
    
    # Reshape to (1, 1, H_out * W_out, H_p * W_p)
    unfolded_image = unfolded_image.contiguous().view(B, 1, H_out * W_out, H_p * W_p)
    # print(unfolded_image.shape)
    
    # Reshape patterns for broadcasting: (K, 1, H_p*W_p)
    patterns_reshaped = patterns.view(K, 1, H_p * W_p)
    # print(patterns_reshaped.shape)

    # Calculate absolute difference and normalize
    difference = torch.abs(patterns_reshaped - unfolded_image)
    # print(difference.shape)
    normalized_diff = torch.mean(difference, dim=-1)  # (K, 1, H_out * W_out)
    # Compute output
    out_image = 1 - normalized_diff
    out_image = out_image.view(B, K, H_out, W_out)
    out_flat = torch.nn.functional.max_pool2d(out_image,(H_out, W_out))
    return out_flat


if __name__ =="__main__":
    blur = torchvision.transforms.GaussianBlur((5,5), sigma=(0.1, 2.0))
    matrix = torch.rand((1,6,6))
    matrix = blur(matrix)
    plt.imshow(matrix.squeeze().flatten().unsqueeze(1),cmap="gray")
    plt.savefig("tmp.png")