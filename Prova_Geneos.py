import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from prova import Circle_indexes, Circle, F_Circle

class GENEO(nn.Module):
    def __init__(self, r_1,r_2,h_3,r_4):
        super().__init__()
        self.r_1 = r_1 
        self.r_2 = r_2 
        self.h_3 = h_3 
        self.r_4 = r_4 

        self.p1 = torch.nn.Parameter(torch.randn(1))
        self.p2 = torch.nn.Parameter(torch.randn(1))
        self.p3 = torch.nn.Parameter(torch.randn(1))
        self.p4 = torch.nn.Parameter(torch.randn(1))
        self.p5 = torch.nn.Parameter(torch.randn(1))
        self.p6 = torch.nn.Parameter(torch.randn(1))
        self.p7 = torch.nn.Parameter(torch.randn(1))
        self.p8 = torch.nn.Parameter(torch.randn(1))


    def forward(self, x):
        L1= (self.p1 * GENEO_1(self.r_1, x) + \
            self.p2 * GENEO_1(self.r_2, x) + \
            self.p3 * GENEO_1(self.h_3, x) + \
            self.p4 * GENEO_1(self.r_4, x)) /\
            (self.p1 + self.p2 + self.p3 + self.p4)
        
        L2 = (self.p1 * GENEO_1(self.r_1, L1) + \
            self.p2 * GENEO_1(self.r_2, L1) + \
            self.p3 * GENEO_1(self.h_3, L1) + \
            self.p4 * GENEO_1(self.r_4, L1)) /\
            (self.p1 + self.p2 + self.p3 + self.p4)

        return L2
    
    
def GENEO_1(r,image): # Works if r<=l//2
    output_image = image.detach().clone()
    l = len(image)
    indexes = Circle_indexes(l, r, l//2, l//2)- [l//2,l//2]
    for c_y,row in enumerate(image):
        for c_x, element in enumerate(row):
            adjusted_indexes = indexes + [c_y,c_x]
            valid_indexes = adjusted_indexes[(adjusted_indexes[:,0]>=0)&\
                                             (adjusted_indexes[:,0]<l )&\
                                             (adjusted_indexes[:,1]>=0)&\
                                             (adjusted_indexes[:,1]<l)]
            differences = 1 - torch.abs(image[c_y, c_x] - 
                                       image[valid_indexes[:, 0], valid_indexes[:, 1]])
            output_image[c_y, c_x] = torch.max(differences)# * image[c_y, c_x]
    return output_image


def GENEO_2(r,image):
    output_image = image.detach().clone()
    l = len(image)
    indexes = Circle_indexes(l, r, l//2, l//2)- [l//2,l//2]
    for c_y,row in enumerate(image):
        for c_x, element in enumerate(row):
            adjusted_indexes = indexes + [c_y,c_x]
            valid_indexes = adjusted_indexes[(adjusted_indexes[:,0]>=0)&\
                                             (adjusted_indexes[:,0]<l )&\
                                             (adjusted_indexes[:,1]>=0)&\
                                             (adjusted_indexes[:,1]<l)]
            differences = 1 - torch.abs(image[c_y, c_x] - 
                                       image[valid_indexes[:, 0], valid_indexes[:, 1]])
            output_image[c_y, c_x] = torch.mean(differences)# * image[c_y, c_x]
    return output_image
    

def GENEO_3(h,image):
    output_image = image.detach().clone()
    for j, r in enumerate(image):
        for i,e in enumerate(r):
            output_image[i][j] = (image[i][j]*h + 1/2)//1
    return output_image

def GENEO_4(r,image):
    output_image = image.detach().clone()
    l = len(image)
    indexes = Circle_indexes(l, r, l//2, l//2)- [l//2,l//2]
    for c_y,row in enumerate(image):
        for c_x, element in enumerate(row):
            adjusted_indexes = indexes + [c_y,c_x]
            valid_indexes = adjusted_indexes[(adjusted_indexes[:,0]>=0)&\
                                             (adjusted_indexes[:,0]<l )&\
                                             (adjusted_indexes[:,1]>=0)&\
                                             (adjusted_indexes[:,1]<l)]

            output_image[c_y, c_x] = torch.mean(image[valid_indexes])# * image[c_y, c_x]
    return output_image



lenght = 100
radius = 5
c_x = 20
c_y = 40

Model = GENEO(7)
image = torch.tensor(F_Circle(lenght,radius,c_x,c_y))
image_2 = torch.rand((100,100))

output_image = Model(image_2)
print(output_image)

plt.imshow(image_2,cmap="gray_r")
plt.show()
plt.close()
plt.imshow(output_image,cmap="gray_r")
plt.show()





