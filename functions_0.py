import numpy as np
import torch
import matplotlib.pyplot as plt

def circle(r,x,y,c_x,c_y):
    if (c_x-x)**2+(c_y-y)**2<r**2 and (c_x-x)**2+(c_y-y)**2>(r-1)**2:
        return True
    else:
        return False

def Circle_indexes(lenght,radius,center_x,center_y):
    grid = np.zeros((lenght,lenght))
    for i in range(lenght):
        for j in range(lenght):
            if circle(radius,i,j,center_x,center_y):
                grid[j,i] = 1
    return np.argwhere(grid ==1)


def Circle(lenght,radius,center_x,center_y):
    grid = np.zeros((lenght,lenght))
    for i in range(lenght):
        for j in range(lenght):
            if circle(radius,i,j,center_x,center_y):
                grid[j,i] = 1
    return grid

def F_circle(r,x,y,c_x,c_y):
    if (c_x-x)**2+(c_y-y)**2<r**2:
        return True
    else:
        return False

def F_Circle_indexes(lenght,radius,center_x,center_y):
    grid = np.zeros((lenght,lenght))
    for i in range(lenght):
        for j in range(lenght):
            if F_circle(radius,i,j,center_x,center_y):
                grid[j,i] = 1
    return np.argwhere(grid ==1)


def F_Circle(lenght,radius,center_x,center_y):
    grid = np.zeros((lenght,lenght))
    for i in range(lenght):
        for j in range(lenght):
            if F_circle(radius,i,j,center_x,center_y):
                grid[j,i] = 1
    return grid

