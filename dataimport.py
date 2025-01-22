import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import os
import numpy as np
from funtions import get_sevens, get_points, save_images, save_images_cuts, save_cuts

dataset = torchvision.datasets.MNIST(root = "Dataset", transform=torchvision.transforms.PILToTensor())

sevens = get_sevens(dataset)

filename = "points.txt"
points = get_points(filename)

# save_images(sevens,points)
# save_images_cuts(sevens,points)
save_cuts(sevens,points)


