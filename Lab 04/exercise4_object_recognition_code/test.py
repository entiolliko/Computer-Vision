from math import floor
import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = []  # numpy array, [nPointsX*nPointsY, 2]
    #TODO: Why?
    h, w = img.shape
    x_sep = (w - 2*border) / nPointsX
    y_sep = (h - 2*border) / nPointsY
    
    x_dir = np.arange(border, w - border, x_sep)
    y_dir = np.arange(border, h - border, y_sep)

    for i in x_dir:
        for j in y_dir:
            vPoints.append([i, j])
    vPoints = np.asarray(vPoints)
    vPoints.reshape((-1, 2))

    return vPoints

img = np.ones((10,6))
vPoints = grid_points(img, 4, 8, 1)
print(vPoints)
