from operator import mod
import numpy as np
import cv2
import scipy
from scipy import signal
from functions.extract_descriptors import filter_keypoints

np.seterr(all='raise')

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    kernel_x = np.array([[0,0,0],[-0.5,0,0.5],[0,0,0]])
    kernel_y = np.array([[0,-0.5,0],[0,0,0],[0,0.5,0]])
    Ix = signal.convolve2d(img, kernel_x[::-1, ::-1], mode='same')
    Iy = signal.convolve2d(img, kernel_y[::-1, ::-1], mode='same')
    
    # Compute local auto-correlation matrix
    # compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    offset = 2 # The window offset. In this case we have a 5x5 window
    offset_gaussian = 2 #The gaussian filter offset. We have a 5x5 window
    Gxx = cv2.GaussianBlur(np.square(Ix), (2*offset_gaussian + 1, 2*offset_gaussian + 1), sigma, borderType=cv2.BORDER_REPLICATE)
    Gyy = cv2.GaussianBlur(np.square(Iy), (2*offset_gaussian + 1, 2*offset_gaussian + 1), sigma, borderType=cv2.BORDER_REPLICATE)
    Gxy = cv2.GaussianBlur(np.multiply(Ix, Iy), (2*offset_gaussian + 1, 2*offset_gaussian + 1), sigma, borderType=cv2.BORDER_REPLICATE)

    # Compute Harris response function
    # compute the Harris response function C here
    C = np.multiply(Gxx, Gyy) - np.multiply(Gxy, Gxy) - k*np.square(Gxx + Gyy)
    
    # Detection with threshold
    # detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    max_filter = scipy.ndimage.filters.maximum_filter(C, size=2*offset + 1)
    # I ckeck if the for which pixels the conditions are met
    conditon_fullfilled = np.logical_and(C > thresh, max_filter == C)
    corners = np.argwhere(conditon_fullfilled == True)
    corners[:, 0], corners[:, 1] = np.copy(corners[:, 1]), np.copy(corners[:, 0])
    #corners = filter_keypoints(img, corners, patch_size=3)
    
    return corners, C

