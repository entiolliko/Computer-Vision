import numpy as np
from scipy.spatial.distance import cdist

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    dist = np.sqrt(cdist(desc1, desc2, metric='sqeuclidean'))
    return dist

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = matches = one_way_corners(distances)
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        return matches
    elif method == "mutual":
        # I check if the matches that are detected with the inverted image are the same as those with the normal image.
        matches_t = one_way_corners(np.transpose(distances))        
        matches_t[:, 0], matches_t[:, 1] = np.copy(matches_t[:, 1]), np.copy(matches_t[:, 0])
        matches = intersection_2d(matches, matches_t) #If the match is the same it will be in both detections
    elif method == "ratio":
        temp_distances = np.sort(distances, axis=-1) # We put the smallest and second smallest value of each row one after the other 
        cond = np.divide(temp_distances[:, 0], temp_distances[:, 1]) < ratio_thresh # We check if the condition is verified
        args_of_interest = np.argwhere(cond == True)
        args_of_interest.reshape(args_of_interest.shape[0])
        
        matches = matches[args_of_interest] # We use only the matches for which the condition is true
        matches = matches.reshape((matches.shape[0], matches.shape[2]))
    else:
        raise NotImplementedError
    return matches


def one_way_corners(distances):
    min_cord = np.argmin(distances, axis=-1)
    matches = np.empty((min_cord.shape[0], 2), dtype=int)
    matches[:, 0] = np.arange(0, min_cord.shape[0])
    matches[:, 1] = min_cord
    return matches

def intersection_2d(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
        'formats':ncols * [A.dtype]}

    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    return C