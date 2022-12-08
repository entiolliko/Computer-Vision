import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    height = frame.shape[0]
    width = frame.shape[1]
    
    selected_frame = frame[ymin:ymax, xmin:xmax, :]

    R_histo, RBin_edges = np.histogram(selected_frame[:, :, 0], hist_bin)
    G_histo, RBin_edges = np.histogram(selected_frame[:, :, 1], hist_bin)
    B_histo, RBin_edges = np.histogram(selected_frame[:, :, 2], hist_bin)

    result = np.asarray([R_histo, G_histo, B_histo])
    result = result / np.sum(result) #We are normalizing over all the values in the histogram
    return result