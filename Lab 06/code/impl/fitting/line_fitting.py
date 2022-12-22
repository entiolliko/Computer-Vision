import numpy as np
from matplotlib import pyplot as plt
import random
import math

np.random.seed(0)
random.seed(0)

def least_square(x,y):
	# TODO
	# return the least-squares solution
	# you can use np.linalg.lstsq
	assert x.shape == (x.shape[0],), "x wrong shape: " + str(x.shape)
	assert y.shape == (y.shape[0],), "y wrong shape: " + str(y.shape)
	
	x = np.vstack([x, np.ones((x.shape[0]))]).T
	
	k_ls, b_ls = np.linalg.lstsq(x, y, rcond=None)[0]
	return k_ls, b_ls

def distance_point_line(x, y, m, b):
	return np.abs(m*x - y + b)/math.sqrt(m**2 + 1)

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# compute the number of inliers and a mask that denotes the indices of inliers
	mask = distance_point_line(x, y, k, b) < thres_dist
	assert mask.shape == x.shape, "Mask has the wrong shape: " + str(mask.shape)
	return np.argwhere(mask == True).shape[0], mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	# TODO
	# ransac
	k_ransac = None
	b_ransac = None
	inlier_mask = None
	best_inliers = 0

	for i in range(iter):
		rng = np.random.default_rng()
		data_subset_idx = rng.choice(x.shape[0], num_subset, replace=False)
		k_ls, b_ls = least_square(x[data_subset_idx], y[data_subset_idx])
		num_inliners, mask = num_inlier(x, y, k_ls, b_ls, n_samples, thres_dist)
		if num_inliners > best_inliers:
			k_ransac = k_ls
			b_ransac = b_ls
			best_inliers = num_inliners
			inlier_mask = mask

	return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()