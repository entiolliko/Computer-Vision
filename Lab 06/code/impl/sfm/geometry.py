import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # Normalize coordinates (to points on the normalized image plane) Are we talking about the matches? The im1 and im2 are just the images so nothing to normalize there
  K_inv = np.linalg.inv(K)
  kp1_cord = im1.kps[matches[:, 0]]
  kp2_cord = im2.kps[matches[:, 1]]
  kp1_cord = np.concatenate((kp1_cord, np.ones((kp1_cord.shape[0], 1))), axis=1)
  kp2_cord = np.concatenate((kp2_cord, np.ones((kp2_cord.shape[0], 1))), axis=1)
  
  normalized_kps1 = np.transpose(np.matmul(K_inv, np.transpose(kp1_cord)))
  normalized_kps2 = np.transpose(np.matmul(K_inv, np.transpose(kp2_cord)))

  assert normalized_kps1.shape == (matches.shape[0], 3), "The shape of the matrix is : " + str(normalized_kps1.shape)
  assert normalized_kps2.shape == (matches.shape[0], 3), "The shape of the matrix is : " + str(normalized_kps2.shape)

  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    # Add the constraints
    m1 = normalized_kps1[i]
    m2 = normalized_kps2[i]
    m1 = np.concatenate((m1, m1, m1))
    m2 = np.repeat(m2, 3)
    constraint_matrix[i] = m1*m2
    
  # Solve for the nullspace of the constraint matrix
  _, _1, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # Reshape the vectorized matrix to it's proper shape again
  assert vectorized_E_hat.shape == (9,) , "vectorized_E_hat has not the right shape. Current shape: " + str(vectorized_E_hat.shape)
  E_hat = np.reshape(vectorized_E_hat, (3,3))
  E_hat = E_hat

  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  u, s, vh = np.linalg.svd(E_hat, False)
  s = np.asarray([1,1,0])
  E = u @ np.diag(s) @ vh

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[i,:]
    kp2 = normalized_kps2[i,:]

    assert(abs(kp2.transpose() @ E @ kp1) < 0.01), "i = " + str(i) + " result = " + str(abs(kp2.transpose() @ E @kp1))
  
  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]
  
  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]

  
  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  # Filter points behind the first camera

  hom_points3D = np.concatenate((points3D, np.ones((points3D.shape[0], 1))), axis=1)
  #Filter points in the first camera
  pix_coord = np.transpose(np.matmul(P1, np.transpose(hom_points3D)))
  mask = np.argwhere(pix_coord[:, 2] > 0) #Greater then what?

  assert mask.shape[1] == 1, "The mask had a different shape"
  mask = np.reshape(mask, mask.shape[0])
  im1_corrs = im1_corrs[mask]
  im2_corrs = im2_corrs[mask]
  points3D = points3D[mask]


  hom_points3D = np.concatenate((points3D, np.ones((points3D.shape[0], 1))), axis=1)
  # Filter points behind the second camera
  pix_coord = np.transpose(np.matmul(P2, np.transpose(hom_points3D)))
  mask = np.argwhere(pix_coord[:, 2] > 0) #Greater then what?

  assert mask.shape[1] == 1, "The mask had a different shape"
  mask = np.reshape(mask, mask.shape[0])
  im1_corrs = im1_corrs[mask]
  im2_corrs = im2_corrs[mask]
  points3D = points3D[mask]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  K_inv = np.linalg.inv(K)
  hom_points2D = np.concatenate((points2D, np.ones((points2D.shape[0], 1))), axis=1)
  normalized_points2D = np.transpose(np.matmul(K_inv, np.transpose(hom_points2D)))

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}
  for other_image in registered_images:
    offset = points3D.shape[0]
    e_matches = GetPairMatches(image_name, other_image, matches)  
    points3D_new, im1_corrs, im2_corrs = TriangulatePoints(K, image, images[other_image], e_matches) #Stesso ordine del file sfm
    points3D = np.append(points3D, points3D_new, 0)

    corrs[image_name] = np.array([im1_corrs, offset + np.arange(points3D_new.shape[0])]) #Offset dell'indice per questa immagine 
    corrs[other_image] = np.array([im2_corrs, offset + np.arange(points3D_new.shape[0])])
  return points3D, corrs
  
