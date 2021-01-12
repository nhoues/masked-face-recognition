import matplotlib.pyplot as plt

import numpy as np
from skimage import draw
from matplotlib import path
from imutils import face_utils 
import argparse 
import dlib
import cv2
import imutils
import matplotlib.pyplot as plt

def get_mask_from_polygon_mpl(image_shape, polygon):
  """Get a mask image of pixels inside the polygon.

  Args:
    image_shape: tuple of size 2.
    polygon: Numpy array of dimension 2 (2xN).
  """
  xx, yy = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
  xx, yy = xx.flatten(), yy.flatten()
  indices = np.vstack((xx, yy)).T
  mask = path.Path(polygon).contains_points(indices)
  mask = mask.reshape(image_shape)
  mask = mask.astype('bool')
  return mask


def get_mask_from_polygon_skimage(image_shape, polygon):
  """Get a mask image of pixels inside the polygon.

  Args:
    image_shape: tuple of size 2.
    polygon: Numpy array of dimension 2 (2xN).
  """
  vertex_row_coords = polygon[:, 1]
  vertex_col_coords = polygon[:, 0]
  fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, image_shape)
  mask = np.zeros(image_shape, dtype=np.bool)
  mask[fill_row_coords, fill_col_coords] = True
  return mask