import cv2
import numpy as np
from skimage.util import random_noise

def add_gaussian_noise(image, mean=0, var=0.01):
    """
    Add Gaussian noise to an image.
    
    :param image: Input image
    :param mean: Mean of the Gaussian distribution (default: 0)
    :param var: Variance of the Gaussian distribution (default: 0.01)
    :return: Image with added Gaussian noise
    """
    return random_noise(image, mode='gaussian', mean=mean, var=var)

def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian filter to an image.
    
    :param image: Input image
    :param kernel_size: Size of the Gaussian kernel (default: (5, 5))
    :param sigma: Standard deviation in X and Y directions (default: 0)
    :return: Filtered image
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)
