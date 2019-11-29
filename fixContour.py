import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

def contour(img, drawn_img, center, radius, epsilon):
    s = np.linspace(0, 2*np.pi, 400)
    r = center[0] + (radius+epsilon)*np.sin(s)
    c = center[1] + (radius+epsilon)*np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=0.001, coordinates='rc')

    for i in range(snake.shape[0]):
        drawn_img[int(snake[i,0]), int(snake[i,1])] = 0
