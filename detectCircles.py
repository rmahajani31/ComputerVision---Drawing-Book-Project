import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.draw import circle_perimeter
from skimage.morphology import binary_dilation
from scipy.ndimage import convolve1d
from PIL import Image
import math

def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])

class Accumulator(object):
    def __init__(self,im,r,useGradient,binSize):
        gim = rgb2gray(im)
        e = feature.canny(gim,sigma=5)
        self.im = e.astype(np.uint8)
        self.gim = gim
        self.r = r
        self.useGradient = useGradient
        self.binSize = binSize
        self.mat = np.zeros(np.ceil(np.array(self.im.shape)/binSize).astype(int),np.uint8)

    def build(self):
        if self.useGradient:
            gradient_filter = [1.0, -1.0]
            gx = convolve1d(self.gim,gradient_filter)
            gy = convolve1d(self.gim,gradient_filter,axis=0)
            self.gradientAngle = np.arctan2(gy,gx)
        angle_thresh = math.pi/6
        for i in range(self.im.shape[0]):
            for j in range(self.im.shape[1]):
                if self.im[i,j]>0:
                    rr,cc = circle_perimeter(i,j,self.r,shape=self.im.shape)
                    if self.useGradient:
                        angles = np.arctan2(cc-j,rr-i)
                        angle_diff = np.abs(angles-self.gradientAngle[i,j])
                        angle_diff = np.minimum(angle_diff,np.abs(angle_diff-math.pi))
                        angle_diff = np.minimum(angle_diff,np.abs(angle_diff-(2.0*math.pi)))
                        angle_condition = angle_diff<=angle_thresh
                        rr = np.extract(angle_condition,rr)
                        cc = np.extract(angle_condition,cc)
                    rr = np.floor(rr/self.binSize).astype(int)
                    cc = np.floor(cc/self.binSize).astype(int)
                    self.mat[rr,cc]+=1

    def find(self):
        sorted_points = np.flip(np.argsort(self.mat.ravel()),axis=0)
        sorted_points = np.array(np.unravel_index(sorted_points,self.mat.shape)).T
        thresh = 0.7*self.mat[sorted_points[0,0],sorted_points[0,1]]
        suppress_dim = 5
        centers = []
        search_mat = self.mat.copy()
        for r,c in sorted_points:
            if search_mat[r,c]>=thresh:
                centers.append([r*self.binSize,c*self.binSize])
                search_mat[max(r-suppress_dim,0):r+suppress_dim,max(c-suppress_dim,0):c+suppress_dim] = 0
        return np.array(centers)

    def display(self):
        plt.title("Accumulator\nRadius: {0}, Bin Size: {1}\nUse Gradient: {2}, Angle: {3}".format(self.r,self.binSize,self.useGradient,30))
        plt.imshow(self.mat,cmap='gray')
        plt.show()

def detectCircles(im, radius, useGradient):
    binSize = 1
    a = Accumulator(im, radius, useGradient, binSize)
    a.build()
    a.display()
    centers = a.find()
    return centers

def drawCircles(im, centers, radius, binSize):
    nim = np.array(im)
    circles = np.zeros(nim.shape[:-1],np.uint8)
    for c in centers:
        rr,cc = circle_perimeter(c[0],c[1],radius,shape=nim.shape)
        circles[rr,cc] = 1
    circles = binary_dilation(circles)
    rr,cc = np.nonzero(circles)
    nim[rr,cc] = [255,0,0]
    
    plt.title("Marked Circles\nRadius: {}, Bin Size: {}".format(radius,binSize))
    plt.imshow(nim)
    plt.show()

# Uncomment lines to try
img = Image.open("bicycle_1.png")
centers = detectCircles(img,25,0)
drawCircles(img, centers, 25, 1)