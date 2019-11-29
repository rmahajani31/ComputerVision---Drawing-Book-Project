import numpy as np
from scipy.cluster.vq import kmeans2
from skimage.color import rgb2hsv,hsv2rgb
import matplotlib.pyplot as plt
from PIL import Image

def quantizeHSV(origImg, k):
    obs = np.array(origImg)
    obs = rgb2hsv(obs)
    hues = obs[:,:,0].reshape(-1,1)
    meanHues,meanIndices = kmeans2(hues,k,minit='points')
    obs[:,:,0] = meanHues[meanIndices].reshape(obs.shape[0],obs.shape[1])
    obs = hsv2rgb(obs)*255
    outputImg = obs.astype(np.uint8)
    return outputImg,meanHues