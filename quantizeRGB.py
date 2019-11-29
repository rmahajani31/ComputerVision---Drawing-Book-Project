import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from PIL import Image

def quantizeRGB(origImg, k):
    obs = np.array(origImg).astype(np.float32)
    dim = obs.shape
    obs = obs.reshape(-1,dim[-1])
    meanColors,outputIndices = kmeans2(obs,k,minit='points')
    closestCenter = meanColors[outputIndices]
    outputImg = closestCenter.reshape(dim).astype(np.uint8)
    return outputImg,meanColors