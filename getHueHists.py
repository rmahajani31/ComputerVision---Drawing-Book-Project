import numpy as np
from quantizeHSV import quantizeHSV
from skimage.color import rgb2hsv,hsv2rgb
import matplotlib.pyplot as plt
from PIL import Image

def getHueHists(im, k):
    oImg,meanHues = quantizeHSV(im,k)
    obs = rgb2hsv(oImg)
    hues = obs[:,:,0].reshape(-1,1)

    meanHues = sorted(meanHues)
    clusterBins = [0.0]
    for i in range(1,k):
        clusterBins.append((meanHues[i]+meanHues[i-1])/2.0)
    clusterBins.append(1.0)
    clusterBins = np.array(clusterBins)
    
    histEqual = np.histogram(hues,bins=k,range=(0.0,1.0))
    histClustered = np.histogram(hues,bins=clusterBins)
    return (histEqual,histClustered)