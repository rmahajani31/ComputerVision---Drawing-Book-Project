import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists

img = Image.open("fish.jpg")

qRGB,meanRGB = quantizeRGB(img,5)
plt.imshow(qRGB)
plt.title("RGB Quantization, k = 5")
plt.show()
print("Quantized RGB SSD Error:",computeQuantizationError(img,qRGB))

qHSV,meanHues = quantizeHSV(img,5)
plt.imshow(qHSV)
plt.title("Hue Quantization, k = 5")
plt.show()
print("Quantized HSV SSD Error:",computeQuantizationError(img,qHSV))

histEqual,histClustered = getHueHists(img,5)
plt.bar(x=histEqual[1][:-1],height=histEqual[0],width=np.diff(histEqual[1]),align='edge',ec='k')
plt.title("Histogram with equally distributed bars")
plt.show()
plt.bar(x=histClustered[1][:-1],height=histClustered[0],width=np.diff(histClustered[1]),align='edge',ec='k')
plt.title("Histogram with cluster-centered bars")
plt.show()