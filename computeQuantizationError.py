import numpy as np

def computeQuantizationError(origImg,quantizedImg):
    error = np.sum(np.square(np.array(origImg)-np.array(quantizedImg)))
    return error