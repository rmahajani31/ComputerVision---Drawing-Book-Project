import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Only Check gradient of single cell
def eraseStrict(origImg, drawnImg):
    sobelx = cv.Sobel(origImg, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(origImg, cv.CV_64F, 0, 1 ,ksize=5)

    imgShape = origImg.shape
    for row in range(0, imgShape[0]):
        for col in range(0, imgShape[1]):
            if drawnImg[row][col] == 0 and sobelx[row][col] == 0 and sobely[row][col] == 0:
                drawnImg[row][col] = 1

    return drawnImg

# Check gradient of neighbors in 9-box
def eraseLoose(origImg, drawnImg):
    sobelx = cv.Sobel(origImg, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(origImg, cv.CV_64F, 0, 1 ,ksize=5)

    imgShape = origImg.shape
    for row in range(0, imgShape[0]):
        for col in range(0, imgShape[1]):
            if drawnImg[row][col] == 0 and checkNeighborGradients(origImg, row, col, imgShape, sobelx, sobely):
                drawnImg[row][col] = 1

    return drawnImg

def checkNeighborGradients(img, row, col, imgShape, sobelx, sobely):
    row_neighbors = [row]
    col_neighbors = [col]

    if row-1 >= 0:
        row_neighbors.append(row-1)
    if row+1 < imgShape[0]:
        row_neighbors.append(row+1)

    if col-1 >= 0:
        col_neighbors.append(col-1)
    if col+1 < imgShape[1]:
        col_neighbors.append(col+1)

    for row in row_neighbors:
        for col in col_neighbors:
            if sobelx[row][col] != 0 or sobely[row][col] != 0:
                return False
    return True
