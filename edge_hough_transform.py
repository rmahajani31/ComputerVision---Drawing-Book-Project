import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def getEdges(img):
    angle_step_size = math.pi / 12
    votes_theta_dim = int(2*math.pi/angle_step_size)
    img_size = np.shape(img)
    votes_d_dim = int(math.sqrt(math.pow(img_size[0],2)+math.pow(img_size[1],2)))
    #print(type(votes_theta_dim))
    #print(type(votes_d_dim))
    votes = np.zeros((votes_d_dim, votes_theta_dim))
    rows = np.shape(img)[0]
    cols = np.shape(img)[1]
    highest_theta = 0
    highest_d = 0
    highest_votes = 0
    for y in range(0,rows):
        for x in range(0,cols):
            if img[y,x] == 0:
                for theta in np.arange(0,2*math.pi,angle_step_size):
                    d = math.ceil(x * math.cos(theta) + y * math.sin(theta))
                    votes[d,math.ceil(theta/angle_step_size)] = votes[d, math.ceil(theta/angle_step_size)] + 1

    original_votes = votes
    top_lines_theta = np.zeros(30)
    top_lines_d = np.zeros(30)
    index = 0
    for i in range(np.shape(top_lines_theta)[0]):
        max_row,max_col = np.unravel_index(np.argmax(votes, axis=None), votes.shape)
        top_lines_d[index] = max_row
        top_lines_theta[index] = max_col * angle_step_size
        votes[max_row,max_col] = 0
        index = index+1

    return top_lines_d, top_lines_theta

def drawEdges(img, top_lines_d, top_lines_theta):
    for i in range(np.shape(top_lines_d)[0]):
        highest_d = top_lines_d[i]
        highest_theta = top_lines_theta[i]

        a = np.cos(highest_theta)
        b = np.sin(highest_theta)
        x0 = highest_d * a
        y0 = highest_d * b
        x1 = math.ceil(x0 + 1000 * (-b))
        y1 = math.ceil(y0 + 1000 * (a))
        x2 = math.ceil(x0 - 1000 * (-b))
        y2 = math.ceil(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 2)
