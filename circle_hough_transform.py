import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def getCircles(img, drawn_img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    print(drawn_img)
    angle_step_size = math.pi / 120
    img_size = np.shape(img)
    #print(type(votes_theta_dim))
    #print(type(votes_d_dim))
    lowerbound = 5
    upperbound = 50
    radius_range = range(lowerbound,upperbound,1)
    votes = np.zeros(shape=(img_size[0],img_size[1],upperbound))
    print(np.shape(votes))
    rows = img_size[0]
    cols = img_size[1]
    r = 25
    gy,gx = np.gradient(edges)
    num_circles = 10
    top_circles = np.empty(num_circles, dtype=object)
    for r in radius_range:
        print("r is", r)
        for y in range(0,rows):
            for x in range(0,cols):
                if edges[y,x] == 255:
                    for theta in np.arange(0,2*math.pi,angle_step_size):
                        a = int(x - r * math.cos(theta))
                        b = int(y + r * math.sin(theta))
                        if b >= 0 and b < rows and a >= 0 and a < cols:
                            votes[b, a, r] = votes[b, a, r] + 1
        index = 0
        getMax = True
        max_row = 0
        max_col = 0
        max_votes = 0
        while index < np.shape(top_circles)[0]:
            if getMax:
                max_votes = np.argmax(votes, axis=None)
                max_row, max_col, radius = np.unravel_index(np.argmax(votes, axis=None), votes.shape)
                getMax = False
            if top_circles[index] is None or max_votes > top_circles[index][3]:
                top_circles[index] = (max_row, max_col, radius, max_votes)
                votes[max_row, max_col, radius] = 0
                getMax = True
            index += 1
        votes.fill(0)

    len = np.shape(top_circles)[0]
    threshold = 0.05
    max_possible_score = 400
    include_arr = np.zeros(shape=(np.shape(top_circles)[0],1))
    print(np.shape(top_circles)[0])
    for i in range(np.shape(top_circles)[0]):
        top_circle = top_circles[i]
        s = np.linspace(0, 2 * np.pi, max_possible_score)
        r = top_circle[0] + top_circle[2] * np.sin(s)
        c = top_circle[1] + top_circle[2] * np.cos(s)
        boundary_points = np.array([r, c]).T
        cur_score = 0
        for j in range(np.shape(boundary_points)[0]):
            boundary_point = boundary_points[j]
            boundary_row = int(boundary_point[0])
            boundary_col = int(boundary_point[1])
            if boundary_row >= 0 and boundary_row < rows and boundary_col >= 0 and boundary_col < cols and edges[boundary_row, boundary_col] == 255 and drawn_img[boundary_row, boundary_col] == 1:
                cur_score += 1
        print("score for circle", i, "is", cur_score)
        if cur_score > threshold * max_possible_score:
            include_arr[i] = 1
    return top_circles, include_arr

def drawCircles(img, top_circles, include_arr):
    for i in range(np.shape(top_circles)[0]):
        if include_arr[i] == 1:
            top_circle = top_circles[i]
            center_row = top_circle[0]
            center_col = top_circle[1]
            r = top_circle[2]
            cv2.circle(img,(int(center_col), int(center_row)), r, (0,0,255), 3)

if __name__=="__main__":
    img = cv2.imread('bicycle.png')
    top_circles, include_arr = getCircles(img)
    img_object = img.copy()
    drawCircles(img_object, top_circles, include_arr)
    cv2.imshow('object', img_object)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
