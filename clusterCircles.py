import numpy as np
import cv2
import math

def sortRadius(circle):
    return circle[2]

def euclidDist(center1, center2):
    return math.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)

def getCircleClusters(top_circles, include_arr):
    circleThresh = 0.03

    # Extract circles to use and sort by radius
    circles = top_circles.flatten()
    inclusion = include_arr.flatten()
    circles = circles[inclusion == 1]
    circles = circles[circles != np.array(None)]
    circles = circles.tolist()
    circles.sort(key=sortRadius, reverse=True)

    # Create the steps using circle threshold
    num_steps = int(len(circles)*circleThresh)
    print(num_steps)
    step_markers = circles[:num_steps]
    del circles[:num_steps]
    num_to_group = len(circles)

    # Compute closest step for circles
    distances = np.zeros(shape=(num_steps, num_to_group))
    for i in range(num_steps):
        for j in range(num_to_group):
            c1 = step_markers[i]
            c2 = circles[j]
            distances[i,j] = euclidDist((c1[0], c1[1]), (c2[0], c2[1]))
    cluster_array = np.argmax(distances, axis=0)

    # Populate Steps
    steps = []
    for i in range(num_steps):
        steps.append([])
        steps[i].append(step_markers[i])
        for j in range(num_to_group):
            if cluster_array[j] == i:
                steps[i].append(circles[j])
    return steps
