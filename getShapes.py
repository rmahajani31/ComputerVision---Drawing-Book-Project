import numpy as np
import cv2
from matplotlib import pyplot as plt
from edge_hough_transform import getEdges, drawEdges
from circle_hough_transform import getCircles, drawCircles
from gradientAnalysis import eraseEdges, eraseCircles
from getSteps import clusterShapes
from fixContour import contour

def getShapes(imageName):
    img = cv2.imread(imageName + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get Lines using Hough Transform
    top_lines_d, top_lines_theta = getEdges(img)

    # Draw Edges and erase strictly
    drawn_img = np.ones(img.shape)
    drawEdges(drawn_img, top_lines_d, top_lines_theta)
    drawn_img = eraseEdges(img, drawn_img)

    top_circles, include_arr = getCircles(img, drawn_img)
    #drawCircles(drawn_img, top_circles, include_arr)


    print("drawing contours")
    # Draw Circle Contours and erase loosely, Filtering circles will help a lot here
    for i in range(top_circles.shape[0]):
        for j in range(top_circles.shape[1]):
            if top_circles[i,j] is not None and include_arr[i,j] == 1:
                top_circle = top_circles[i,j]
                contour(img, drawn_img, (top_circle[0], top_circle[1]), top_circle[2], 5)

    print("performing gradient analysis")
    drawn_img = eraseCircles(img, drawn_img)

    #Clustering
    '''
    circle_coordinates = []
    for i in range(np.shape(top_circles_a)[0]):
        circle_coordinates.append([top_circles_a[i], top_circles_b[i]])
    circle_labels = clusterShapes(circle_coordinates)
    print(circle_labels)

    line_coordinates = []
    for i in range(np.shape(top_lines_d)[0]):
        line_coordinates.append([top_lines_d[i], top_lines_theta[i]])
    line_labels = clusterShapes(line_coordinates)
    print(line_labels)
    '''

    # Save Image
    cv2.imwrite(imageName + '_output.png', drawn_img)
    cv2.imshow('image', drawn_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    #getShapes('glasses')
    #getShapes('spoon')
    #getShapes('airplane')
    getShapes('bicycle')
    #getShapes('bell')
