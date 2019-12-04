import sys
import numpy as np
import cv2
from edge_hough_transform import getEdges, drawEdges
from circle_hough_transform import getCircles, drawCircles
from gradientAnalysis import eraseEdges, eraseCircles
from fixContour import contour
from clusterCircles import getCircleClusters
from lineOrder import lineOrder

if len(sys.argv) == 1:
    sys.exit("Need to Specify Image")
else:
     imageName = sys.argv[1]

img = cv2.imread('input/' + imageName + '.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get Lines using Hough Transform
top_lines_d, top_lines_theta = getEdges(img)

# Draw Lines and erase strictly
drawn_img = np.ones(img.shape)
#drawEdges(drawn_img, top_lines_d, top_lines_theta)
a, b = lineOrder(img=img,n_steps=3)
for i in range(len(a)):
    for j in range(len(a[i])):
        drawn_img[a[i][j],b[i][j]] = 0
        cv2.imshow('image', drawn_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
drawn_img = eraseEdges(img, drawn_img)


# Get and Group Circles
top_circles, include_arr = getCircles(img, drawn_img)
circle_clusters = getCircleClusters(top_circles, include_arr)

print("drawing contours")
# Draw Circle Contours and erase loosely, Filtering circles will help a lot here
for cluster in circle_clusters:
    for circle in cluster:
        contour(img, drawn_img, (circle[0], circle[1]), circle[2], 5)

    print("performing gradient analysis")
    drawn_img = eraseEdges(img, drawn_img)
    cv2.imshow('image', drawn_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save Image
cv2.imwrite('output/' + imageName + '_output.png', drawn_img)
cv2.imshow('image', drawn_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
if __name__=="__main__":
    #getShapes('glasses')
    #getShapes('spoon')
    #getShapes('airplane')
    getShapes('bicycle')
    #getShapes('bell')
'''
