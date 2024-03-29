import sys
import cv2
import numpy as np
from gradientAnalysis import eraseStrict, eraseLoose
from circle_hough_transform import getCircles, drawCircles
from fixContour import drawContour
from circleOrder import circleOrder
from lineOrder import lineOrder

if len(sys.argv) == 1:
    sys.exit("Need to Specify Image")
else:
     imageName = sys.argv[1]

img = cv2.imread('input/' + imageName + '.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

step_num = 0

print("Getting & Grouping Lines")
drawn_img = np.ones(img.shape)
a, b = lineOrder(img=img,n_steps=3)
for i in range(len(a)):
    for j in range(len(a[i])):
        step_num += 1
        drawn_img[a[i][j],b[i][j]] = 0
        drawn_img = eraseStrict(img, drawn_img)
        cv2.imwrite('steps/' + imageName + '_step'  + str(step_num) + '.png', drawn_img*255)

print("Getting & Grouping Circles")
top_circles, include_arr = getCircles(img, drawn_img)
circle_clusters = circleOrder(top_circles, include_arr)

print("Drawing Contours")
for cluster in circle_clusters:
    print(len(cluster))
    step_num += 1
    for circle in cluster:
        drawContour(img, drawn_img, (circle[0], circle[1]), circle[2], 5)
    drawn_img = eraseStrict(img, drawn_img)
    cv2.imwrite('steps/' + imageName + '_step'  + str(step_num) + '.png', drawn_img*255)

print("Saving Image")
cv2.imwrite('output/' + imageName + '_output.png', drawn_img*255)
cv2.imshow('image', drawn_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
