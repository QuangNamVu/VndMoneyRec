import numpy as np
import cv2

im=cv2.imread("/home/nam/VndMoneyRec_Nam/img_test/IMG_20181201_170631.jpg",4)

im=cv2.imread("/home/nam/Tiền/abcd.jpg",4)


imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, 127, 255, 0)

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hull = []

for i in range(len(contours)):
    hull.append(cv2.convexHull(contours[i], False))

drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

max_idx = 0
max_contour = contours[0]
for idx, each_contour in enumerate(contours):
    if cv2.contourArea(each_contour) > cv2.contourArea(max_contour):
        max_contour = each_contour
        max_idx = idx


color_contours = (0, 255, 0)
color = (255, 0, 0)
# draw ith contour
cv2.drawContours(im, contours, max_idx, color_contours, 3, 8, hierarchy)
# draw ith convex hull object point
cv2.drawContours(im, hull, i, max_idx, 1, 100)

# cv2.imwrite( "/home/nam/VndMoneyRec_Nam/output/Countour_convexHull.jpg", im)
rect = cv2.boundingRect(max_contour)
x, y, w, h = rect
cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('dst',im)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
print(max_idx)
print(len(contours))
