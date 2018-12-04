import numpy as np
import cv2
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename

img1=cv2.imread("/home/nam/VndMoneyRec_Nam/img_source/500k_2t.jpg",4)
# img2=cv2.imread("/home/nam/VndMoneyRec_Nam/img_source/500k_1.jpg",4)
img2=cv2.imread("/home/nam/VndMoneyRec_Nam/img_test/IMG_20181201_170631.jpg",4)

# Initiate SURF detector
surf=cv2.xfeatures2d.SURF_create()

# find the keypoints and descriptors with SURF
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

# BFMatcher with default params
# trong
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        a=len(good)
        percent=(a*100)/len(kp2)
        print("{} % similarity".format(percent))
        if percent >= 75.00:
            print('Match Found')
        if percent < 75.00:
            print('Match not Found')

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
plt.imshow(img3),plt.show()
