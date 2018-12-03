import _thread
import cv2
import getopt
import numpy as np
import pyaudio
import sys
import time
import wave

import common
from common import anorm, getsize

# --------------------------------------------------------------------------------------------------------
dict = {'1k_1': '1k vnd', '1k_2': '1k vnd', '10k_1': '10k vnd', '10k_2': '10k vnd', '20k_1': '20k vnd',
        '20k_2': '20k vnd', '50k_1': '50k vnd', '50k_2': '50k vnd', '100k_1': '100k vnd',
        '100k_2': '100k vnd'}

MIN_POINT = 30
chunk = 1024
FLANN_INDEX_KDTREE = 5
LEN_DICT = len(dict)

def init_feature():
    detector = cv2.xfeatures2d.SIFT_create()
    norm = cv2.NORM_L1
    # http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/

    # flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=MIN_POINT)
    matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio=0.75):  # ratio = 0.75
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs


def explore_match(img1, img2, kp_pairs, status=None, H=None):
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32(cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv2.polylines(vis, [corners], True, (0, 255, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis, showText, (corners[0][0], corners[0][1]), font, 1, (0, 0, 255), 2)
    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)

    # p1 = np.int32([kpp[0].pt for kpp in kp_pairs])
    # p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
    # for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
    #     if inlier:
    #         col = (0, 255, 0)
    #         cv2.circle(vis, (x1, y1), 2, col, -1)
    #         cv2.circle(vis, (x2, y2), 2, col, -1)
    return vis


def match_and_draw(checksound, found, count):
    if (len(kp2) <= 0):
        return None, None, None

    # matching feature
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2

    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

    if len(p1) >= MIN_POINT:
        if not found:
            checksound = True
        found = True
        count = count + 1
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        vis = explore_match(img1, img2, kp_pairs, status, H)

    else:
        found = False
        checksound = False
        H, status = None, None
        count = 0

        vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2

    cv2.imshow('find_obj', vis)

    if (count > 3 and checksound):

        checksound = False
        count = 0

    return found, checksound, count


# --------------------------------------------------------------------------------------------------------
cv2.useOptimized()

cap = cv2.VideoCapture(0)
detector, matcher = init_feature()

checksound = True
found = False
searchIndex = 1
count = 0



img_source_list = []
detectAndCompute_list = []

for each_word in dict:
    img_source = cv2.imread("img_source/%s.jpg" % each_word, 0)
    temp_kp, temp_desc = detector.detectAndCompute(img_source, None)

    img_source_list.append((img_source, dict[each_word]))
    detectAndCompute_list.append((temp_kp, temp_desc))

while (True):
    
    t1 = cv2.getTickCount()
    p = pyaudio.PyAudio()
    # switch template
    if not found:
        if searchIndex <= LEN_DICT:
            img1, showText = img_source_list[searchIndex - 1]
            kp1, desc1 = detectAndCompute_list[searchIndex - 1]

            searchIndex = searchIndex + 1
        else:
            searchIndex = 1
            img1, _ = img_source_list[searchIndex - 1]

    # Capture frame-by-frame
    ret, frame = cap.read()
    # img2 = frame
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # calculate features
    kp2, desc2 = detector.detectAndCompute(img2, None)

    found, checksound, count = match_and_draw(checksound, found, count)

    t2 = cv2.getTickCount()
    # calculate fps
    time = (t2 - t1) / cv2.getTickFrequency()
    print ('FPS = ', 1 / time)

    if cv2.waitKey(1) & 0xFF == 27:
        break

p.terminate()
cap.release()
cv2.destroyAllWindows()
