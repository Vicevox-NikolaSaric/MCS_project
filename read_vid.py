"""
Zadani video rastavlja na individualne frameove
"""


import cv2
import numpy as np
import glob


vidcap = cv2.VideoCapture(r'my_datasets\test_video\video-mete_square.mp4')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(r"my_datasets\test_video_frames_squared\frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
