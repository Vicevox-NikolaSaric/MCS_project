import cv2
import numpy as np
import glob


vidcap = cv2.VideoCapture(r'')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(r"frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
