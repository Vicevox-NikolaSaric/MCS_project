"""
Individualne frameove spaja nazad u video
"""

import cv2
import numpy as np
import glob

img_array = []
for i in range(364):
    filename = rf'C:\Users\nikol\Desktop\vid_lab\frame{i}_lab.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(r'C:\Users\nikol\Desktop\test_1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
