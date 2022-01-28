import cv2
import numpy as np
import glob
from params import *

img_array = []
for i in range(NUM_OF_FRAMES):
    filename = rf"frame{i}_lab.jpg"
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter(r"test_1_trace.avi", cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
