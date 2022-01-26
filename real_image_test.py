"""
Koristi puno koda koji je postojao vec i mora se nastimavat ovisno o slici, ali sluzi za crtanje crvenog kruga
na zadanoj pravoj slici
"""

import imageio
import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from network import Net
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from params import *
import random
import cv2


def float64_to_uint8(data):
    # info = np.iinfo(data.dtype)  # Get the information of the incoming image type
    data = data.astype(np.float64) / data.max()  # normalize the data to 0 - 1
    # data = 255 * data  # Now scale by 255
    img = data.astype(np.uint8)
    return img


def uint8_to_float64(data):
    # info = np.iinfo(data.dtype)  # Get the information of the incoming image type
    data = data.astype(np.uint8) / data.max()  # normalize the data to 0 - 1
    img = data.astype(np.float64)
    return img


def meta_perimeter(x, y, r, img):
    rr, cc, val = [], [], []

    h, w = 2 * r + 1, 2 * r + 1
    for li, gi in zip(range(h), range(-r + y, r + y)):
        # if i < 0 or i >= img.shape[0]: continue
        for lj, gj in zip(range(w), range(-r + x, r + x)):
            # if j < 0 or j >= img.shape[1]: continue
            if img[li, lj]:
                rr.append(gi)
                cc.append(gj)
                val.append(img[li, lj])

    return np.array(rr), np.array(cc), np.array(val)


def draw_meta(img, row, col, rad):
    meta = Image.open(f'my_datasets/mete/{PICTURE}').convert("L")
    h, w = meta.size
    new_h, new_w = 2 * rad + 1, 2 * rad + 1
    meta = meta.resize((new_h, new_w))
    meta = np.array(meta)
    meta = 1 - uint8_to_float64(meta)

    rr, cc, val = meta_perimeter(row, col, rad, meta)

    valid = (rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1])
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float64)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(30, max(30, radius))
    draw_meta(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img, model_name):
    model = Net(PICTURE_SIZE)
    checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        image = np.expand_dims(np.asarray(img), axis=0)
        image = torch.from_numpy(np.array(image, dtype=np.float32))
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize(image)
        image = image.unsqueeze(0)
        output = model(image)

    return [round(i) for i in (PICTURE_SIZE * output).tolist()[0]]


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return shape0.intersection(shape1).area / shape0.union(shape1).area


def draw_detected(detected, tp, lb):

    x, y, r = detected
    valid_pic = Image.open(tp).convert("RGB")
    enhancer = ImageEnhance.Brightness(valid_pic)
    valid_pic = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(valid_pic)
    valid_pic = enhancer.enhance(1)
    valid_pic = valid_pic.resize((256, 256))
    draw = ImageDraw.Draw(valid_pic)
    draw.ellipse([(y - r, x - r), (y + r, x + r)], outline=(255, 0, 0))
    draw.ellipse([(y - 1, x - 1), (y + 1, x + 1)], outline=(255, 0, 0))
    valid_pic.save(lb)


def main():
    model_name = f'saved_models/{MODEL_NAME}'

    for i in range(312):
        tp = rf'C:\Users\nikol\Desktop\vid\frame{i}.jpg'

        img = Image.open(tp).convert("L")
        img = img.resize((256, 256))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.05)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(0.95)
        img = np.array(img)

        detected = find_circle(img, model_name)
        # print(detected)
        lb = rf'C:\Users\nikol\Desktop\vid_lab\frame{i}_lab.jpg'
        draw_detected(detected, tp, lb)


if __name__ == '__main__':
    main()
