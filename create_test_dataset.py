import csv
import imageio
import numpy as np
import random

from PIL import Image

from params import *


def float64_to_uint8(data):
    data = data.astype(np.float64) / data.max()  # normalize the data to 0 - 1
    data = 255 * data  # Now scale by 255
    img = data.astype(np.uint8)
    return img


def uint8_to_float64(data):
    data = data.astype(np.uint8) / data.max()  # normalize the data to 0 - 1
    img = data.astype(np.float64)
    return img


def meta_perimeter(x, y, r, img):
    rr, cc, val = [], [], []

    h, w = 2*r + 1, 2*r + 1
    for li, gi in zip(range(h), range(-r + y, r + y)):
        for lj, gj in zip(range(w), range(-r + x, r + x)):
            if img[li, lj]:
                rr.append(gi)
                cc.append(gj)
                val.append(img[li, lj])

    return np.array(rr), np.array(cc), np.array(val)


def draw_circle(img, row, col, rad):
    meta = Image.open(f'my_datasets/targets/{PICTURE}').convert("L")
    new_h, new_w = 2*rad + 1, 2*rad + 1
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
    draw_circle(img, row, col, rad)

    # Noise
    img += random.uniform(0.01, 1) * noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def img_convert(data):
    data = data.astype(np.float64) / data.max()  # normalize the data to 0 - 1
    data = 255 * data  # Now scale by 255
    img = data.astype(np.uint8)
    return img


def train_set():
    number_of_images = PIC_NUM

    picture_size = PICTURE_SIZE
    max_radius = picture_size // 2
    level_of_noise = NOISE_CREATE

    with open("train_set.csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(number_of_images):
            params, img = noisy_circle(picture_size, max_radius, level_of_noise)
            np.save("my_datasets/train/" + str(i) + ".npy", img)
            img = img_convert(img)
            imageio.imsave("my_datasets/train_pics/" + str(i) + ".png", img[:, :])
            write(outFile, ["my_datasets/train/" + str(i) + ".npy", params[0], params[1], params[2]])


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    train_set()
