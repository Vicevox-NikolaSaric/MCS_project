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
from trace_line import trace


def find_circle(model, img):
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


def draw_detected_16_9(detected, tp, lb):

    y, x, r = detected
    valid_pic = Image.open(tp).convert("RGB")

    draw = ImageDraw.Draw(valid_pic)
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=(255, 0, 0))
    draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], outline=(255, 0, 0))
    valid_pic.save(lb)


def main():
    model_name = f'saved_models\\{MODEL_NAME}'
    model = Net(PICTURE_SIZE)
    checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    inCsvFile = open("traced_video.csv", 'w', newline='')
    trace(inCsvFile, ["x", "y"])
    for i in range(NUM_OF_FRAMES):
        tp = rf"frame{i}.jpg"

        img = Image.open(tp).convert("L")
        img = img.resize((256, 256))
        img = np.array(img)

        detected = find_circle(model, img)

        detected[0] = round(detected[0] * (1080. / 256.))
        detected[1] = round(detected[1] * (1920. / 256.))
        detected[2] = round(detected[2] * (1080. / 256.))

        lb = rf"frame{i}_lab.jpg"
        draw_detected_16_9(detected, tp, lb)

        y = detected[0]
        x = detected[1]
        r = detected[2]

        # Stationary pic radius / detected radius
        scale = 262./r

        dist_x = round((960. - x) * scale)
        dist_y = round((540. - y) * scale)
        trace(inCsvFile, [540 + dist_x, 540 + dist_y])


if __name__ == '__main__':
    main()
