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
from PIL import Image, ImageDraw
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

    h, w = 2 * r + 1, 2 * r + 1
    for li, gi in zip(range(h), range(-r + y, r + y)):
        for lj, gj in zip(range(w), range(-r + x, r + x)):
            if img[li, lj]:
                rr.append(gi)
                cc.append(gj)
                val.append(img[li, lj])

    return np.array(rr), np.array(cc), np.array(val)


def draw_meta(img, row, col, rad):
    meta = Image.open(f'my_datasets/targets_1_1/{PICTURE}').convert("L")
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


def draw_and_save_validation(i, img, params, detected, elite):
    img = float64_to_uint8(img)
    imageio.imsave("my_datasets/validate_pics/pics/" + str(i) + ".png", img[:, :])

    x, y, r = detected
    px, py, pr = params
    valid_pic = Image.open("my_datasets/validate_pics/pics/" + str(i) + ".png").convert("RGB")
    draw = ImageDraw.Draw(valid_pic)
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=(255, 0, 0))
    draw.ellipse([(x - 1, y - 1), (x + 1, y + 1)], outline=(255, 0, 0))
    draw.ellipse([(px, py), (px, py)], outline=(0, 255, 0))
    valid_pic.save("my_datasets/validate_pics/labeled/" + str(i) + "_labeled.png")
    if elite:
        valid_pic.save("my_datasets/validate_pics/best_labeled/" + str(i) + "_labeled.png")
        print("elite found!", i)

    if x == px and y == py:
        valid_pic.save("my_datasets/validate_pics/center_labeled/" + str(i) + "_labeled.png")
        print("center hit!", i)

    pass


def main():
    results = []
    model_name = f'saved_models/{MODEL_NAME}'
    best_count = 0
    for i in range(PIC_NUM_VALIDATE):
        params, img = noisy_circle(PICTURE_SIZE, PICTURE_SIZE // 2, NOISE_TEST)
        detected = find_circle(img, model_name)
        result = iou(params, detected)
        if (params[0], params[1]) == (detected[0], detected[1]):
            print(i, "real:", params, "predicted:", detected, "result:", result)
            draw_and_save_validation(i, img, params, detected, result > 0.98)
        if result > 0.9:
            print(i, "real:", params, "predicted:", detected, "result:", result)
            draw_and_save_validation(i, img, params, detected, result > 0.98)
            best_count += 1

        results.append(result)
    results = np.array(results)
    print("mean:", results.mean(), "veci od 0.9:", best_count)


if __name__ == '__main__':
    main()
