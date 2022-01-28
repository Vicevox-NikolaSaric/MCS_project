import csv
import numpy as np
import random

from PIL import Image, ImageEnhance, ImageFilter

from params import PIC_NUM

# Constants
COL_RESIZE = [1053./2071., 984./1956., 907./1802., 1080./2157.]
ROW_RESIZE = [1039./2071., 976./1956., 910./1802., 1073./2157.]
RAD_RESIZE = [995./2071., 960./1956., 872./1902., 1045./2157.]
SIZE = [2071., 1956., 1802., 2157.]

BG_WIDTH = 4608.
BG_HEIGHT = 2592.


def main():
    with open("train_set.csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(PIC_NUM):
            pic_size = random.randint(600, 2500)

            rand_col = random.randint(-(pic_size//2) + 5, 4608 - (pic_size//2) - 5)
            rand_row = random.randint(-(pic_size//2) + 10, 2592 - (pic_size//2) - 10)

            BG = random.randint(1, 5)
            bg = Image.open(rf'my_datasets/targets_16_9/bg_{BG}.jpg').convert("L")

            FG = random.randint(1, 4)
            fg = Image.open(rf'my_datasets/targets_16_9/fg_{FG}.jpg').convert("L")
            fg = fg.resize((pic_size, pic_size))

            new_col = COL_RESIZE[FG - 1] * pic_size + rand_col
            new_col = round(new_col * (256. / BG_WIDTH))

            new_row = ROW_RESIZE[FG - 1] * pic_size + rand_row
            new_row = round(new_row * (256. / BG_HEIGHT))

            new_rad = round(RAD_RESIZE[FG - 1] * pic_size * (256. / 2592.))

            bg.paste(fg, (rand_col, rand_row))

            # BLUR
            if random.random() <= 0.2:
                bg = bg.filter(ImageFilter.BoxBlur(random.randint(1, 5)))

            # DETAIL
            elif random.random() <= 0.2:
                bg = bg.filter(ImageFilter.DETAIL)

            # SHARPEN
            if random.random() <= 0.2:
                bg = bg.filter(ImageFilter.SHARPEN)

            # BRIGHTNESS
            if random.random() <= 0.2:
                factor = (random.random() / 3. - 0.33) + 1.
                enhancer = ImageEnhance.Brightness(bg)
                bg = enhancer.enhance(factor)

            # CONTRAST
            if random.random() <= 0.2:
                factor = (random.random() / 3. - 0.33) + 1.
                enhancer = ImageEnhance.Contrast(bg)
                bg = enhancer.enhance(factor)

            bg = bg.resize((256, 256))
            bg.save(rf'my_datasets/train_pics/{i}.png', quality=100)

            bgnpy = np.array(bg, dtype=np.float64)
            np.save(rf'my_datasets/train/{i}.npy', bgnpy)
            write(outFile, [rf'my_datasets/train/{i}.npy', new_row, new_col, new_rad])


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    main()
