import csv
import numpy as np
import random

from PIL import Image, ImageDraw, ImageFilter

from params import PIC_NUM

# Constants
COL_RESIZE = [855./1675., 290./581., 802./1612.]
ROW_RESIZE = [859./1675., 292./581., 822./1612.]
RAD_RESIZE = [774./1675., 270./581., 740./1612.]


def main():
    with open("train_set.csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(PIC_NUM):
            pic_size = random.randint(60, 256)

            rand_col = random.randint(-(pic_size//2) + 5, 256 - (pic_size//2) - 5)
            rand_row = random.randint(-(pic_size//2) + 10, 256 - (pic_size//2) - 10)

            BG = random.randint(1, 7)
            bg = Image.open(rf'my_datasets/targets/bg_{BG}.jpg').convert("L")
            bg = bg.resize((256, 256))

            FG = random.randint(1, 3)
            fg = Image.open(rf'my_datasets/targets/fg_{FG}.jpg').convert("L")
            fg = fg.resize((pic_size, pic_size))

            new_col = round(COL_RESIZE[FG - 1] * pic_size) + rand_col
            new_row = round(ROW_RESIZE[FG - 1] * pic_size) + rand_row
            new_rad = round(RAD_RESIZE[FG - 1] * pic_size)

            bg.paste(fg, (rand_col, rand_row))
            bg.save(rf'my_datasets/train_pics/{i}.png', quality=100)

            bgnpy = np.array(bg, dtype=np.float64)
            np.save(rf'my_datasets/train/{i}.npy', bgnpy)
            write(outFile, [rf'my_datasets/train/{i}.npy', new_row, new_col, new_rad])


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    main()
