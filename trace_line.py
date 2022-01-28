import csv

from PIL import Image, ImageDraw


def trace(inCsvFile, row):
    writer = csv.writer(inCsvFile)
    writer.writerows([row])
    pass


def main():
    tracedxy = open("traced_video.csv").read().split()[:-1]
    header = tracedxy.pop(0)
    tp = r"my_datasets\untraced_target.jpg"
    lb = r"traced_target.jpg"
    lb_frame = r"traced_target.jpg"

    xydata = []
    for xy in tracedxy:
        x, y = xy.split(",")
        xydata.append((int(x), int(y)))

    img = Image.open(tp)
    draw = ImageDraw.Draw(img)
    scale = 1
    for i in range(len(xydata) - 1):
        lb_frame = rf"frame{i}.jpg"
        draw.line([xydata[i], xydata[i + 1]], fill="red", width=2)
        draw.ellipse([(1080 // 2 - 5, 1080 // 2 - 5), (1080 // 2 + 5, 1080 // 2 + 5)], fill="blue", outline="blue")
        img.save(lb_frame)

    img.save(lb)
    pass


if __name__ == "__main__":
    main()
