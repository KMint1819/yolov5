from pathlib import Path
import argparse
import numpy as np
from cv2 import cv2
from myutils.general import loadcsv

IMG_TYPES = [".jpg", ".png"]


def main(opt):
    img_list = [p for p in opt.dir.iterdir() if p.suffix.lower() in IMG_TYPES]
    idx = 0
    n_imgs = len(img_list)
    with_label = False
    while True:
        img = cv2.imread(str(img_list[idx]))
        if with_label:
            label_p = opt.labels / f"{img_list[idx].stem}.csv"
            labels = loadcsv(str(label_p))
            for x, y in labels:
                img = cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        cv2.imshow("frame", img)
        key = cv2.waitKey()
        if key == ord("k"):
            idx = idx + 1 if idx < n_imgs - 1 else idx
        elif key == ord("j"):
            idx = idx - 1 if idx > 0 else idx
        elif key == ord("s"):
            with_label = not with_label
        elif key == ord("q"):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory of images")
    parser.add_argument("--labels", required=True,
                        help="Directory of all labels")
    opt = parser.parse_args()
    opt.dir = Path(opt.dir)
    opt.labels = Path(opt.labels)
    main(opt)
