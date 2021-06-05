from pathlib import Path
import argparse
import numpy as np
from cv2 import cv2
from myutils.general import loadcsv

IMG_TYPES = [".jpg", ".png"]

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', (720, 1080))

test_dir = Path.cwd() / "dataset/raw/test/"


def nothing(x):
    pass


cv2.namedWindow('trackbars')
cv2.createTrackbar('threshold', 'trackbars', 0, 100, nothing)
cv2.setTrackbarPos('threshold', 'trackbars', 30)


def get_thres():
    return cv2.getTrackbarPos('threshold', 'trackbars')

def main(opt):
    img_list = [p for p in opt.dir.iterdir() if p.suffix.lower() in IMG_TYPES]
    idx = 0
    n_imgs = len(img_list)
    with_label = False
    with_conf = False
    print("Controls:")
    print("     d: previous")
    print("     f: next")
    print("     s: with/without labels")
    print("     c: with/without confidence(only enabled when labels are enabled)")
    print("     q: quit")
    while True:
        img = cv2.imread(str(img_list[idx]))

        if with_label:
            label_p = opt.labels / f"{img_list[idx].stem}.csv"
            labels = loadcsv(str(label_p))
            if labels.shape[1] == 3:
                if with_conf:
                    labels = labels[labels[:, 0] > get_thres()]
                for conf, x, y in labels:
                    img = cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
                    if with_conf and conf > get_thres():
                        img = cv2.putText(img,
                                          str(conf),
                                          (x, y - 3),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.45,
                                          (153, 0, 204), 1, cv2.LINE_AA)
            elif labels.shape[1] == 2:
                if with_conf:
                    print("No confidence to show!")
                for x, y in labels:
                    img = cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

        cv2.imshow("frame", img)
        key = cv2.waitKey(10)
        if key == ord("f"):
            if idx < n_imgs - 1:
                idx = idx + 1
                print(str(img_list[idx]))
        elif key == ord("d"):
            if idx > 0:
                idx = idx - 1
                print(str(img_list[idx]))
        elif key == ord("s"):
            with_label = not with_label
        elif key == ord("c"):
            with_conf = not with_conf
        elif key == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", required=True,
                        help="Directory of images")
    parser.add_argument("-l", "--labels", required=True,
                        help="Directory of all labels")
    opt = parser.parse_args()
    if not (Path.cwd() / (opt.dir)).is_dir():
        if opt.dir == "public":
            opt.dir = test_dir / "test_public"
        elif opt.dir == "private":
            opt.dir = test_dir / "test_private"
        elif opt.dir == "all":
            opt.dir = test_dir / "test_all"
        else:
            print("Dir not found!")
            exit(-1)
        print("Redirect to", opt.dir)
    opt.dir = Path(opt.dir)
    opt.labels = Path(opt.labels)
    main(opt)
