from pathlib import Path
import argparse
import numpy as np
import cv2 as cv
from myutils.general import loadcsv

IMG_TYPES = [".jpg", ".png"]
test_dir = Path.cwd() / "dataset/raw/test"



ori_img = None
painted = None
labels = None
def on_threshold_changed(x):
    global painted
    painted = repaint(ori_img.copy(), labels)

# cv.namedWindow('trackbars')
cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', (720, 1080))
cv.createTrackbar('threshold', 'frame', 0, 100, on_threshold_changed)
cv.setTrackbarPos('threshold', 'frame', 30)
cv.createTrackbar('font size', 'frame', 0, 200, on_threshold_changed)
cv.setTrackbarPos('font size', 'frame', 50)


def get_thres():
    return cv.getTrackbarPos('threshold', 'frame')


def get_font_size():
    return cv.getTrackbarPos('font size', 'frame') / 100


def repaint(img, labels):
    painted = img
    if labels.shape[1] == 3:
        print("threshold:", get_thres())
        labels = labels[labels[:, 0] > get_thres()]
        for conf, x, y in labels:
            painted = cv.circle(painted, (x, y), 4, (255, 0, 0), -1)
            painted = cv.putText(painted,
                                    str(conf),
                                    (x, y - 3),
                                    cv.FONT_HERSHEY_SIMPLEX,
                                    get_font_size(),
                                    (153, 0, 204),
                                    np.round(
                                        1 + (get_font_size() - 0.5) * 1.8).astype(int),
                                    cv.LINE_AA)
    elif labels.shape[1] == 2:
        for x, y in labels:
            painted = cv.circle(painted, (x, y), 4, (255, 0, 0), -1)
    return painted



def main(opt):
    global ori_img, painted, labels
    img_list = [p for p in opt.dir.iterdir() if p.suffix.lower() in IMG_TYPES]
    n_imgs = len(img_list)
    idx = 0
    ori_img = cv.imread(str(img_list[idx]))
    label_p = opt.labels / f"{img_list[idx].stem}.csv"
    labels = loadcsv(label_p)
    with_label = False
    painted = repaint(ori_img.copy(), labels)
    print("Controls:")
    print("     a: previous")
    print("     d: next")
    print("     w: increase confidence threshold")
    print("     s: decrease confidence threshold")
    print("     e: with/without labels")
    print("     q: quit")
    while True:
        if with_label:
            cv.imshow("frame", painted)
        else:
            cv.imshow("frame", ori_img)
        key = cv.waitKey(20)
        if (key == ord("a") and idx > 0) or (key == ord("d") and idx < n_imgs - 1):
            if key == ord('a') and idx > 0:
                idx = idx - 1
            elif key == ord("d") and idx < n_imgs - 1:
                idx = idx + 1
            print(str(img_list[idx]))
            label_p = opt.labels / f"{img_list[idx].stem}.csv"
            labels = loadcsv(label_p)
            ori_img = cv.imread(str(img_list[idx]))
            painted = repaint(ori_img.copy(), labels)
        elif key == ord("w") and get_thres() < 100:
            cv.setTrackbarPos('threshold', 'frame', get_thres() + 1)
            painted = repaint(ori_img.copy(), labels)
        elif key == ord("s") and get_thres() > 0:
            cv.setTrackbarPos('threshold', 'frame', get_thres() - 1)
            painted = repaint(ori_img.copy(), labels)
        elif key == ord("e"):
            with_label = not with_label
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
            opt.dir = test_dir / "test_public_all"
        elif opt.dir == "i":
            opt.dir = test_dir / 'test_public/i'
        elif opt.dir == "d":
            opt.dir = test_dir / 'test_public/d'
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
