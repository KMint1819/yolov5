from pathlib import Path
import argparse
import numpy as np
from cv2 import cv2
from myutils.general import loadcsv

IMG_TYPES = [".jpg", ".png"]
test_dir = Path.cwd() / "dataset/raw/test/"

def nothing(x):
    pass

# cv2.namedWindow('trackbars')
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', (720, 1080))
cv2.createTrackbar('threshold', 'frame', 0, 100, nothing)
cv2.setTrackbarPos('threshold', 'frame', 30)
cv2.createTrackbar('font size', 'frame', 0, 200, nothing)
cv2.setTrackbarPos('font size', 'frame', 50)
def get_thres():
    return cv2.getTrackbarPos('threshold', 'frame')
def get_font_size():
    return cv2.getTrackbarPos('font size', 'frame') / 100

def main(opt):
    img_list = [p for p in opt.dir.iterdir() if p.suffix.lower() in IMG_TYPES]
    idx = 0
    n_imgs = len(img_list)
    with_label = False
    print("Controls:")
    print("     a: previous")
    print("     d: next")
    print("     w: increase confidence threshold")
    print("     s: decrease confidence threshold")
    print("     e: with/without labels")
    print("     q: quit")
    while True:
        img = cv2.imread(str(img_list[idx]))
        if with_label:
            label_p = opt.labels / f"{img_list[idx].stem}.csv"
            labels = loadcsv(str(label_p))
            if labels.shape[1] == 3:
                labels = labels[labels[:, 0] > get_thres()]
                for conf, x, y in labels:
                    img = cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
                    img = cv2.putText(img,
                                        str(conf),
                                        (x, y - 3),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        get_font_size(),
                                        (153, 0, 204), 
                                        np.round(1 + (get_font_size() - 0.5) * 1.8).astype(int), 
                                        cv2.LINE_AA)
            elif labels.shape[1] == 2:
                for x, y in labels:
                    img = cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

        cv2.imshow("frame", img)
        key = cv2.waitKey(10)
        if key == ord("a"):
            if idx > 0:
                idx = idx - 1
                print(str(img_list[idx]))
        elif key == ord("d"):
            if idx < n_imgs - 1:
                idx = idx + 1
                print(str(img_list[idx]))
        elif key == ord("w") and get_thres() < 100:
            cv2.setTrackbarPos('threshold', 'frame', get_thres() + 1)    
        elif key == ord("s") and get_thres() > 0:
            cv2.setTrackbarPos('threshold', 'frame', get_thres() - 1)
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
