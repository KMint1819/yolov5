from pathlib import Path
import argparse
import numpy as np
import cv2 as cv
from utils.general import increment_path
from myutils.general import loadcsv

IMG_TYPES = [".jpg", ".png"]
save_dir = Path(increment_path("runs/label/exp", mkdir=True))

cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', (720, 1080))


ori_img = None
painted = None
labels = None
labeling = True
pt_front, pt_back = None, None
pressed = None
show_label = True


def repaint(img, labels):
    global show_label
    if show_label:
        for x, y in labels:
            img = cv.circle(img, (x, y), 4, (255, 0, 0), -1)
    return img


def mouse_clicked_callback(event, x, y, flags, params):
    def inside_rect(x, y, left_up, right_bot):
        return right_bot[0] > x > left_up[0] and right_bot[1] > y > left_up[1]
    global ori_img, painted, labels, pt_front, pt_back, labeling
    if event == cv.EVENT_LBUTTONDBLCLK:
        if labeling:
            labels = np.vstack((labels, (x, y)))
            print(x, y, "added to labels")
        else:
            if pt_front is None:
                pt_front = np.array((x, y), dtype=int)
            else:
                pt_back = np.array((x, y), dtype=int)
                pt1 = np.array((min(pt_front[0], pt_back[0]), min(
                    pt_front[1], pt_back[1])), dtype=int)
                pt2 = np.array((max(pt_front[0], pt_back[0]), max(
                    pt_front[1], pt_back[1])), dtype=int)
                n_old = labels.shape[0]
                xy_pts = labels[:, [0, 1]]
                labels = np.delete(labels, np.all(
                    ((xy_pts > pt1) & (xy_pts < pt2)), axis=1), axis=0)
                n_new = labels.shape[0]
                print(f"{n_old - n_new} labels from {pt1}, {pt2} removed. ")
                pt_front = None
        painted = repaint(ori_img.copy(), labels)
    if event == cv.EVENT_MOUSEMOVE and not pt_front is None and not labeling:
        pt_back = np.array((x, y), dtype=int)
        pt1 = (min(pt_front[0], pt_back[0]), min(pt_front[1], pt_back[1]))
        pt2 = (max(pt_front[0], pt_back[0]), max(pt_front[1], pt_back[1]))
        painted = cv.rectangle(ori_img.copy(), pt1, pt2, (0, 255, 0), 3)
        painted = repaint(painted, labels)


cv.setMouseCallback('frame', mouse_clicked_callback)


def main(opt):
    global ori_img, painted, labels, labeling, show_label, pt_front
    img_list = [p for p in opt.dir.iterdir() if p.suffix.lower() in IMG_TYPES]
    n_imgs = len(img_list)
    idx = 0
    ori_img = cv.imread(str(img_list[idx]))
    assert 'labels' in opt, 'Must be labeled before'
    label_p = opt.labels / f"{img_list[idx].stem}.csv"
    labels = loadcsv(label_p)
    painted = repaint(ori_img.copy(), labels)

    print("Controls:")
    print("     a: previous")
    print("     d: next")
    print("     c: with/without labels")
    print("     e: switch mode between labeling & erasing")
    print("     q: quit")
    while True:
        cv.imshow("frame", painted)
        key = cv.waitKey(10)
        if (key == ord("a") and idx > 0) or (key == ord("d") and idx < n_imgs - 1):
            np.savetxt(str(save_dir / label_p.name),
                       labels, fmt='%d', delimiter=',')
            if key == ord('a') and idx > 0:
                idx = idx - 1
            elif key == ord("d") and idx < n_imgs - 1:
                idx = idx + 1
            print(str(img_list[idx]))
            ori_img = cv.imread(str(img_list[idx]))
            label_p = opt.labels / f"{img_list[idx].stem}.csv"
            if (save_dir / label_p.name).is_file():
                label_p = save_dir / f"{img_list[idx].stem}.csv"
            labels = loadcsv(label_p)
            painted = repaint(ori_img.copy(), labels)
        elif key == ord("e"):
            labeling = not labeling
            s = "label mode" if labeling else "erase mode"
            pt_front = None
            print("Switching to", s)
        elif key == ord('c'):
            show_label = not show_label
            painted = repaint(ori_img.copy(), labels)
        elif key == ord("q"):
            np.savetxt(str(save_dir / label_p.name),
                       labels, fmt='%d', delimiter=',')
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir",
                        help="Directory of images")
    parser.add_argument("-l", "--labels",
                        help="Directory of all labels")
    opt = parser.parse_args()
    # opt.dir = Path.cwd() / "dataset/raw/test/test_public/d"
    # opt.labels = Path.cwd() / "uploads/0607-2324/upload"
    # test_dir = Path.cwd() / "dataset/raw/test"
    # if not (Path.cwd() / (opt.dir)).is_dir():
    #     if opt.dir == "public":
    #         opt.dir = test_dir / "test_public_all"
    #     elif opt.dir == "i":
    #         opt.dir = test_dir / 'test_public/i'
    #     elif opt.dir == "d":
    #         opt.dir = test_dir / 'test_public/d'
    #     elif opt.dir == "private":
    #         opt.dir = test_dir / "test_private"
    #     elif opt.dir == "all":
    #         opt.dir = test_dir / "test_all"
    #     else:
    #         print("Dir not found!")
    #         exit(-1)
    #     print("Redirect to", opt.dir)
    opt.dir = Path(opt.dir)
    opt.labels = Path(opt.labels)
    main(opt)
