import cv2
import numpy as np
from pathlib import Path
import argparse

# IMG_SHAPE = (256, 256)
IMG_SHAPE = (640, 640)
# D_IMG_SHAPE = (2560, 3200)  # (2000, 3000)
# I_IMG_SHAPE = (2560, 3200)  # (1728, 2304)


def imageCut(p, dshape, ishape, with_label=False, with_circle=False, stride=None):
    """Cut the image and label accordingly

    Args:
        p (str or Path): Path to image
        l (int): Length of image size
        with_label (bool, optional): If cut the label or not. Defaults to False.
    """
    if with_circle:
        assert with_label, "Cannot draw circles without labels!"
    if stride:
        assert IMG_SHAPE[0] % stride == 0 and IMG_SHAPE[1] % stride == 0, "Stride must be a factor of IMG_SHAPE."

    img = cv2.imread(str(p))
    if img.shape[0] == 2000:
        scale_y = dshape[0] / img.shape[0]
        scale_x = dshape[1] / img.shape[1]
        img = cv2.resize(img, (dshape[1], dshape[0]))
    else:
        scale_y = ishape[0] / img.shape[0]
        scale_x = ishape[1] / img.shape[1]
        img = cv2.resize(img, (ishape[1], ishape[0]))
    shape = IMG_SHAPE

    if with_label:
        p = Path(p)
        idx = p.stem
        lbl_path = p.parent / f"{idx}.csv"
        label = np.loadtxt(lbl_path, delimiter=",", dtype=int)
        # Scale the x, y label correspondly
        label[:, 0] = (label[:, 0] * scale_x).astype(int)
        label[:, 1] = (label[:, 1] * scale_y).astype(int)

    # cv2.imshow("frmae", img)
    # cv2.waitKey()
    print(img.shape)
    print(shape)
    assert img.shape[0] % shape[0] == 0 and img.shape[1] % shape[1] == 0
    if stride:
        n_rows = np.around((img.shape[0] - shape[0]) / stride + 1).astype(int)
        n_cols = np.around((img.shape[1] - shape[1]) / stride + 1).astype(int)
    else:
        n_rows = np.around(img.shape[0] / shape[0]).astype(int)
        n_cols = np.around(img.shape[1] / shape[1]).astype(int)

    imgs = np.empty((n_rows, n_cols), dtype=object)
    labels = np.empty((n_rows, n_cols), dtype=object)
    stride_x = stride if stride else shape[1]
    stride_y = stride if stride else shape[0]
    for r in range(0, n_rows):
        for c in range(0, n_cols):
            r_idx = r * stride_y
            c_idx = c * stride_x
            imgs[r, c] = img[r_idx:r_idx + shape[0], c_idx:c_idx + shape[1]]
            col_label = []
            if with_label:
                for x, y in label:
                    if y >= r_idx and y < r_idx + shape[0] and x >= c_idx and x < c_idx + shape[1]:
                        tmp_lbl = np.around(
                            np.array((x - c_idx, y - r_idx))).astype(int)
                        col_label.append(tmp_lbl)
                labels[r, c] = col_label

            if with_circle:
                for x, y in col_label:
                    imgs[r, c] = cv2.circle(
                        imgs[r, c], (x, y), 2, (0, 0, 255), 5)

    if with_label:
        return img, imgs, labels
    return img, imgs


def cutDatasetAndSave(p: str, output_dir: str, dshape, ishape, stride=None, with_label=False, with_circle=False):
    p = Path(p)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True)

    for img_p in p.iterdir():
        if img_p.suffix.lower() == ".jpg":
            print("Cutting", img_p)
            if with_label:
                _, small_imgs, small_labels = imageCut(
                    img_p, stride=stride, with_label=True, with_circle=with_circle, dshape=dshape, ishape=ishape)
                for i in range(small_imgs.shape[0]):
                    for j in range(small_imgs.shape[1]):
                        small_img = small_imgs[i, j]
                        small_label = small_labels[i, j]
                        pre = f"{img_p.stem}_{i}-{j}"
                        cv2.imwrite(str(out_dir / f"{pre}.jpg"), small_img)
                        np.savetxt(
                            str(out_dir / f"{pre}.csv"), small_label, delimiter=",", fmt="%d")
            else:
                _, small_imgs = imageCut(
                    img_p, stride=stride, with_label=False, with_circle=with_circle, dshape=dshape, ishape=ishape)
                for i in range(small_imgs.shape[0]):
                    for j in range(small_imgs.shape[1]):
                        small_img = small_imgs[i, j]
                        pre = f"{img_p.stem}_{i}-{j}"
                        cv2.imwrite(str(out_dir / f"{pre}.jpg"), small_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        default=None, help="Path to original directory")

    parser.add_argument("-o", "--output", type=str,
                        default=None, help="Path to output directory")

    parser.add_argument("-s", "--stride", type=int,
                        default=None, help="Number of strides")

    parser.add_argument("-label", "--with_label", action="store_true", default=False,
                        help="Add this flag if you want to cut the label file additionally")

    parser.add_argument("-circle", "--with_circle", action="store_true", default=False,
                        help="Add this flag if you want to draw the circles according to the labels")

    parser.add_argument("-dsh", "--dshape", type=str, default='1920,2560',
                        help="shape for d image to be resized to. format is '1920,2560'")

    parser.add_argument("-ish", "--ishape", type=str, default='1920,2560',
                        help="shape for i image to be resized to. format is '1920,2560'")

    opt = parser.parse_args()
    spt = opt.dshape.split(',')
    opt.dshape = (int(spt[0]), int(spt[1]))
    spt = opt.ishape.split(',')
    opt.ishape = (int(spt[0]), int(spt[1]))

    cutDatasetAndSave(opt.input, opt.output, stride=opt.stride,
                      with_label=opt.with_label, with_circle=opt.with_circle,
                      dshape=opt.dshape, ishape=opt.ishape)
