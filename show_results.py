from pathlib import Path
import argparse
import numpy as np
from cv2 import cv2
from tqdm import tqdm
from utils.general import increment_path

IMG_TYPES = [".jpg", ".png"]


def main(opt):
    out_dir = Path.cwd() / "runs/show_results/exp"
    out_dir = increment_path(out_dir, mkdir=True)

    img_list = [p for p in opt.dir.iterdir() if p.suffix.lower() in IMG_TYPES]
    with tqdm(img_list) as itemlist:
        for img_p in itemlist:
            if img_p.suffix.lower() not in IMG_TYPES:
                continue
            label_p = opt.labels / f"{img_p.stem}.csv"
            labels = np.loadtxt(label_p, dtype=int, delimiter=",", ndmin=2)
            img = cv2.imread(str(img_p))
            for label in labels:
                img = cv2.circle(img, tuple(label.tolist()),
                                 4, (0, 0, 255), -1)
            cv2.imwrite(str(out_dir / img_p.name), img)
    print("Results saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory of images")
    parser.add_argument("--labels", required=True,
                        help="Directory of all labels")
    opt = parser.parse_args()
    opt.dir = Path(opt.dir)
    opt.labels = Path(opt.labels)
    main(opt)
