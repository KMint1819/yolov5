from pathlib import Path
import argparse
import numpy as np
from cv2 import cv2
from tqdm import tqdm

IMG_TYPES = [".jpg", ".png"]
def main(opt):
    if "out" in opt:
        opt.out.mkdir(parents=True)
    total = len([p for p in opt.dir.iterdir() if p.suffix.lower() in IMG_TYPES])
    with tqdm(opt.dir.iterdir(), total=total) as itemlist:
        for img_p in itemlist:
            if img_p.suffix.lower() not in IMG_TYPES:
                continue
            label_p = opt.labels / f"{img_p.stem}.csv"
            labels = np.loadtxt(label_p, dtype=int, delimiter=",", ndmin=2)
            img = cv2.imread(str(img_p))
            for label in labels:
                img = cv2.circle(img, tuple(label.tolist()), 3, (0, 0, 255), 3)
            
            if "out" in opt:
                out_p = opt.out / img_p.name
                cv2.imwrite(str(out_p), img)
            else:
                img = cv2.resize(img, None, fx=0.3, fy=0.3)
                cv2.imshow("Frame", img)
                if cv2.waitKey() == ord("q"):
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Directory of images")
    parser.add_argument("--labels", required=True, help="Directory of all labels")
    parser.add_argument("-o", '--out', type=str, help='If specified, images will be stored, else shown')
    opt = parser.parse_args()
    opt.dir = Path(opt.dir)
    opt.labels = Path(opt.labels)
    if "out" in opt:
        opt.out = Path(opt.out)
    main(opt)