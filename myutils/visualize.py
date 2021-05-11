import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

def visualize(opt):
    out_dir = Path.cwd() / "out_visualize"
    out_dir.mkdir()
    total = len(list(opt.path.iterdir()))
    for img_p in tqdm(opt.path.iterdir(), total=total):
        lbl_p = Path(str(img_p).replace("images", "labels"))
        lbl_p = lbl_p.parent / f"{lbl_p.stem}.txt"
        labels = np.loadtxt(str(lbl_p), delimiter=" ", ndmin=2)
        if labels.size == 0:
            continue
        img = cv2.imread(str(img_p))
        labels[:, [1, 3]] *= img.shape[1]
        labels[:, [2, 4]] *= img.shape[0]
        labels[:, 1] = labels[:, 1] - labels[:, 3] / 2.
        labels[:, 2] = labels[:, 2] - labels[:, 4] / 2.
        labels = np.around(labels).astype(int)
        for _, x, y, w, h in labels:
            # print(x, y, w, h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        save_p = out_dir / img_p.name
        cv2.imwrite(str(save_p), img)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    opt = parser.parse_args()
    opt.path = Path(opt.path)
    visualize(opt)