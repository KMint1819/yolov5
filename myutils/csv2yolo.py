from pathlib import Path
import numpy as np
import argparse
import shutil

# Only one side, so the width is expand * 2
expand_i = 10
expand_d = 20
def csv_to_yolo(opt):
    out_dir = Path.cwd() / "out_yolo"
    img_out_d = out_dir / "images"
    lbl_out_d = out_dir / "labels"
    img_out_d.mkdir(parents=True)
    lbl_out_d.mkdir(parents=True)
    for p in opt.path.iterdir():
        if p.suffix == ".csv":
            continue
        clas = 0 if str(p.stem).lower().startswith("i") else 1
        expand = expand_i if clas == 0 else expand_d
        label_p = p.parent / f"{p.stem}.csv"
        label = np.loadtxt(label_p, delimiter=",", ndmin=2)
        # print(label)
        shutil.copy2(p, img_out_d)
        out_p = lbl_out_d / f"{p.stem}.txt"
        with out_p.open("w") as f:
            for i, (x, y) in enumerate(label):
                w = min(expand * 2, min(x + expand, 639) - max(x - expand, 0))
                h = min(expand * 2, min(y + expand, 639) - max(y - expand, 0))
                x /= 640.
                y /= 640.
                w /= 640.
                h /= 640.
                print(x, y, w, h)
                f.write(f"{np.round(clas)} {np.round(x, 7)} {np.round(y, 7)} {np.round(w, 7)} {np.round(h, 7)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to dir of images and csvs")
    opt = parser.parse_args()
    opt.path = Path(opt.path)
    csv_to_yolo(opt)