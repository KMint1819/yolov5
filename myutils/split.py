from pathlib import Path
import argparse
import numpy as np
import shutil

def split_dataset(opt):
    n_images = len(list((opt.path / "images").iterdir()))
    out_dir = opt.path.parent / "val"
    out_img_dir = out_dir / "images"
    out_lbl_dir = out_dir / "labels"
    out_img_dir.mkdir(parents=True)
    out_lbl_dir.mkdir(parents=True)

    all_images = list((opt.path / "images").iterdir())
    np.random.seed(0)
    np.random.shuffle(all_images)
    val_img_ps = all_images[:int((1-opt.ratio) * n_images)]
    for val_img_p in val_img_ps:
        label_p = Path(str(val_img_p).replace("jpg", "txt").replace("images", "labels"))
        # print(label_p)
        # print(val_img_p)
        shutil.move(str(val_img_p), str(out_img_dir))
        shutil.move(str(label_p), str(out_lbl_dir))
        # shutil.copy2(val_img_p, out_img_dir)
        # shutil.copy2(label_p, out_lbl_dir)
        # break
    # print(val_img_ps)
    # print(val_img_ps.shape)
    # for p in (opt.path / "images").iterdir():

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="Path to dir that contains images and csvs")
    parser.add_argument("-r", "--ratio", required=True, type=float, help="Number between 0 to 1")
    opt = parser.parse_args()
    opt.path = Path(opt.path)
    split_dataset(opt)