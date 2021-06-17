#! /home/kmint/anaconda3/bin/python
from datetime import datetime
from pathlib import Path
import shutil
import argparse
import numpy as np
import json 

def get_last_exp(d_all_exps="../runs/apply"):
    d_all_exps = Path(d_all_exps)
    last_exp = None
    exps = list(d_all_exps.glob("exp*"))
    m = -1
    for exp in exps:
        if exp.name != 'exp':
            value = int(exp.name.split('exp')[1])
            m = value if value > m else m
    if m == -1:
        last_exp = d_all_exps / "exp1"
    else:
        last_exp = d_all_exps / f"exp{str(m)}"
    return last_exp


def main(opt):
    t = datetime.now().strftime("%m%d-%H%M")
    out_dir = Path.cwd() / t
    out_dir.mkdir()
    shutil.copytree("empty_submit", out_dir / "upload")
    shutil.copytree(str(opt.exp_d), out_dir / opt.exp_d.name)
    
    print(f"Packing {opt.exp_d} to {out_dir}")
    params_path = [p for p in out_dir.rglob("params*")][0]
    with params_path.open("r") as f:
        js = json.load(f)
    upload_dir = out_dir / "upload"
    upload_dir.mkdir(exist_ok=True)
    for p in out_dir.glob("exp*/labels/*"):
        shutil.copy2(p, upload_dir)
    stride = input("stride of the dataset (default 0): ")
    extra = input("Is extra set used, 1 for true, 0 for false (default 0): ")
    box_i = input("box size for i image (default 15): ")
    box_d = input("box size for d image (default 20): ")
    ishape = input("ishape (default '1920,2560'): ")
    dshape = input("dshape (default '1920,2560'): ")
    stride = 0 if stride == "" else int(stride)
    box_i = 15 if box_i == "" else int(box_i)
    box_d = 20 if box_d == "" else int(box_d)
    ishape = '1920,2560' if ishape == '' else ishape
    dshape = '1920,2560' if dshape == '' else dshape
    print("Compressing...")
    if len(list(out_dir.glob("exp*/labels/*.csv"))) > 90:
        dataset = 'all'
    else:
        dataset = 'public'
    zip_name_splits = [
        str(js["img_size"]),
        str(f"data-{dataset}"),
        str(f"extra-{extra}"),
        str(f"conf{js['i_conf_thres']}:{js['d_conf_thres']}"),
        str(f"shape{ishape}:{dshape}"),
        str(f"iou{js['iou_thres']}"),
        str(f"border{js['border']}"),
        str(f"stride{stride}"),
        str(f"box{box_i}:{box_d}"),
    ]
    out_name = out_dir / "_".join(zip_name_splits)
    shutil.make_archive(out_name, 'zip', upload_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--exp_d', default=get_last_exp(), help=f"Directory of exp (default: {get_last_exp()})")
    opt= parser.parse_args()
    opt.exp_d = Path(opt.exp_d)
    main(opt)


