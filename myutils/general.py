import numpy as np
from pathlib import Path


def loadcsv(path_or_file):
    return np.loadtxt(str(path_or_file), ndmin=2, delimiter=",", dtype=int)


def get_max_exp_path(path):
    d = Path(path)
    out = "exp"
    m = 0
    if Path.is_dir(d):
        arr = [-1]
        for p in d.glob('exp*'):
            if p.stem == 'exp':
                arr.append('-2')
                continue
            arr.append(str(p.stem).split('exp')[1])
        arr = np.array(arr, dtype=np.int)
        if len(arr) == 2 and arr.max() == -1:
            return d / "exp"
        m = arr.max()
    return d / (out + str(m))