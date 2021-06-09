import numpy as np
from pathlib import Path


def loadcsv(path_or_file):
    return np.loadtxt(str(path_or_file), ndmin=2, delimiter=",", dtype=int)

