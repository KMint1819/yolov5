import numpy as np

def loadcsv(path_or_file):
    return np.loadtxt(path_or_file, delimiter=",", dtype=int)