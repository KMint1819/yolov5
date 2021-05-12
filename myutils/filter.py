import cv2
import numpy as np
import argparse
from sklearn.cluster import DBSCAN

TOLERANCE = 15


def filter_too_close(arr, tolerance=TOLERANCE):
    assert arr.ndim == 2, "arr should be a 2d-array"
    cluster = DBSCAN(eps=tolerance, min_samples=1).fit(arr)
    clas = []
    # Get the class number that occurs more than once
    mx = np.max(cluster.labels_)
    for i in range(mx):
        if (cluster.labels_ == i).sum() > 1:
            clas.append(i)

    # Insert coordinates of the isolated points and midpoints of the clusters
    # to final points
    final_points = []
    for i in set(cluster.labels_):
        c = np.where(cluster.labels_ == i)
        if len(c[0]) <= 1:
            final_points.append(arr[c[0][0], :])
        else:
            mid_point = np.mean(arr[c[0]], axis=0, dtype=int)
            final_points.append(mid_point)
    return np.array(final_points, dtype=int)
    


if __name__ == "__main__":
    arr = np.arange(6).reshape(3, 2)
    print(arr)