import cv2
import numpy as np
import argparse
from numpy.lib.function_base import append
from sklearn.cluster import DBSCAN

TOLERANCE = 30


def filter_too_close(arr, tolerance=TOLERANCE, h_axis=None, v_axis=None, axis_expand=30):
    assert arr.ndim == 2, "arr should be a 2d-array"
    assert arr.shape[1] == 3, f"axis 1 should be (conf, x, y), given {arr.shape}"
    data = []
    ori_data = []
    if h_axis is not None and v_axis is not None:
        for conf, x, y in arr:
            appended = False
            for h in h_axis:
                if np.abs(h - y) < axis_expand:
                    data.append(np.array((conf, x, y), dtype=int))
                    appended = True
                    break
            if not appended:
                for v in v_axis:
                    if np.abs(x - v) < axis_expand:
                        data.append(np.array((conf, x, y), dtype=int))
                        appended = True
                        break
            if not appended:
                ori_data.append(np.array((conf, x, y), dtype=int))
    else:
        data = arr
    ori_data = np.array(ori_data, dtype=int)
    data = np.array(data, dtype=int)
    assert ori_data.shape[0] + data.shape[0] == arr.shape[0], "Assigned shape not matched!"
    # return ori_data, data
    cluster_labels = DBSCAN(eps=tolerance, min_samples=2).fit_predict(data[:, 1:])
    # print("Labels", cluster_labels)
    # Insert coordinates of the isolated points and midpoints of the clusters
    # to final points
    clustered_points = None
    for i in set(cluster_labels):
        idx = np.where(cluster_labels == i)[0]
        if i == -1:
            if clustered_points is None:
                clustered_points = data[idx]
            else:
                clustered_points = np.concatenate((clustered_points, data[idx]), axis=0, dtype=int)
        else:
            # print("data[idx]:", data[idx])
            mid_point = np.mean(data[idx], axis=0, dtype=int, keepdims=True)
            # print("mid point:", mid_point)
            if clustered_points is None:
                clustered_points = mid_point
            else:
                clustered_points = np.concatenate((clustered_points, mid_point), axis=0, dtype=int)
    # print(clustered_points)
    clustered_points = np.around(np.array(clustered_points, dtype=int)).astype(int)
    all_points = np.concatenate((ori_data, clustered_points), axis=0, dtype=int)
    return all_points
    # return data, clustered_points

if __name__ == "__main__":
    img = cv2.imread("/media/kmint/hdd/data/projects/yolov5/dataset/raw/test/test_sub/DSC080633.JPG")
    label = np.loadtxt("/media/kmint/hdd/data/projects/yolov5/uploads/05121337/uploads/DSC080633.csv", delimiter=",", ndmin=2, dtype=int)
    print(label.shape)
    label = np.insert(label, 0, np.zeros((1, label.shape[0]), dtype=int), axis=1)
    print(label.shape)
    scale_x = 3000 / 2560.
    scale_y = 2000 / 1920.
    axis_expand = 30
    h_axis = np.around(np.arange(640, 1920, 640) * scale_y).astype(int)
    v_axis = np.around(np.arange(640, 2560, 640) * scale_x).astype(int)
    filtered_points = filter_too_close(label, tolerance=TOLERANCE, h_axis=h_axis, v_axis=v_axis, axis_expand=axis_expand)
    for h in h_axis:
        img = cv2.line(img, (0, h), (img.shape[1], h), color=(0,0,0), thickness=4)
        img = cv2.line(img, (0, h - axis_expand), (img.shape[1], h - axis_expand), color=(200, 200, 200), thickness=2)
        img = cv2.line(img, (0, h + axis_expand), (img.shape[1], h + axis_expand), color=(200, 200, 200), thickness=2)
    for v in v_axis:
        img = cv2.line(img, (v, 0), (v, img.shape[0]), color=(0,0,0), thickness=2)
        img = cv2.line(img, (v - axis_expand, 0), (v - axis_expand, img.shape[0]), color=(200, 200, 200), thickness=2)
        img = cv2.line(img, (v + axis_expand, 0), (v + axis_expand, img.shape[0]), color=(200, 200, 200), thickness=2)


    for conf, x, y in label:
        img = cv2.circle(img, (x, y), 6, (0, 255, 0), -1)

    for conf, x, y in filtered_points:
        img = cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

    cv2.imwrite("frame.jpg", img)
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # cv2.imshow("frame", img)
    # cv2.waitKey()