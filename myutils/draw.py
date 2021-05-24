import cv2
import numpy as np


def draw_grid(img, v_grid_starts, h_grid_starts):
    for v_grid_start in v_grid_starts:
        img = cv2.line(img, (v_grid_start, 0),
                       (v_grid_start, img.shape[0] - 1), (0, 0, 0), 2)
    for h_grid_start in h_grid_starts:
        img = cv2.line(img, (0, h_grid_start),
                       (img.shape[1] - 1, h_grid_start), (0, 0, 0), 2)
    return img


def draw_border(img, border_width):
    xy = []
    xy.append(np.array((border_width, border_width), dtype=int))
    xy.append(
        np.array((img.shape[1] - border_width - 1, border_width), dtype=int))
    xy.append(np.array(
        (img.shape[1] - border_width - 1, img.shape[0] - border_width - 1), dtype=int))
    xy.append(
        np.array((border_width, img.shape[0] - border_width - 1), dtype=int))
    xy.append(np.array((border_width, border_width), dtype=int))
    xy = np.array(xy, dtype=int)

    for i in range(4):
        img = cv2.line(img, (xy[i, 0], xy[i, 1]), (xy[i + 1, 0], xy[i + 1, 1]),
                       color=(0, 0, 0), thickness=2)
    return img