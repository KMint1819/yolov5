import cv2
import numpy as np
from pathlib import Path
from utils.general import increment_path

out_dir = Path.cwd() / "runs/test_color/exp"
out_dir = increment_path(out_dir, exist_ok=False)

# from utils.datasets import augment_hsv

img = cv2.imread("test.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_green = np.array([25, 4, 5])
upper_green = np.array([77, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
print(mask)
# res = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow('Input', img)
# cv2.imshow('Result', res)
# cv2.waitKey(0)