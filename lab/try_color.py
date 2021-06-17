import cv2 as cv
import numpy as np
from pathlib import Path
# from matplotlib import pyplot as plt

cv.namedWindow('frame', cv.WINDOW_NORMAL)
cv.resizeWindow('frame', (720, 1080))

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(
        sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
    cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR, dst=img)  # no return needed

save_dir = Path.cwd() / "out"
save_dir.mkdir()
img = cv.imread('d4.jpg')

hgain = 0.015
sgain = 0.5
vgain = 0.7

for i in range(50):
    aug = img.copy()
    augment_hsv(aug, hgain, sgain, vgain)
    cv.imwrite(str(save_dir / f'{i}.jpg'), aug)