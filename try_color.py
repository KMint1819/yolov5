import cv2
import numpy as np

def nothing(a):
    pass

cv2.namedWindow('bars')
cv2.createTrackbar("h", "bars", 0, 255, nothing)
cv2.createTrackbar("s", "bars", 0, 255, nothing)
cv2.createTrackbar("v", "bars", 0, 255, nothing)

img = np.zeros((100, 100, 3), dtype=np.uint8)
while True:
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.getTrackbarPos("h", "bars")
    s = cv2.getTrackbarPos("s", "bars")
    v = cv2.getTrackbarPos("v", "bars")
    # print(h, s, v)
    img[:, :, 0] = h
    img[:, :, 1] = s
    img[:, :, 2] = v
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow("frame", img)
    if cv2.waitKey(5) == ord("q"):
        break
h = cv2.getTrackbarPos("h", "bars")
s = cv2.getTrackbarPos("s", "bars")
v = cv2.getTrackbarPos("v", "bars")
print(h, s, v)