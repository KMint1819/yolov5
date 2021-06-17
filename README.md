# rice
https://aidea-web.tw/topic/9c88c428-0aa7-480b-85e0-2d8fb2fcf3fc?focus=team

## Dataset format
- `Original`:
    - Some are 2000 x 3000, some are 1728 x 2304
    - Original csv data are labeled in x, y format
## Dataset preparation
    - split train/val manually
    - `yolov5/myutils/image_cutter.py`
    - `yolov5/myutils/csv2yolo.py`
## Try
- TODO
    - Add a green-only-channel(or only)
    - Try custom mean/variance for normalization
    - Model with big kernel(100 x 100) to find the patterns of the grid.
    - Cut without resizing
    - Larger bbox for d-img
    - Decrease weight of cls loss in total loss
    - yolo 1280
    - Train all
    - Use DBSCAN to filter extreme points

- Tried and beneficial
    - Merge the edge
        - Can be further improved by sliding window + nms
    - Decrease IOU threshold
    - Decrease conf threshold
    - Add hsv augmentation to detect rice in very bright/dark situations
        - Does not help since bright/dark situations won't be labeled in the ground truth

- Tried but doesn't help
    - Use different detectors accordingly
    - Stride training

## Learn
- 不要一開始就找完美解答
- 大model != 高準確度
- 