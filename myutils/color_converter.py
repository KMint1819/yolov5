import argparse
import numpy as np
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
from general import get_save_dir


def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

IMG_TYPES = ['.jpg', '.png']
def main(opt):
    save_dir = get_save_dir('../runs/color')
    n_items = len(list(opt.dir.iterdir()))
    with tqdm(opt.dir.iterdir(), total=n_items) as itemlist:
        for img_p in itemlist:
            if not img_p.suffix.lower() in IMG_TYPES:
                continue

            img = cv.imread(str(img_p))
            adjusted, alpha, beta = automatic_brightness_and_contrast(img)
            cv.imwrite(str(save_dir / img_p.name), adjusted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory to convert color')
    opt = parser.parse_args()
    opt.dir = Path(opt.dir)
    main(opt)