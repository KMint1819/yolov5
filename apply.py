# python apply.py \
# --weights all_stride0_box10-20.pt \
# --source dataset/raw/test/test_public \
# --conf-thres 0.25 \
# --iou-thres 0.45 \
# --grid


import argparse
import time
from pathlib import Path
import sys 

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadRiceImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from myutils.filter import filter_too_close


def apply(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    with (save_dir / f"{Path(opt.source).name}.txt").open("w") as f:
        f.write(str(opt))

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadRiceImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, imgs, imgs0, _ in dataset:
        path = Path(path)
        ori_img = cv2.imread(str(path))
        save_path = str(save_dir / path.name)
        txt_path = str(save_dir / "labels" / f"{path.stem}.csv")
        coords = []
        img_type = str(path.name)[0].lower()
        for r in range(imgs.shape[0]):
            for c in range(imgs.shape[1]):
                img = imgs[r, c]
                im0s = imgs0[r, c]
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for cl in det[:, -1].unique():
                            n = (det[:, -1] == cl).sum()  # detections per class
                            s += f"{n} {names[int(cl)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cl in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                x, y = xywh[:2]
                                x, y = x * im0.shape[1], y * im0.shape[0]
                                x += c * im0.shape[1]
                                y += r * im0.shape[0]
                                coords.append(np.array((conf.cpu().item() * 100, x, y)))

                            # if save_img or view_img:  # Add bbox to image
                            #     c = int(cls)  # integer class
                            #     label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                                # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)

        v_grid_starts, h_grid_starts = [], []
        if opt.grid:
            x = imgs[0, 0].shape[2]
            while x < ori_img.shape[1]:
                v_grid_starts.append(x)
                x += imgs[0, 0].shape[2]
            y = imgs[0, 0].shape[1]
            while y < ori_img.shape[0]:
                h_grid_starts.append(y)
                y += imgs[0, 0].shape[1]
        v_grid_starts, h_grid_starts = np.array(v_grid_starts, dtype=float), np.array(h_grid_starts, dtype=float) 
        # imgs[0, 0].shape is (c, h, w)
        scale_x = ori_img.shape[1] / (imgs.shape[1] * imgs[0, 0].shape[2])
        scale_y = ori_img.shape[0] / (imgs.shape[0] * imgs[0, 0].shape[1])
        coords = np.array(coords)
        coords[:, 1] *= scale_x
        coords[:, 2] *= scale_y
        coords = np.around(coords).astype(int)
        tol = opt.tol_i if img_type == "i" else opt.tol_d
        v_grid_starts *= scale_x
        h_grid_starts *= scale_y
        v_grid_starts, h_grid_starts = np.around(v_grid_starts).astype(int), np.around(h_grid_starts).astype(int)
        gt_path = path.parent / f"{path.stem}.csv"
        if img_type == "d":
            coords = filter_too_close(coords, tolerance=tol, h_axis=h_grid_starts, v_axis=v_grid_starts, axis_expand=opt.axis_expand)
        if save_txt:
            with open(txt_path, "w") as f:
                np.savetxt(f, coords[:, 1:], fmt="%d", delimiter=",")
        if save_img:
            if opt.grid:
                for v_grid_start in v_grid_starts:
                    ori_img = cv2.line(ori_img, (v_grid_start, 0), (v_grid_start, ori_img.shape[0] - 1), (0, 0, 0), 2)
                for h_grid_start in h_grid_starts:
                    ori_img = cv2.line(ori_img, (0, h_grid_start), (ori_img.shape[1] - 1, h_grid_start), (0, 0, 0), 2)
            if opt.with_gt:
                gts = np.loadtxt(gt_path, dtype=int, delimiter=",", ndmin=2)
                for x, y in gts:
                    ori_img = cv2.circle(ori_img, (x, y), 9, (255, 255, 255), 2)
            for conf, x, y in coords:
                if not opt.hide_conf:
                    # print(conf)
                    ori_img = cv2.putText(ori_img, f"{conf}%", (x, y - 3), 0, 1, (255, 255, 0), 2)
                ori_img = cv2.circle(ori_img, (x, y), 4, (255, 0, 0), -1)

            cv2.imwrite(save_path, ori_img)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/apply', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--with-gt', action="store_true", default=False, help='Whether to show the ground truth')
    parser.add_argument('--hide-labels', action='store_true', default=False, help='hide labels')
    parser.add_argument('--hide-conf', action='store_true', default=False, help='hide confidences')
    parser.add_argument('--grid', action='store_true', default=False, help='draw grid lines')
    parser.add_argument('--tol-i', default=25, type=int, help='tolerance of close points for i image(pixels)')
    parser.add_argument('--tol-d', default=40, type=int, help='tolerance of close points for d image(pixels)')
    parser.add_argument('--axis-expand', default=30, type=int, help='width of axis to merge')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                apply(opt=opt)
                strip_optimizer(opt.weights)
        else:
            apply(opt=opt)
