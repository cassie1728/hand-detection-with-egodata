# -*- coding:utf-8 -*-
# nora
# 2020-11-24
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import math
import os
import codecs
from glob import glob

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def init_model(model_path, device):
    # Load model
    #model = torch.load(model_path, map_location={'cuda:1':'cuda:0'})
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    return model



def yolov5_detect(model, img_ori, img_size, device):
    img = letterbox(img_ori, img_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # img = torch.from_numpy(img)
    # device = select_device('0')
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=True)
        det = pred[0]
        
        # Rescale boxes from img_size to im0 size此处为转换后的结果
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_ori.shape).round()
        # print('det2', det)
    return det



if __name__ == "__main__":

    # 模型配置文件路径
    model_path = './runs/train/exp4/weights/best.pt'
    # 所有label
    classes = ['hand']
    # 自动标注的类别
    select = ['hand']
    # 获取图片列表
    img_dir = '/home/psdz/Desktop/img/'
    # filelist = glob('/home/zhu/project/detection/pjreddie/darknet-master/ct_train4/test/*.jpg')
    # xml标注文件保存文件夹路径
    save_xml_path = '/home/psdz/Desktop/save_img/'
    if not os.path.exists(save_xml_path):
        os.makedirs(save_xml_path)
    #  模型初始化
    img_size = 960  #inference size (pixels) 必须是32的倍数
    conf_thres = 0.5  #object confidence threshold
    iou_thres = 0.5  #IOU threshold for NMS
    device1 = '1'  # 0,1,2,3 or cpu
    # 注意：在服务器上训练且未完成下载下来的中间模型，需要修改对应的显卡号
    device2 =torch.device('cuda:0')
    #device2 =  {'cuda:4':'cuda:1'}  #
    augment = 'store_true'
    agnostic_nms = 'store_true'
    
    device1 = select_device(device1)
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half =  False

    model = init_model(model_path, device2)

    for file in os.listdir(img_dir):
        print(file)
        file_name = file.split('/')[-1]
        file_path = os.path.join(img_dir, file_name)
        ori_image = cv2.imread(file_path)
        height, width, depth = ori_image.shape
        # 检测结果
        pred = yolov5_detect(model, ori_image, img_size, device1)
        # tensor 转 list
        print(type(pred))
        if pred is None:
            continue
        pred = pred.tolist()

        if pred is not None:
            save_name = file_name.split('/')[-1].split('.')[0]
            with codecs.open(save_xml_path + save_name + '.xml', 'w', 'utf-8') as xml:
                xml.write('<annotation>\n')
                xml.write('\t<filename>' + save_name + '.jpg' + '</filename>\n')
                xml.write('\t<size>\n')
                xml.write('\t\t<width>'+ str(width) + '</width>\n')
                xml.write('\t\t<height>'+ str(height) + '</height>\n')
                xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
                # xml.write('\t\t<segmented>0</segmented>\n')
                xml.write('\t</size>\n')

                for det_ in pred:
                    class_id = det_[5]
                    bbox = det_[0:4]
                    label_index = int(class_id)
                    label_en = classes[label_index]
                    if label_en not in select:
                        continue
                    name = label_en
                    xml.write('\t<object>\n')
                    xml.write('\t\t<name>' + name + '</name>\n')
                    xml.write('\t\t<pose>Unspecified</pose>\n')
                    xml.write('\t\t<truncated>1</truncated>\n')
                    xml.write('\t\t<difficult>0</difficult>\n')
                    xml.write('\t\t<bndbox>\n')

                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    
                    xml.write('\t\t\t<xmin>' + str(x1) + '</xmin>\n')
                    xml.write('\t\t\t<ymin>' + str(y1) + '</ymin>\n')
                    xml.write('\t\t\t<xmax>' + str(x2) + '</xmax>\n')
                    xml.write('\t\t\t<ymax>' + str(y2) + '</ymax>\n')
                    xml.write('\t\t</bndbox>\n')
                    xml.write('\t</object>\n')

                xml.write('</annotation>')
