import os
import time
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM
from models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
import cv2
from deep_utils import Box, split_extension
import matplotlib.pyplot as plt

device = 'cuda'
input_size = (640, 640)
model = YOLOV5TorchObjectDetector('yolov5s.pt', device, img_size=input_size)

def get_cam_map(img_path):
    img = cv2.imread(img_path)
    saliency_method = YOLOV5GradCAM(model=model, layer_name='model_23_cv3_act', img_size=input_size)
    torch_img = model.preprocessing(img[..., ::-1])
    masks, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
    mask = masks[0][0][0].cpu()
    mask[mask > 0.8] = 1
    mask[mask != 1] = 0
    return mask,boxes

if __name__ == '__main__':
    mask ,info = get_cam_map('objects365.jpg')
    print(info[0])
