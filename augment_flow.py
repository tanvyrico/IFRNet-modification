import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import read


def random_resize(flow):
    flow = cv2.resize(flow, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
    return flow


def random_crop(img0, flow,x ,y,crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    # x = np.random.randint(0, ih-h+1)
    # y = np.random.randint(0, iw-w+1)
    flow = flow[x:x+h, y:y+w, :]
    return flow


# def random_reverse_channel(img0, imgt, img1, flow, p=0.5):
#     if random.uniform(0, 1) < p:
#         img0 = img0[:, :, ::-1]
#         imgt = imgt[:, :, ::-1]
#         img1 = img1[:, :, ::-1]
#     return img0, imgt, img1, flow


def random_vertical_flip( flow, p=0.3):
    flow = flow[::-1]
    flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
    return flow


def random_horizontal_flip( flow, p=0.5):

    flow = flow[:, ::-1]
    flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
    return  flow


def random_rotate(flow, p=0.05):
 
    flow = flow.transpose((1, 0, 2))
    flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
    return  flow


def random_reverse_time(flow, p=0.5):
    flow = np.concatenate((flow[:, :, 2:4], flow[:, :, 0:2]), 2)
    return flow


