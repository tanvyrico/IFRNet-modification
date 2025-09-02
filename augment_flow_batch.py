import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import read
from liteflownet.run import estimate_batch,estimate



def random_resize_and_crop(flow,xy,dic,crop_size=(224,224)):
    resized_crop = []
    h, w = crop_size
    for i in range(flow.shape[0]):
      img = flow[i]
      if dic["random_resize"][i] == True :
        img = cv2.resize(flow[i], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0

      x = xy[0][i]
      y = xy[1][i]

      resized_crop.append(img[x:x+h, y:y+w, :])
    
    return np.stack(resized_crop)


        # flow = flow[::-1]
        # flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)


def random_vertical_flip(flow,dic ,p=0.3):
    """
    flow shape: (B, H, W, C)
    """
    for i,f in enumerate(flow):
      if dic["random_vertical_flip"][i] == True :
        f = f[::-1]
        f = np.concatenate((f[:, :, 0:1], -f[:, :, 1:2], f[:, :, 2:3], -f[:, :, 3:4]), 2)
        flow[i] = f 
    return flow


# def random_vertical_flip(img0, imgt, img1, flow, dic,p=0.3):
#     if random.uniform(0, 1) < p:
#         img0 = img0[::-1]
#         imgt = imgt[::-1]
#         img1 = img1[::-1]
#         flow = flow[::-1]
#         flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
#         dic["random_vertical_flip"] = True
#     else :
#         dic["random_vertical_flip"] = False
      


# def random_vertical_flip(flow, p=0.3):
#     """
#     flow shape: (B, H, W, C)
#     """
#     if np.random.rand() < p:
#         # Flip along the vertical axis (height)
#         flow = flow[:, ::-1, :, :]

#         # Negate the vertical components (2nd and 4th channels)
#         flow = np.concatenate((
#             flow[:, :, :, 0:1],        # keep x1
#             -flow[:, :, :, 1:2],       # flip y1
#             flow[:, :, :, 2:3],        # keep x2
#             -flow[:, :, :, 3:4]        # flip y2
#         ), axis=3)

#     return flow

def random_horizontal_flip(flow,dic ,p=0.3):
    """
    flow shape: (B, H, W, C)
    """
    for i,f in enumerate(flow):
      if dic["random_horizontal_flip"][i] == True :
        f = f[:, ::-1]
        f = np.concatenate((-f[:, :, 0:1], f[:, :, 1:2], -f[:, :, 2:3], f[:, :, 3:4]), 2)
        flow[i] = f 
    return flow

# def random_horizontal_flip( flow, p=0.5):
#     flow = flow[:, ::-1]
#     flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
#     return  flow

def random_rotate(flow,dic):
    """
    flow shape: (B, H, W, C)
    """
    for i,f in enumerate(flow):
      if dic["random_rotate"][i] == True :
        f = f.transpose((1, 0, 2))
        f= np.concatenate((f[:, :, 1:2], f[:, :, 0:1], f[:, :, 3:4], f[:, :, 2:3]), 2)
        flow[i] = f 
    return flow


# def random_rotate(flow, p=0.05):
#     flow = flow.transpose((1, 0, 2))
#     flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
#     return  flow

def random_reverse_time(flow,dic):
    """
    flow shape: (B, H, W, C)
    """
    
    for i,f in enumerate(flow):
      if dic["random_reverse_time"][i] == True :
        f = np.concatenate((f[:, :, 2:4], f[:, :, 0:2]), 2)
        flow[i] = f 
    return flow


def pred_flow_batch(img1, img2):
    B = img1.shape[0]

    flows = []
    for i in range(B):
        # take a single image [H, W, 3]
        im1 = img1[i].float().permute(2, 0, 1) / 255.0  # -> [3, H, W]
        im2 = img2[i].float().permute(2, 0, 1) / 255.0  # -> [3, H, W]

        # run the estimator (expects [3, H, W])
        flow = estimate(im1, im2)  # -> [2, H, W]

        # put it back into [H, W, 2]
        flow = flow.permute(1, 2, 0).cpu().numpy()
        flows.append(flow)

    flows = np.stack(flows, axis=0)  # -> [B, H, W, 2]

    return flows

def get_flow_no_augment(img0,imgt,img1):
   
  flow_t0 = estimate_batch(imgt,img0).numpy()
  flow_t1 = estimate_batch(imgt,img1).numpy()

  flow = np.concatenate((flow_t0, flow_t1), 3).astype(np.float64)

  flow = torch.from_numpy(flow.transpose((0, 3, 1, 2)).astype(np.float32))

  return flow
   
def get_flow(img0,imgt,img1,crop_x_y,dic):
    
  # flow_t0 = pred_flow_batch(imgt, img0)
  # flow_t1 = pred_flow_batch(imgt, img1)

  flow_t0 = estimate_batch(imgt,img0).numpy()
  flow_t1 = estimate_batch(imgt,img1).numpy()

  flow = np.concatenate((flow_t0, flow_t1), 3).astype(np.float64)

  flow = random_resize_and_crop(flow,crop_x_y,dic,crop_size=(224,224))

  flow = random_vertical_flip(flow,dic)

  flow = random_horizontal_flip(flow,dic)    

  flow = random_rotate(flow,dic)

  flow = random_reverse_time(flow,dic)

  flow = torch.from_numpy(flow.transpose((0, 3, 1, 2)).astype(np.float32))

  return flow


# def random_reverse_time(flow, p=0.5):
#     flow = np.concatenate((flow[:, :, 2:4], flow[:, :, 0:2]), 2)
#     return flow


