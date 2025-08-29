import os
import numpy as np
import torch
import torch.nn.functional as F
from liteflownet.run import estimate
from utils import read, write
import cv2
# from augment_flow_batch import random_resize_and_crop, random_vertical_flip, random_horizontal_flip,random_rotate,random_reverse_time
from datasets_no_flow import Vimeo90K_Train_Dataset_No_Flow, Vimeo90K_Test_Dataset
from datasets_test import Vimeo90K_Train_Dataset
from augment_flow_batch import get_flow,pred_flow_batch
# from datasets import Vimeo90K_Train_Dataset, Vimeo90K_Test_Dataset
from torch.utils.data import DataLoader




# , random_crop,random_vertical_flip, random_horizontal_flip,random_rotate,random_reverse_time


# set vimeo90k_dir with your Vimeo90K triplet dataset path, like '/.../vimeo_triplet'
# vimeo90k_dir = '/home/ltkong/Datasets/Vimeo90K/vimeo_triplet'

# vimeo90k_sequences_dir = os.path.join(vimeo90k_dir, 'sequences')
# vimeo90k_flow_dir = os.path.join(vimeo90k_dir, 'flow')

# if not os.path.exists(vimeo90k_flow_dir):
#     os.makedirs(vimeo90k_flow_dir)

# for sequences_path in sorted(os.listdir(vimeo90k_sequences_dir)):
#     vimeo90k_sequences_path_dir = os.path.join(vimeo90k_sequences_dir, sequences_path)
#     vimeo90k_flow_path_dir = os.path.join(vimeo90k_flow_dir, sequences_path)
#     if not os.path.exists(vimeo90k_flow_path_dir):
#         os.mkdir(vimeo90k_flow_path_dir)
        
#     for sequences_id in sorted(os.listdir(vimeo90k_sequences_path_dir)):
#         vimeo90k_flow_id_dir = os.path.join(vimeo90k_flow_path_dir, sequences_id)
#         if not os.path.exists(vimeo90k_flow_id_dir):
#             os.mkdir(vimeo90k_flow_id_dir)

# print('Built Flow Path')

def pred_flow(img1, img2):


    print(f'shape : {img1.shape}')
    img1 = img1.float().permute(2, 0, 1) / 255.0
    img2 = img2.float().permute(2, 0, 1) / 255.0

    flow = estimate(img1, img2)

    flow = flow.permute(1, 2, 0).cpu().numpy()
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

def pred_flow(img1, img2):
    img1 = torch.from_numpy(img1).float().permute(2, 0, 1) / 255.0
    img2 = torch.from_numpy(img2).float().permute(2, 0, 1) / 255.0

    # print(f'shape : {img1.shape}')
    # img1 = img1.float().permute(2, 0, 1) / 255.0
    # img2 = img2.float().permute(2, 0, 1) / 255.0

    flow = estimate(img1, img2)

    flow = flow.permute(1, 2, 0).cpu().numpy()
    return flow


# if __name__ == "__main__":
    # vimeo90k_dir = "C:\\Users\\enric\\Monash\\Y3S2\\model\\vimeo90k-test-and-train\\vimeo_triplet"
    # vimeo90k_sequences_dir = os.path.join(vimeo90k_dir, 'sequences/00001/0001')

    # img0_path = os.path.join(vimeo90k_sequences_dir, 'im1.png')
    # imgt_path = os.path.join(vimeo90k_sequences_dir, 'im2.png')
    # img1_path = os.path.join(vimeo90k_sequences_dir, 'im3.png')

    # img0 = read(img0_path)
    # imgt = read(imgt_path)
    # img1 = read(img1_path)
        
    # flow_t0 = pred_flow(imgt, img0)
    # flow_t1 = pred_flow(imgt, img1)

    # print(type(flow_t0))


    # ts = torch.from_numpy(flow_t0)


    # flow = cv2.resize(flow_t0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0

# if __name__ == "__main__":
#     dataset_train = Vimeo90K_Train_Dataset_No_Flow(dataset_dir= "C:\\Users\\enric\\Monash\\Y3S2\\model\\vimeo90k-test-and-train\\vimeo_triplet", augment=True)
#     dataloader_train = DataLoader(dataset_train, batch_size=6, num_workers=1, pin_memory=True, drop_last=True)

#     data = next(iter(dataloader_train))



#     dataset_train_actual = Vimeo90K_Train_Dataset(dataset_dir= "C:\\Users\\enric\\Monash\\Y3S2\\model\\vimeo90k-test-and-train\\vimeo_triplet", augment=True)
#     dataloader_train_actual = DataLoader(dataset_train_actual, batch_size=6, num_workers=1, pin_memory=True, drop_last=True)


#     first_act = next(iter(dataloader_train_actual))

#     img0, imgt, img1, flow_act, embt, dic, crop_x_y, flow_resize_crop, flow_reverse, flow_vertical,flow_horizontal,flow_rotate ,flow_reverse_time= first_act

#     img0_ ,imgt_ ,img1_ , aug0_ , augt_, aug1_, embt_ , _, _ = data


#     flow_t0 = pred_flow_batch(imgt_, img0_)
#     flow_t1 = pred_flow_batch(imgt_, img1_)

#     flow = np.concatenate((flow_t0, flow_t1), 3).astype(np.float64)
 
#     flow = random_resize_and_crop(flow,crop_x_y,dic,crop_size=(224,224))

#     flow = random_vertical_flip(flow,dic)

#     flow = random_horizontal_flip(flow,dic)    

#     flow = random_rotate(flow,dic)

#     flow = random_reverse_time(flow,dic)

#     flow = torch.from_numpy(flow.transpose((0, 3, 1, 2)).astype(np.float32))



#     with open("actual_tensor.txt", "w") as f:
#         for i, channel in enumerate(flow_reverse_time[2]):
#             f.write(f"Channel {i}:\n")
#             np_arr = channel.numpy()
#             for row in np_arr:
#                 f.write(" ".join(map(str, row)) + "\n")
#             f.write("\n")


#     with open("gotten_tensor.txt", "w") as f:
#         for i, channel in enumerate(flow[2]):
#             f.write(f"Channel {i}:\n")
#             np_arr = channel.numpy()
#             for row in np_arr:
#                 f.write(" ".join(map(str, row)) + "\n")
#             f.write("\n")


# if __name__ == "__main__":
#     dataset_train = Vimeo90K_Train_Dataset_No_Flow(dataset_dir= "C:\\Users\\enric\\Monash\\Y3S2\\model\\vimeo90k-test-and-train\\vimeo_triplet", augment=True)
#     dataloader_train = DataLoader(dataset_train, batch_size=6, num_workers=1, pin_memory=True, drop_last=True)

#     data = next(iter(dataloader_train))


#     dataset_train_actual = Vimeo90K_Train_Dataset(dataset_dir= "C:\\Users\\enric\\Monash\\Y3S2\\model\\vimeo90k-test-and-train\\vimeo_triplet", augment=True)
#     dataloader_train_actual = DataLoader(dataset_train_actual, batch_size=6, num_workers=1, pin_memory=True, drop_last=True)


#     data_act = next(iter(dataloader_train_actual))

#     img0_act, imgt_act, img1_act, flow_act, embt_act, dic_act, crop_x_y_act = data_act

#     img0,imgt,img1, aug0, augt, aug1, embt,dic, crop_x_y = data

#     start = time.time()
#     flow = get_flow(img0,imgt,img1,crop_x_y_act,dic_act)
#     end = time.time()

#     print(f"Elapsed: {end - start:.6f} seconds")


#     with open("actual_tensor.txt", "w") as f:
#         for i, channel in enumerate(flow_act[5]):
#             f.write(f"Channel {i}:\n")
#             np_arr = channel.numpy()
#             for row in np_arr:
#                 f.write(" ".join(map(str, row)) + "\n")
#             f.write("\n")


#     with open("gotten_tensor.txt", "w") as f:
#         for i, channel in enumerate(flow[5]):
#             f.write(f"Channel {i}:\n")
#             np_arr = channel.numpy()
#             for row in np_arr:
#                 f.write(" ".join(map(str, row)) + "\n")
#             f.write("\n")



