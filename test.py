import os
import numpy as np
import torch
from models.IFRNet_S import Model
from utils import read
from imageio import mimsave
from imageio import imwrite
import time


device = torch.device("cpu")  # force CPU
model = Model().to(device).eval() 

# model.load_state_dict(torch.load('./checkpoint/IFRNet_S/2025-08-12_03-42-12/IFRNet_S_best.pth'))

i = 1

source_path = './testing_data/input_frames'
save_path = './testing_data/output_frames'
file_type = 'png'

prev_img_np =  read(f'{source_path}/frame_{str(i).zfill(4)}.{file_type}')
prev_img = (torch.tensor(prev_img_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0)

imwrite(f'{save_path}/img_0001.{file_type}',prev_img_np)
total_time = 0
while True :
  try :
    i += 1 
    next_img_np =  read(f'{source_path}/frame_{str(i).zfill(4)}.{file_type}')
    next_img = (torch.tensor(next_img_np .transpose(2, 0, 1)).float() / 255.0).unsqueeze(0)

    embt = torch.tensor(1/2).view(1, 1, 1, 1).float()

    start = time.perf_counter()

    imgt_pred = model.inference(prev_img, next_img, embt)

    end = time.perf_counter()

    print(f"Elapsed time: {end - start:.4f} seconds")

    total_time += end - start

    imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    imwrite(f'{save_path}/img_{str((i-1)*2).zfill(4)}.{file_type}',imgt_pred_np)

    imwrite(f'{save_path}/img_{str((i-1)*2 + 1).zfill(4)}.{file_type}',next_img_np)

    prev_img = next_img

    print(f'i_th : {i}')

  except :
    print(f'average time : {total_time/(i-1)}')
  
    break

