import os
import numpy as np
import torch
# from models.IFRNet import Model
from models.IFRNet_S import Model
from utils import read
from imageio import mimsave
from imageio import imwrite
import time


device = torch.device("cpu")  # force CPU
model = Model().to(device).eval() 



i = 1


prev_img_np =  read(f'./test_walking/60_fps/output_{str(i).zfill(4)}.png')
prev_img = (torch.tensor(prev_img_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0)

imwrite('./test_walking/120_fps_interpolated/img_0001.png',prev_img_np)
total_time = 0
while True :
  try :
    i += 1 
    next_img_np =  read(f'./test_walking/60_fps/output_{str(i).zfill(4)}.png')
    next_img = (torch.tensor(next_img_np .transpose(2, 0, 1)).float() / 255.0).unsqueeze(0)

    embt = torch.tensor(1/2).view(1, 1, 1, 1).float()

    start = time.perf_counter()

    imgt_pred = model.inference(prev_img, next_img, embt)

    end = time.perf_counter()

    print(f"Elapsed time: {end - start:.4f} seconds")

    total_time += end - start

    imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


    imwrite(f'./test_walking/120_fps_interpolated/img_{str((i-1)*2).zfill(4)}.png',imgt_pred_np)

    imwrite(f'./test_walking/120_fps_interpolated/img_{str((i-1)*2 + 1).zfill(4)}.png',next_img_np)

    prev_img = next_img

    print(f'i_th : {i}')

  except :
    print(f'average time : {total_time/(i-1)}')
  
    break

# images = [img0_np, imgt_pred_np, img1_np]



# imwrite('./interpolateframe.png',imgt_pred_np)

# mimsave('./figures/out_2x.gif', images, fps=3)
