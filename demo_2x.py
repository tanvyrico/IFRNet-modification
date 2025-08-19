import os
import numpy as np
import torch
# from models.IFRNet_S_ori import Model
from models.IFRNet_S import Model
from utils import read
from imageio import mimsave, imwrite
import time



model = Model().cuda().eval()
# model.load_state_dict(torch.load('./checkpoint/IFRNet_S/2025-08-12_16-09-07/IFRNet_S_best.pth'))

# model.load_state_dict(torch.load('./checkpoints/IFRNet_small/IFRNet_S_Vimeo90k.pth'))


img0_np = read('./output_images/frame_0001.jpg')
img1_np = read('./output_images/frame_0003.jpg')

img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()


start = time.perf_counter()  # High-resolution timer
imgt_pred = model.inference(img0, img1, embt)
end = time.perf_counter()

print(f"Execution time: {end - start:.6f} seconds")

imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

images = [img0_np, imgt_pred_np, img1_np]
mimsave('./figures/out_2x.gif', images, fps=3)
imwrite('./figures/interpolated_frame.jpg', imgt_pred_np)