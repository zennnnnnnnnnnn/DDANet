import scipy.io as sio
import h5py
import numpy as np
import torch
import torch.nn as nn

# from models.PanSDLNet import PanNet
from models.PanNet import PanNet
# from models.DDANet import PanNet

from UDL.pansharpening.common.evaluate import D_lambda, D_s, qnr


SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

data_path = "D:/MyData/pythonData/big_directory/test_wv3_OrigScale_multiExm1.h5"
# model_path = "Weights/200.pth"
model_path = "E:/新建文件夹/CV/start!/PanHDNet_result/0-3——PanNet终极版/420.pth"
output_name = '200-SDLNet.mat'

data = h5py.File(data_path, 'r')
print(data.keys())

PAN = data['pan']
MS = data['ms']

pan = PAN[...]/2047.0     # [:64, :64, ...]
pan = torch.from_numpy(pan).float()

ms = MS[...]/2047.0      # [:64, :64, ...]
ms = torch.from_numpy(ms).float()

print(ms.shape)
print(pan.shape)

model = PanNet()
a = model.load_state_dict(torch.load(model_path))
print(a)
model.eval()

import time
time1 = time.time()
output = model(ms, pan)
print(time.time() - time1)

print(output.shape)

metrics = []

for i in range((output.shape[0])):
    # m = D_lambda(output[i], ms[i])
    # m = D_s(output[i], ms[i], pan[i])
    m = qnr(output[i], ms[i], pan[i])
    metrics.append(m)

mean_metrics = torch.tensor(metrics).mean(dim=0)
std_metrics = torch.tensor(metrics).std(dim=0)

print(mean_metrics)  # Dlamda Ds qnr
print(std_metrics)

sio.savemat("Weights/" + output_name, {'output': output.detach().numpy()})

