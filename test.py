from models.DDANet import PanNet

from common.evaluate import analysis_accu
from common.for_train import *

# data_path = "D:/MyData/pythonData/big_directory/test_wv3_OrigScale_multiExm1.h5"
data_path = "D:/MyData/pythonData/big_directory/test_wv3_multiExm1.h5"
# model_path = "Weights_DDA/540.pth"
model_path = "E:/新建文件夹/CV/start!/PanHDNet_result/2-4——DDANet_mid/100.pth"
# output_name = '100-DDANet.mat'
output_name = None

model = PanNet()

import scipy.io as sio
import h5py
import numpy as np
import torch
import torch.nn as nn

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def test(model):
    pan, ms, lms, gt = get_data(data_path)

    load_result = model.load_state_dict(torch.load(model_path))
    print("loading weight from ", data_path, " result:", load_result)
    model.eval()

    import time
    time1 = time.time()
    output = model(ms, pan)
    print(time.time() - time1)

    metrics = []
    for i in range((gt.shape[0])):
        m = analysis_accu(gt[i].permute(1, 2, 0), output[i].permute(1, 2, 0).detach(), ratio=4, flag_cut_bounds=True,
                          dim_cut=21)
        metrics.append(list(m.values()))
    mean_metrics = torch.tensor(metrics).mean(dim=0)
    std_metrics = torch.tensor(metrics).std(dim=0)

    print(mean_metrics)  # sam ergas psnr
    print(std_metrics)

    if output_name:
        sio.savemat("Weights/" + output_name, {'output': output.detach().numpy()})


if __name__ == "__main__":
    test(model)
