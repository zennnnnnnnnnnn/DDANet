from models_DDA.LACNet_DDA import LACNET

from common.evaluate import analysis_accu
from common.for_train import *

# data_path = "D:/MyData/pythonData/big_directory/test_wv3_OrigScale_multiExm1.h5"
data_path = "D:/MyData/pythonData/big_directory/NY1_WV3_Data/NY1_WV3_RR.mat"
# model_path = "Weights_MSDCNN_DDA/180.pth"
model_path = "E:/新建文件夹/CV/start!/PanHDNet_result/5-6——LACNet_DDA/600.pth"
# output_name = '100-DDANet.mat'
output_name = "LACDDA.mat"

model = LACNET()

import scipy.io as sio
import torch
from torchsummary import summary

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def test(model, is_cuda=True):
    pan, ms, lms, _ = get_data(data_path, is_cuda=is_cuda, data_mode="full")
    num = 4
    # pan = pan[num:num+1, ...]
    # ms = ms[num:num+1, ...]
    # lms = lms[num:num+1, ...]
    # gt = gt[num:num+1, ...]

    if is_cuda:
        model = model.cuda()

    load_result = model.load_state_dict(torch.load(model_path))
    print("loading weight from ", data_path, " result:", load_result)
    model.eval()

    output = model(lms, pan)

    # metrics = []
    # for i in range((gt.shape[0])):
    #     m = analysis_accu(gt[i].permute(1, 2, 0), output[i].permute(1, 2, 0).detach(), ratio=4, flag_cut_bounds=True,
    #                       dim_cut=21)
    #     print(m)
    #     metrics.append(list(m.values()))
    # mean_metrics = torch.tensor(metrics).mean(dim=0)
    # std_metrics = torch.tensor(metrics).std(dim=0)
    #
    # print(mean_metrics)  # sam ergas psnr
    # print(std_metrics)
    output = output.permute(0, 2, 3, 1)
    print(output[0].shape)
    if output_name:
        sio.savemat("result/" + output_name, {'output': output[0].cpu().detach().numpy()})


if __name__ == "__main__":
    # ########### test result #############
    test(model)

    # ########### get params size #############
    # summary(model, [(8, 64, 64), (1, 256, 256)])
