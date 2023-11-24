from models_DDA.PanNet_DDA import PanNet

from common.evaluate import analysis_accu
from common.for_train import *

data_path = "D:/MyData/pythonData/big_directory/test_wv3_multiExm1.h5"
model_path = "Weights_MSDCNN_DDA/180.pth"
# model_path = "E:/新建文件夹/CV/start!/PanHDNet_result/5-5——LACNet/200.pth"
# output_name = '100-DDANet.mat'
output_name = None

model = PanNet().cuda()

import scipy.io as sio
import torch
from torchsummary import summary

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def test(model):
    pan, ms, lms, gt = get_data(data_path, is_cuda=True)

    load_result = model.load_state_dict(torch.load(model_path))
    print("loading weight from ", data_path, " result:", load_result)
    model.eval()

    output = model(lms, pan)

    metrics = []
    for i in range((gt.shape[0])):
        m = analysis_accu(gt[i].permute(1, 2, 0), output[i].permute(1, 2, 0).detach(), ratio=4, flag_cut_bounds=True,
                          dim_cut=21)
        print(m)
        metrics.append(list(m.values()))
    mean_metrics = torch.tensor(metrics).mean(dim=0)
    std_metrics = torch.tensor(metrics).std(dim=0)

    print(mean_metrics)  # sam ergas psnr
    print(std_metrics)

    if output_name:
        sio.savemat("Weights/" + output_name, {'output': output.detach().numpy()})


if __name__ == "__main__":
    # ########### test result #############
    # test(model)

    # ########### get params size #############
    summary(model, [(8, 64, 64), (1, 256, 256)])
