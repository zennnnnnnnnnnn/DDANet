from models.PanNet import PanNet


from common.evaluate import *
from common.for_train import *

data_path = "D:/MyData/pythonData/big_directory/test_wv3_OrigScale_multiExm1.h5"
# model_path = "Weights_DDA/540.pth"
model_path = "E:/新建文件夹/CV/start!/PanHDNet_result/0-4——PanNet终极版之无限究极版/160.pth"
# output_name = '100-DDANet.mat'
output_name = None

model = PanNet()

import scipy.io as sio
import torch

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def test(model):
    pan, ms, lms, gt = get_data(data_path, data_mode="full")

    load_result = model.load_state_dict(torch.load(model_path))
    print("loading weight from ", data_path, " result:", load_result)
    model.eval()

    import time
    time1 = time.time()
    output = model(ms, pan)
    print(time.time() - time1)

    metrics = []
    for i in range((output.shape[0])):
        d_lambda = D_lambda(output[i], ms[i])
        d_s = D_s(output[i], ms[i], pan[i])
        qnr = Qnr(output[i], ms[i], pan[i])
        metrics.append([d_lambda, d_s, qnr])
    mean_metrics = torch.tensor(metrics).mean(dim=0)
    std_metrics = torch.tensor(metrics).std(dim=0)

    print(mean_metrics)  # Dlamda Ds qnr
    print(std_metrics)

    if output_name:
        sio.savemat("Weights/" + output_name, {'output': output.detach().numpy()})


if __name__ == "__main__":
    test(model)


