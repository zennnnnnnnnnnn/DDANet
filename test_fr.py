from models_DDA.MSDCNN_DDA import MSDCNN


from common.evaluate import *
from common.for_train import *

data_path = "D:/MyData/pythonData/big_directory/test_wv3_OrigScale_multiExm1.h5"
model_path = "E:/新建文件夹/CV/start!/PanHDNet_result/5-4——MSDCNN_DDA/200.pth"
# output_name = '100-DDANet.mat'
output_name = None

model = MSDCNN()

import scipy.io as sio
import torch

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def test(model):
    pan, ms, lms, gt = get_data(data_path, data_mode="full")

    load_result = model.load_state_dict(torch.load(model_path))
    print("loading weight from ", model_path, " result:", load_result)
    model.eval()

    import time
    time1 = time.time()
    output = model(lms, pan)
    print(time.time() - time1)

    metrics = []
    for i in range((output.shape[0])):
        d_lambda = D_lambda(output[i], ms[i])
        d_s = D_s(output[i], ms[i], pan[i])
        qnr = Qnr(output[i], ms[i], pan[i])
        metrics.append([d_lambda, d_s, qnr])
    mean_metrics = torch.tensor(metrics).mean(dim=0)
    std_metrics = torch.tensor(metrics).std(dim=0)

    print(mean_metrics)  # Dlambda Ds qnr
    print(std_metrics)

    if output_name:
        sio.savemat("Weights/" + output_name, {'output': output.detach().numpy()})


if __name__ == "__main__":
    test(model)

"""
pannet
tensor([0.0320, 0.0636, 0.9068], dtype=torch.float64)
tensor([0.0183, 0.0304, 0.0421], dtype=torch.float64)

pannet_DDA
tensor([0.0368, 0.0571, 0.9084], dtype=torch.float64)
tensor([0.0166, 0.0312, 0.0403], dtype=torch.float64)


MSDCNN
tensor([0.0317, 0.0678, 0.9030], dtype=torch.float64)
tensor([0.0183, 0.0324, 0.0437], dtype=torch.float64)

MSDCNN_DDA



fusionnet
tensor([0.0344, 0.0504, 0.9173], dtype=torch.float64)
tensor([0.0198, 0.0294, 0.0419], dtype=torch.float64)

fusionnet_DDA
tensor([0.0367, 0.0546, 0.9110], dtype=torch.float64)
tensor([0.0203, 0.0274, 0.0401], dtype=torch.float64)


LACNet

"""

