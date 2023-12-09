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
tensor([0.0327, 0.0667, 0.9031], dtype=torch.float64)
tensor([0.0193, 0.0313, 0.0417], dtype=torch.float64)


fusionnet
tensor([0.0344, 0.0504, 0.9173], dtype=torch.float64)
tensor([0.0198, 0.0294, 0.0419], dtype=torch.float64)

fusionnet_DDA
tensor([0.0367, 0.0546, 0.9110], dtype=torch.float64)
tensor([0.0203, 0.0274, 0.0401], dtype=torch.float64)


LACNet


\begin{table*}[t]
\caption{Table: Quantitative results on 20 reduced-resolution and 20 full-resolution samples of WV3. (red: best)}
\begin{center}
\begin{tabular}{ccccccc}
\toprule
\multirow{2}{*}{\bf Method} & \multicolumn{5}{c}{\bf Reduced-Resolution} & \multirow{2}{*}{\bf Params}\\
\cmidrule{2-6}
& PSNR & Q8 & SAM & ERGAS & SCC \\
\midrule
\bf PanNet          & 37.381±2.643 & 0.901±0.092 & 3.624±0.695 & 2.641±0.605 & 0.973±0.023  & 0.60MB\\
\bf PanNet+DDA      & \textcolor{red}{38.014}±2.541 & \textcolor{red}{0.908}±0.092 & \textcolor{red}{3.328}±0.622 & \textcolor{red}{2.440}±0.614 & \textcolor{red}{0.977}±0.020   & 0.61MB\\
\bf MSDCNN          & 37.152±2.576 & 0.900±0.090 & 3.707±0.758 & 2.719±0.640 & 0.972±0.022   & 0.87MB\\
\bf MSDCNN+DDA      & \textcolor{red}{37.371}±2.713 & \textcolor{red}{0.903}±0.090 & \textcolor{red}{3.580}±0.668 & \textcolor{red}{2.666}±0.677 & \textcolor{red}{0.973}±0.022   & 0.88MB\\
\bf FusionNet       & 37.647±2.601 & 0.903±0.091 & 3.388±0.657 & 2.544±0.615 & 0.974±0.022  & 0.58MB\\
\bf FusionNet+DDA   & \textcolor{red}{37.834}±2.564 & \textcolor{red}{0.906}±0.090 & \textcolor{red}{3.317}±0.643 & \textcolor{red}{2.480}±0.632 & \textcolor{red}{0.975}±0.022  & 0.60MB\\
\bf LAGNet          & 38.584±2.519 & 0.916±0.087 & 3.129±0.642 & 2.297±0.593 & 0.980±0.017  & 0.58MB\\
\bf LAGNet+DDA      & \textcolor{red}{38.666}±2.637 & \textcolor{red}{0.918}±0.086 & \textcolor{red}{3.085}±0.576 & \textcolor{red}{2.261}±0.565 & \textcolor{red}{0.980}±0.017  & 0.59MB\\
\midrule
\bf Ideal value     &0      &0      &+$\infty$  &1      &1      
\bottomrule
\end{tabular}
\end{center}
\end{table*}


"""

