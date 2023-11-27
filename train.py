
start_epoch = 0
end_epoch = 200
batch_size = 32

# 笔记本
train_data_path = 'D:/MyData/pythonData/big_directory/train_wv3.h5'
test_data_path = 'D:/MyData/pythonData/big_directory/test_wv3_multiExm1.h5'
# AutoDL
# train_data_path = '/root/autodl-tmp/Data/train_wv3.h5'
# test_data_path = '/root/autodl-tmp/Data/test_wv3_multiExm1.h5'
# 6
# train_data_path = '/Data2/Datasets/PanCollection/training_data/train_wv3_9714.h5'
# test_data_path = '/Data2/Datasets/PanCollection/test_data/test_wv3_multiExm1.h5'

testing = False

lr = 0.001
lr_list = {100: 3e-4, 120: 1e-4}
ckpt = 20
gpu_device = "0"

import os
from tqdm import tqdm

import torch
from winsound import Beep

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from common.evaluate import analysis_accu
from common.for_train import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
tem_control = True
tem = 65
tem_kp = 0.01

# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
criterion = nn.MSELoss().cuda()

pan_all, ms_all, lms_all, gt_all = get_data(train_data_path)
pan_test, ms_test, lms_test, gt_test = get_data(test_data_path, is_cuda=True)

train_dataset = FuseDataset(pan_all, ms_all, lms_all, gt_all, testing=testing)
loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    # pin_memory=True,
    # num_workers=1,
)


# ############################### Main Train ####################################
def train(model, optimizer, start_epoch=0, epochs=1500, weight_path="Weights_DDA"):
    tem_time = 0

    if start_epoch != 0:
        load_result = model.load_state_dict(torch.load(weight_path + "/{}.pth".format(start_epoch)))
        print("loading weight from ", weight_path, " result:", load_result)

    writer = SummaryWriter(log_dir=weight_path)

    # ============Epoch Train=============== #
    print('Start training...')
    for epoch in range(start_epoch + 1, epochs + 1, 1):

        model.train()
        prev_time = time.time()

        # train
        for pan, ms, lms, gt in tqdm(loader):
            pan = pan.cuda()
            ms = ms.cuda()
            lms = lms.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()
            out = model(lms, pan)
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()

            # control computer temperature
            time.sleep(tem_time)

        # print train info
        elapsed_time = time.time() - prev_time
        print('Elapsed:{} | Epoch: {}/{} | training loss: {:.7f}'.format(elapsed_time, epoch, epochs, loss.item()))

        # reload time of control temperature
        temperature = get_tem()
        tem_time += tem_kp * (temperature - tem)
        if tem_time < 0:
            tem_time = 0

        # save model
        if epoch % ckpt == 0 and not testing:  # if each ckpt epochs, then start to save model
            torch.save(model.state_dict(), weight_path + '/' + "{}.pth".format(epoch))
            Beep(300, 1000)

        # adjust learn rate
        for e, l in list(lr_list.items())[::-1]:
            if epoch > e:
                for p in optimizer.param_groups:
                    p['lr'] = l
                break

        # test
        with torch.no_grad():
            model.eval()

            out = model(lms_test, pan_test)  # call model
            metrics = []
            for i in range((pan_test.shape[0])):
                m = analysis_accu(gt_test[i].permute(1, 2, 0), out[i].permute(1, 2, 0), ratio=4, flag_cut_bounds=True,
                                  dim_cut=21)
                metrics.append(list(m.values()))
            mean_metrics = torch.tensor(metrics).mean(dim=0)
            print(mean_metrics)  # sam ergas psnr SCC Q8
            write_lot(writer, mean_metrics, epoch)

        torch.cuda.empty_cache()

    writer.close()


if __name__ == "__main__":

    from models.FusionNet import FusionNet

    model = FusionNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train(model, optimizer, 120, 300, "Weights_FusionNet")




