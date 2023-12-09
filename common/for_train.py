import os
import h5py
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def get_tem():
    # reload time of control temperature
    shell_str = 'nvidia-smi | findstr "%"'
    result = os.popen(shell_str).read().split('\n')[0]
    parts = result.split()
    return int(parts[2][:-1])


class FuseDataset(Dataset):
    def __init__(self, pan, ms, lms, gt, testing=False):
        self.pan = pan
        self.ms = ms
        self.lms = lms
        self.gt = gt

        if testing:
            self.pan = pan[:30]
            self.ms = ms[:30]
            self.lms = lms[:30]
            self.gt = gt[:30]

    def __len__(self):
        return self.gt.size(0)

    def __getitem__(self, index):
        return self.pan[index], self.ms[index], self.lms[index], self.gt[index]


def get_data(path, is_cuda=False, data_mode="reduce"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist")

    elif path.endswith('.h5'):
        if data_mode == "reduce":   # 256
            print("loading h5 reduced data from: ", path)
            data = h5py.File(path, 'r')

            pan = torch.from_numpy(data['pan'][...] / 2047.0).float()
            ms = torch.from_numpy(data['ms'][...] / 2047.0).float()
            lms = torch.from_numpy(data['lms'][...] / 2047.0).float()
            gt = torch.from_numpy(data['gt'][...] / 2047.0).float()
        elif data_mode == "full":   # 512
            print("loading h5 full data from: ", path)
            data = h5py.File(path, 'r')

            pan = torch.from_numpy(data['pan'][...] / 2047.0).float()
            ms = torch.from_numpy(data['ms'][...] / 2047.0).float()
            lms = torch.from_numpy(data['lms'][...] / 2047.0).float()
            gt = None

    elif path.endswith('.mat'):
        print("loading mat data from: ", path)
        data = sio.loadmat(path)

        pan = data['I_PAN'] / 2047.0
        pan = torch.from_numpy(pan.astype(float)).unsqueeze(2).permute(2, 0, 1).unsqueeze(0).float()
        ms = data['I_MS_LR'] / 2047.0
        ms = torch.from_numpy(ms.astype(float)).permute(2, 0, 1).unsqueeze(0).float()
        lms = data['I_MS'] / 2047.0
        lms = torch.from_numpy(lms.astype(float)).permute(2, 0, 1).unsqueeze(0).float()
        gt = data['I_GT'] / 2047.0
        gt = torch.from_numpy(gt.astype(float)).permute(2, 0, 1).float()

    else:
        print("path endswith something wrong")
        raise ValueError(f"The file format of {path} is not correct. It should be a CSV file.")

    if is_cuda:
        if data_mode == "full":
            return pan.cuda(), ms.cuda(), lms.cuda(), gt
        return pan.cuda(), ms.cuda(), lms.cuda(), gt.cuda()

    return pan, ms, lms, gt


def write_lot(writer, log, epoch):
    writer.add_scalar('SAM', log[0], epoch)
    writer.add_scalar('ERGAS', log[1], epoch)
    writer.add_scalar('PSNR', log[2], epoch)
    writer.add_scalar('SCC', log[3], epoch)
    writer.add_scalar('Q8', log[4], epoch)


if __name__ == "__main__":
    # ################## testing get data #############################
    # pan, ms, lms, gt = get_data("D:/MyData/pythonData/big_directory/test_wv3_multiExm1.h5")
    pan, ms, lms, gt = get_data("D:/MyData/pythonData/big_directory/NY1_WV3_Data/NY1_WV3_R.mat")
    print(gt.shape)

    # ###################

