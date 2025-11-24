import json
from pathlib import Path
import psutil
import traceback

from torch.utils.data import Dataset, DataLoader

# server121 = r"/kulang/bigmodeldata/era5/"
# serverxueyuan = r"/export/home/jsj_2817/fangteng/bigmodeldata/era5/"
# server160g = r"/root/autodl-tmp/big_model_data/"

# all_time_path = ["/root/autodl-tmp/new_low_sky/low_sky_data/2024_07",
#                  "/root/autodl-tmp/new_low_sky/low_sky_data/2024_07/19to25",
#                  "/root/autodl-tmp/new_low_sky/low_sky_data/2024_07/25to30"]
all_time_path = ["/lixj/low_sky_data/2024_07",
                 "/lixj/low_sky_data/2024_07/19to25",
                 "/lixj/low_sky_data/2024_07/25to30"]

root_path = "/lixj/low_sky_data/"
# root_path = "/root/autodl-tmp/new_low_sky/low_sky_data/"

# folder_path = '/root/autodl-tmp/new_low_sky/low_sky_data/ERA5/'
# npy_path = "/root/autodl-tmp/new_low_sky/low_sky_data/low_sky_npy/"

nc_test_path = ["/lixj/low_sky_data/2024_08",
                "/lixj/low_sky_data/2024_09"]
# nc_test_path = ["/root/autodl-tmp/new_low_sky/low_sky_data/2024_08",
#                 "/root/autodl-tmp/new_low_sky/low_sky_data/2024_09"]
process = psutil.Process()

def get_nc(path):
    nc_files = []
    # 获取文件夹下所有文件（不包括子目录）
    for folder_path in path:
        temp = []
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)) and (f.endswith('.nc') or f.endswith('.npy')):
                temp.append(os.path.join(folder_path, f))
        temp.sort()
        nc_files.append(temp)
    return nc_files


# nc_train.append(["/root/autodl-tmp/new_low_sky/low_sky_data/2024_09/20240924.nc"])

variables_name = ["LU_INDEX", "U", "V", "W", "HGT", "T", "Q2", "TH2", "U10", "V10",
                  "QCLOUD", "QRAIN"]
channels_num = 8
variables_c_num = {
    "LU_INDEX": 1,
    "U": channels_num,
    "V": channels_num,
    "W": channels_num,
    "HGT": 1,
    "T": channels_num,
    "Q2": 1,
    "TH2": 1,
    "U10": 1,
    "V10": 1,
    "QCLOUD": channels_num,
    "QRAIN": channels_num
}

super_param = 5


class datancMuti(Dataset):
    def __init__(self, image_size=224, en_channels=4, de_channels=4, interval=1, mode_str="train", total_time=60,
                 select_range=None, start_time=0, if_random=False, nc_files = None, datasetNums = None, nc_data = None):
        self.en_channels = en_channels
        self.de_channels = de_channels
        self.select_range = select_range
        if select_range is None:
            self.select_range = de_channels
        self.start_time = start_time
        self.datasetNums = datasetNums
        self.mode = mode_str
        self.if_random = if_random
        self.nc_data = nc_data
        self.dataarr = []
        self.means = None
        self.stds = None
        self.image_size = image_size
        self.total_time = total_time
        if interval < 1:
            interval = 1
        self.interval = interval
        var_num = 0
        self.diff_end = []
        self.index_reflect = []
        self.diff_limit = [0]
        self.nc_paths = None
        self.nc_paths = nc_files
        for i in range(0, len(self.datasetNums)):
            self.diff_limit.append(
                self.datasetNums[i][-1] - (self.en_channels + self.start_time + self.total_time - 1) * self.interval + self.diff_limit[-1])
        jsonpath = f'{root_path}one_month_train_stds_means.json'
        # self.count_stds_means(jsonpath)
        self.diff_limit = torch.tensor(self.diff_limit)
        self.stds, self.means = self.load_stds_means(jsonpath)

    def __getitem__(self, index):
        """
        batchsize, var, time, width, height
        """
        result = []
        en_geti = int(index)
        random_int = 0
        if self.if_random:
            random_int = torch.randint(0, self.total_time // self.select_range, (1,)).item()
        de_geti = int(index) + self.en_channels + random_int * self.select_range + self.start_time
        i = torch.nonzero(en_geti < self.diff_limit)[0]
        en_geti = en_geti - self.diff_limit[i - 1]
        dataset_num = self.datasetNums[i - 1]
        files = self.nc_paths[i - 1]
        i = torch.nonzero(en_geti < dataset_num)[0]
        j = en_geti - dataset_num[i - 1]
        end = en_geti + self.en_channels
        de_begin = end + random_int * self.select_range + self.start_time
        if end > dataset_num[i]:
            data1 = self.nc_data[f"{Path(files[i-1]).stem}"][j:]
            data2 = self.nc_data[f"{Path(files[i]).stem}"][:end - dataset_num[i]]
            data = np.concatenate((data1, data2), axis=0)
        else:
            data = self.nc_data[f"{Path(files[i-1]).stem}"][j:end - dataset_num[i - 1]]
        en_data = data
        i = torch.nonzero(de_begin < dataset_num)[0]
        j = de_begin - dataset_num[i - 1]
        end = de_begin + self.de_channels
        if end > dataset_num[i]:
            data1 = self.nc_data[f"{Path(files[i-1]).stem}"][j:]
            data2 = self.nc_data[f"{Path(files[i]).stem}"][:end - dataset_num[i]]
            data = np.concatenate((data1, data2), axis=0)
        else:
            data = self.nc_data[f"{Path(files[i-1]).stem}"][j:end - dataset_num[i - 1]]
        de_data = data
        return en_data, de_data, random_int

    def __len__(self):
        return self.diff_limit[-1]

    def load_stds_means(self, jsonpath):
        assert os.path.exists(jsonpath)
        global channels_num
        # 如果文件存在，打开文件
        with open(jsonpath, 'r') as f:
            check = json.load(f)
            stds = check["std"]
            means = check["mean"]
        tempstd = []
        tempmean = []
        for var in variables_name:
            if variables_c_num[var] == channels_num:
                s_data = torch.tensor(stds[var][:channels_num])
                m_data = torch.tensor(means[var][:channels_num])
            else:
                s_data = torch.tensor(stds[var])
                m_data = torch.tensor(means[var])
            tempstd.append(s_data)
            tempmean.append(m_data)
        stds = torch.cat(tempstd, dim=0)
        means = torch.cat(tempmean, dim=0)
        return stds, means


SHM_BASE_TRAIN = "nc_shared_train"
SHM_BASE_TEST = "nc_shared_test"


def load_stds_means_1(jsonpath):
    assert os.path.exists(jsonpath)
    global channels_num
    # 如果文件存在，打开文件
    with open(jsonpath, 'r') as f:
        check = json.load(f)
        stds = check["std"]
        means = check["mean"]
    tempstd = []
    tempmean = []
    for var in variables_name:
        if variables_c_num[var] == channels_num:
            s_data = torch.tensor(stds[var][:channels_num])
            m_data = torch.tensor(means[var][:channels_num])
        else:
            s_data = torch.tensor(stds[var])
            m_data = torch.tensor(means[var])
        tempstd.append(s_data)
        tempmean.append(m_data)
    stds = torch.cat(tempstd, dim=0)
    means = torch.cat(tempmean, dim=0)
    return stds, means
mean2 = 0.0014126956
std2 = 0.0025898707

import warnings
import sys
from collections import OrderedDict
from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import torch
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

jsonpath="/lixj/low_sky_data/one_month_train_stds_means.json"
gradient = 2

def denormalize_predict(predictions, std, mean):
    std = std.reshape(1, 1, -1, 1, 1)
    mean = mean.reshape(1, 1, -1, 1, 1)
    return predictions * std + mean
def normalize_predict(predictions, std, mean):
    std = std.reshape(1, 1, -1, 1, 1)
    mean = mean.reshape(1, 1, -1, 1, 1)
    return (predictions - mean) / std



def student(device_ids, device, train_loader, test_loader, batch_size, epochs, num_input, num_pre, img_size, model_ori):


    std_array, mean_array = load_stds_means_1(jsonpath)

    std_array = torch.tensor(std_array).float().to(device)
    mean_array = torch.tensor(mean_array).float().to(device)

    model_ori.to(device)

    optimizer = torch.optim.AdamW(model_ori.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=0)

    best_mse = 100
    start_time = time.time()
    for epoch in range(epochs):
        train_loss_set = []
        tmp_mae = 0.
        model_ori.train()
        for i,(in_put, target, _) in enumerate(train_loader):

            in_put = in_put.to(device).float()
            target = target.to(device).float()
            in_put = normalize_predict(in_put, std_array, mean_array)
            preds = model_ori(in_put)


            preds = denormalize_predict(preds, std_array, mean_array)

            loss = 0.
            gama = 0.4

            for i in range(num_pre):
                loss_1 = ((1 - 2.718**(-i))*((preds[:, i, :, :, :]-target[:, i, :, :, :])**2)).mean()
                loss_2 = ((2.718**(-(preds[:, i, :, :, :]-target[:, i, :, :, :])**2))*((preds[:, i, :, :, :]-target[:, i, :, :, :])**2)).mean()
                loss += gama * loss_1 + (1 - gama) * loss_2



            # loss = loss + loss1  # wast
            train_loss_set.append(loss.cpu().detach().numpy())
            ###################################################################################

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        end_time = time.time()  # 记录训练结束时间
        training_time = end_time - start_time  # 计算训练时间差
        f = open('./xiaowei/tau_time_30_my_0_dot_4.txt', 'a')
        f.write(str('epoch：{:.4f}'.format(epoch)) + str('训练时间：{:.4f}'.format(training_time)))
        f.write('\n')
        f.close()
        model_ori.eval()
        if epoch % 1 == 0:
            test_loss_set = []
            MSE = []
            with torch.no_grad():
                for i,(x, y, _) in enumerate(test_loader):


                    x = x.to(device).float()
                    y = y.to(device).float()
                    x = normalize_predict(x, std_array, mean_array)
                    preds_ = model_ori(x)


                    preds_ = denormalize_predict(preds_, std_array, mean_array)



                    loss_mse = torchMSE(preds_, y)

                    MSE.append(loss_mse.cpu().detach().numpy())

                    test_loss_set.append(loss_mse.cpu().detach().numpy())
            if np.array(MSE).mean() < best_mse:
                best_mse = np.array(MSE).mean()
                torch.save(model_ori.state_dict(), './xiaowei/tau_time_30_my_0_dot_4.pth')

            f = open('./xiaowei/tau_time_30_my_0_dot_4.txt', 'a')
            f.write(str('Student-Epoch:{}\tLoss:{:.4f}'
                        .format(epoch + 1,loss)))

            f.write('\n')
            f.write(
                'train_loss=' + str(np.array(train_loss_set).mean()) + ',test_loss=' + str(
                    np.array(test_loss_set).mean()) + ',mse=' + str(np.array(MSE).mean()))
            f.write('\n')
            f.close()
        model_ori.train()

    f = open('./xiaowei/tau_time_30_my_0_dot_4.txt', 'a')
    f.write(str('train over'))
    f.close()

def relative_absolute_error(true, pred):
    true_mean = torch.mean(true)
    squared_error_num = torch.sum(torch.abs(true - pred))
    squared_error_den = torch.sum(torch.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss
def accuracy_score(y_true, y_pred, threshold):
    # 获取张量的形状信息
    batch_size, num_images_per_batch, level, image_height, image_width = y_true.shape

    # 将四维张量展平为二维张量，第一维是总样本数，第二维是图像长宽的乘积
    y_true_flat = y_true.view(batch_size * num_images_per_batch * level, -1)
    y_pred_flat = y_pred.view(batch_size * num_images_per_batch * level, -1)

    # 计算预测与真实值之间的差异
    diff = torch.abs(y_pred_flat - y_true_flat)

    # 判断预测是否在阈值内认为正确
    correct = torch.sum(diff <= threshold, dim=(0,1))

    # 计算准确率
    total = batch_size * num_images_per_batch * level * image_height * image_width  # 总样本数量
    accuracy = correct / total

    return accuracy

def R2_score(pred, true):
    ssr = torch.sum((pred - true) ** 2)
    sse = torch.sum((torch.mean(true) - true) ** 2)
    return max(1 - ssr/(sse+1e-12), 0)

def torchMSE(pred, true):
    a = pred.float() - true.float()
    return torch.mean(a ** 2)

def load_paral(model, path):
    state_dict = torch.load(path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():  # k为module.xxx.weight, v为权重
        name = k[7:] if k.startswith("module.") else k   # 截取`module.`后面的xxx.weight
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)


def load_paral_self(model, path):
    state_dict = torch.load(path, map_location="cpu")
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():  # k为module.xxx.weight, v为权重
        # 如果是不需要的层，跳过
        if 'readout' in k:
            continue

        name = k[7:] if k.startswith("module.") else k  # 截取`module.`后面的xxx.weight

        # 如果模型中不存在这个层（即新加入的层），可以跳过或者添加默认初始化
        if name not in model.state_dict():
            print(f"Layer {name} is a new layer and will be initialized.")
            continue

        new_state_dict[name] = v

    # 加载权重
    model.load_state_dict(new_state_dict, strict=False)


warnings.filterwarnings("ignore", message="The given NumPy array is not writable, and pyTorch does not suport non-writable tensors")

def run(obj):

    from model.simvp_model_my_15min import SimVP_Model as my_model_15min
    from model.my_model_paper_2 import MY_model as MY_model_my
    epochs = 50
    num_input = 5
    num_pre = 30
    batch_size = 16
    img_size = 224
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device_ids = [0, 1, 2, 3]
    device = torch.device("cuda:3")
    train_Dataset = datancMuti(en_channels=num_input, de_channels=num_pre, mode_str="train",
                        nc_files=obj["train_file"],
                        datasetNums=obj["data"][f"{SHM_BASE_TRAIN}_index"],
                        nc_data=obj["data"])
    test_Dataset = datancMuti(en_channels=num_input, de_channels=num_pre,
                        mode_str="test", nc_files=obj["test_file"],
                        datasetNums=obj["data"][f"{SHM_BASE_TEST}_index"],
                        nc_data=obj["data"])

    traindataloader = DataLoader(dataset=train_Dataset, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=1)
    testdataloader = DataLoader(dataset=test_Dataset, batch_size=batch_size, shuffle=True,drop_last=True, num_workers=1)

    my_model = my_model_15min((num_input, 54, img_size,img_size),hid_S = 64,hid_T = 512,N_T = 8,N_S = 4,
                                       model_type="tau", drop_path=0.1, spatio_kernel_enc=3, spatio_kernel_dec = 3)
    load_paral(my_model, "/lixj/fangteng/lowsky_muti/xiaowei/model_time_15.pth")
    model_ori = MY_model_my((num_input, 54, img_size,img_size), my_model)
    student(device_ids, device, traindataloader, testdataloader, batch_size, epochs, num_input, num_pre, img_size, model_ori)


def main(obj):
    with open("result_model_time.txt", 'w', buffering=1) as file:  # 行缓冲模式
        sys.stdout = file
        sys.stderr = file
        try:    # 使用上下文管理器控制重定向范围
            run(obj)
        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            sys.exit(1)  # 使用sys.exit替代exit()
