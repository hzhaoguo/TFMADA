import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import scipy.io as sio
import pandas as pd
from scipy.signal import resample


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""

    def __init__(self, data, labels, transform=None):
        """
        参数:
            data: 时间序列数据 [samples, channels, time_steps]
            labels: 标签 [samples]
            transform: 数据变换
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def load_MFD_dataset(root_dir='data/MFD', domain='H'):
    """加载机器故障诊断数据集"""
    file_path = os.path.join(root_dir, f'domain_{domain}.mat')
    data = sio.loadmat(file_path)

    # 提取特征和标签
    X = data['X']  # [samples, time_steps, channels]
    y = data['y'].squeeze()  # [samples]

    # 转换为 [samples, channels, time_steps] 格式
    X = np.transpose(X, (0, 2, 1))

    # 将类别标签从1-based转换为0-based
    y = y - 1

    return X, y


def load_HAR_dataset(root_dir='data/HAR', domain='A'):
    """加载人类活动识别数据集"""
    file_path = os.path.join(root_dir, f'domain_{domain}.npz')
    data = np.load(file_path)

    X = data['X']  # [samples, channels, time_steps]
    y = data['y']  # [samples]

    return X, y


def load_SSC_dataset(root_dir='data/SSC', domain='EDF'):
    """加载睡眠阶段分类数据集"""
    file_path = os.path.join(root_dir, f'domain_{domain}.npz')
    data = np.load(file_path)

    X = data['X']  # [samples, channels, time_steps]
    y = data['y']  # [samples]

    # 确保统一的序列长度
    target_length = 3000
    if X.shape[2] != target_length:
        X_resampled = []
        for i in range(X.shape[0]):
            X_resampled.append(resample(X[i, 0, :], target_length).reshape(1, -1))
        X = np.array(X_resampled)

    return X, y


def get_datasets(dataset_name, source_domain, target_domain, test_size=0.2, random_state=42):
    """获取源域和目标域的训练集和测试集"""
    # 根据数据集名称选择加载函数
    if dataset_name.upper() == 'MFD':
        load_dataset = load_MFD_dataset
    elif dataset_name.upper() == 'HAR':
        load_dataset = load_HAR_dataset
    elif dataset_name.upper() == 'SSC':
        load_dataset = load_SSC_dataset
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    # 加载源域数据
    source_X, source_y = load_dataset(domain=source_domain)

    # 加载目标域数据
    target_X, target_y = load_dataset(domain=target_domain)

    # 分割源域数据为训练集和测试集
    source_X_train, source_X_test, source_y_train, source_y_test = train_test_split(
        source_X, source_y, test_size=test_size, random_state=random_state, stratify=source_y
    )

    # 分割目标域数据为训练集和测试集
    target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(
        target_X, target_y, test_size=test_size, random_state=random_state, stratify=target_y
    )

    # 创建数据集对象
    source_train_dataset = TimeSeriesDataset(source_X_train, source_y_train)
    source_test_dataset = TimeSeriesDataset(source_X_test, source_y_test)
    target_train_dataset = TimeSeriesDataset(target_X_train, target_y_train)
    target_test_dataset = TimeSeriesDataset(target_X_test, target_y_test)

    return source_train_dataset, source_test_dataset, target_train_dataset, target_test_dataset