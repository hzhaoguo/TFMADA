import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class TimeDiscriminator(nn.Module):
    """时间域判别器"""

    def __init__(self, input_dim, hidden_dim=None):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度，默认为input_dim的2倍
        """
        super(TimeDiscriminator, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, alpha=1.0):
        """
        输入:
            x: [batch_size, input_dim] 时间域特征
            alpha: 梯度反转放大因子
        返回:
            domain_pred: [batch_size, 1] 域预测
        """
        # 应用梯度反转
        reversed_x = GradientReversalLayer.apply(x, alpha)

        # 域分类
        domain_pred = self.layers(reversed_x)

        return domain_pred


class FrequencyDiscriminator(nn.Module):
    """频率域判别器"""

    def __init__(self, input_dim, hidden_dim=None):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度，默认为input_dim的2倍
        """
        super(FrequencyDiscriminator, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, alpha=1.0):
        """
        输入:
            x: [batch_size, input_dim] 频率域特征
            alpha: 梯度反转放大因子
        返回:
            domain_pred: [batch_size, 1] 域预测
        """
        # 应用梯度反转
        reversed_x = GradientReversalLayer.apply(x, alpha)

        # 域分类
        domain_pred = self.layers(reversed_x)

        return domain_pred


class MultiDiscriminator(nn.Module):
    """多判别器"""

    def __init__(self, time_dim, freq_dim, hidden_dims=None):
        """
        参数:
            time_dim: 时间域特征维度
            freq_dim: 频率域特征维度
            hidden_dims: 字典，指定两个判别器的隐藏层维度
        """
        super(MultiDiscriminator, self).__init__()

        if hidden_dims is None:
            hidden_dims = {'time': None, 'freq': None}

        self.time_discriminator = TimeDiscriminator(time_dim, hidden_dims.get('time'))
        self.freq_discriminator = FrequencyDiscriminator(freq_dim, hidden_dims.get('freq'))

    def forward(self, time_features, freq_features, alpha=1.0):
        """
        输入:
            time_features: [batch_size, time_dim] 时间域特征
            freq_features: [batch_size, freq_dim] 频率域特征
            alpha: 梯度反转放大因子
        返回:
            time_pred: 时间域判别器预测
            freq_pred: 频率域判别器预测
        """
        time_pred = self.time_discriminator(time_features, alpha)
        freq_pred = self.freq_discriminator(freq_features, alpha)

        return time_pred, freq_pred