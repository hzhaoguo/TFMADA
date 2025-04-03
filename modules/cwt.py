import torch
import torch.nn as nn
import numpy as np
from scipy import signal


class ContinuousWaveletTransform(nn.Module):
    """连续小波变换模块"""

    def __init__(self, scales=32, wavelet='morlet'):
        """
        参数:
            scales: 尺度数量，默认为32个对数间隔的尺度
            wavelet: 小波类型，默认为'morlet'
        """
        super(ContinuousWaveletTransform, self).__init__()

        self.scales = np.logspace(np.log10(0.1), np.log10(1.0), scales)
        self.wavelet = wavelet

    def get_wavelet(self, scale, points):
        """获取给定尺度下的小波函数"""
        if self.wavelet == 'morlet':
            # Morlet小波
            w = 5.0  # 小波参数
            t = np.arange(-points // 2, points // 2)
            norm = 1.0 / (np.sqrt(scale) * np.pi ** (1 / 4))
            wavelet = norm * np.exp(-(t ** 2) / (2 * scale ** 2)) * np.exp(1j * w * t / scale)
            return wavelet.real
        else:
            raise NotImplementedError(f"小波类型 {self.wavelet} 尚未实现")

    def forward(self, x):
        """
        输入:
            x: [time_steps] 时间序列
        返回:
            cwt_result: [scales, time_steps] CWT系数
        """
        # 确保输入是在CPU上，因为我们使用numpy处理
        device = None
        if isinstance(x, torch.Tensor):
            device = x.device
            x = x.cpu().numpy()

        # 计算所有尺度下的CWT
        cwt_result = []
        for scale in self.scales:
            # 获取当前尺度的小波
            wavelet_func = self.get_wavelet(scale, min(len(x), 1024))

            # 卷积计算CWT系数
            coeffs = np.abs(signal.convolve(x, wavelet_func, mode='same'))
            cwt_result.append(coeffs)

        # 堆叠所有尺度的结果
        cwt_result = np.stack(cwt_result)

        # 转换回PyTorch张量
        result_tensor = torch.from_numpy(cwt_result).float()
        if device is not None:
            result_tensor = result_tensor.to(device)

        return result_tensor