import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cwt import ContinuousWaveletTransform


class TimeFeatureExtractor(nn.Module):
    """时间域特征提取器"""

    def __init__(self, input_channels, hidden_dim, kernel_sizes=[7, 5, 3], use_attention=False):
        """
        参数:
            input_channels: 输入通道数
            hidden_dim: 隐藏层维度
            kernel_sizes: 卷积核大小序列，从粗粒度到细粒度
            use_attention: 是否使用多头注意力（用于较长时间序列）
        """
        super(TimeFeatureExtractor, self).__init__()

        self.use_attention = use_attention
        self.layers = nn.ModuleList()

        in_channels = input_channels
        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = hidden_dim * (2 ** i)
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ))
            in_channels = out_channels

        if use_attention:
            self.attention = nn.MultiheadAttention(in_channels, num_heads=4, batch_first=True)

        self.output_dim = in_channels

    def forward(self, x):
        """
        输入:
            x: [batch_size, input_channels, time_steps]
        返回:
            features: [batch_size, output_dim, time_steps']
        """
        features = x
        for layer in self.layers:
            features = layer(features)

        if self.use_attention:
            features_perm = features.permute(0, 2, 1)  # [B, T, C]
            attn_out, _ = self.attention(features_perm, features_perm, features_perm)
            features = attn_out.permute(0, 2, 1)  # [B, C, T]

        return features


class FrequencyFeatureExtractor(nn.Module):
    """频率域特征提取器"""

    def __init__(self, input_channels, hidden_dim, kernel_sizes=[(3, 5), (3, 3)]):
        """
        参数:
            input_channels: 输入通道数
            hidden_dim: 隐藏层维度
            kernel_sizes: 2D卷积核大小序列
        """
        super(FrequencyFeatureExtractor, self).__init__()

        self.layers = nn.ModuleList()

        in_channels = input_channels
        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = hidden_dim * (2 ** i)
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 2),
                          padding=(kernel_size[0] // 2, kernel_size[1] // 2)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))
            in_channels = out_channels

        # 频率注意力模块
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

        self.output_dim = in_channels

    def forward(self, x):
        """
        输入:
            x: [batch_size, input_channels, freq_bins, time_steps]
        返回:
            features: [batch_size, output_dim, freq_bins', time_steps']
        """
        features = x
        for layer in self.layers:
            features = layer(features)

        # 应用频率注意力
        attn = self.freq_attention(features)
        features = features * attn

        return features


class TFFeatureExtractor(nn.Module):
    """时间-频率联合特征提取器"""

    def __init__(self, config):
        """
        参数:
            config: 配置对象，包含模型参数
        """
        super(TFFeatureExtractor, self).__init__()

        self.time_enc = TimeFeatureExtractor(
            config.input_channels,
            config.hidden_dim,
            kernel_sizes=config.time_kernels,
            use_attention=config.use_attention
        )

        self.cwt = ContinuousWaveletTransform(
            config.cwt_scales,
            config.wavelet
        )

        self.freq_enc = FrequencyFeatureExtractor(
            1,  # CWT输出单通道时频表示
            config.hidden_dim,
            kernel_sizes=config.freq_kernels
        )

        # 投影器，用于时频共享嵌入空间
        self.time_projector = nn.Linear(self.time_enc.output_dim, config.shared_dim)
        self.freq_projector = nn.Linear(self.freq_enc.output_dim, config.shared_dim)

        self.config = config

    def forward(self, x):
        """
        输入:
            x: [batch_size, input_channels, time_steps]
        返回:
            time_features: 时间域特征
            freq_features: 频率域特征
            time_embed: 时间域嵌入
            freq_embed: 频率域嵌入
        """
        # 时间域特征提取
        time_features = self.time_enc(x)

        # 使用CWT获取时频表示
        batch_size = x.size(0)
        cwt_features = []
        for i in range(batch_size):
            # 对每个样本独立应用CWT
            cwt_result = self.cwt(x[i].mean(dim=0))  # [freq_bins, time_steps]
            cwt_features.append(cwt_result.unsqueeze(0))  # 添加通道维度

        # 合并批次维度
        cwt_features = torch.stack(cwt_features)  # [batch_size, 1, freq_bins, time_steps]

        # 频率域特征提取
        freq_features = self.freq_enc(cwt_features)

        # 全局池化，得到固定维度向量
        time_pooled = F.adaptive_avg_pool1d(time_features, 1).squeeze(-1)
        freq_pooled = F.adaptive_avg_pool2d(freq_features, 1).squeeze(-1).squeeze(-1)

        # 映射到共享嵌入空间
        time_embed = self.time_projector(time_pooled)
        freq_embed = self.freq_projector(freq_pooled)

        return time_features, freq_features, time_embed, freq_embed