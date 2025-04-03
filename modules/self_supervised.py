import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeContrastiveAugmentation:
    """时间域对比学习数据增强"""

    def __init__(self, jitter_scale=0.1, time_mask_ratio=0.2):
        """
        参数:
            jitter_scale: 扰动幅度
            time_mask_ratio: 时间掩码比例
        """
        self.jitter_scale = jitter_scale
        self.time_mask_ratio = time_mask_ratio

    def __call__(self, x):
        """应用时间域增强"""
        # 复制输入以避免修改原始数据
        x_aug = x.clone()
        batch_size, channels, time_steps = x.shape

        # 1. 添加随机扰动
        noise = torch.randn_like(x_aug) * self.jitter_scale * x_aug.std(dim=2, keepdim=True)
        x_aug = x_aug + noise

        # 2. 随机时间掩码
        mask_length = int(time_steps * self.time_mask_ratio)
        if mask_length > 0:
            for i in range(batch_size):
                mask_start = np.random.randint(0, time_steps - mask_length + 1)
                x_aug[i, :, mask_start:mask_start + mask_length] = 0

        # 3. 随机缩放
        scale_factors = 0.8 + 0.4 * torch.rand(batch_size, channels, 1).to(x.device)
        x_aug = x_aug * scale_factors

        return x_aug


class FreqContrastiveAugmentation:
    """频率域对比学习数据增强"""

    def __init__(self, mask_ratio=0.2, enhance_ratio=0.3):
        """
        参数:
            mask_ratio: 频率分量掩码比例
            enhance_ratio: 增强比例
        """
        self.mask_ratio = mask_ratio
        self.enhance_ratio = enhance_ratio

    def __call__(self, cwt_features):
        """应用频率域增强"""
        # 复制输入以避免修改原始数据
        x_aug = cwt_features.clone()
        batch_size, channels, freq_bins, time_steps = cwt_features.shape

        # 随机频率分量掩码
        num_masks = int(freq_bins * self.mask_ratio)
        if num_masks > 0:
            for i in range(batch_size):
                mask_indices = torch.randperm(freq_bins)[:num_masks]
                x_aug[i, :, mask_indices, :] = 0

        # 随机增强低幅度频率分量
        mean_amplitude = torch.mean(x_aug, dim=(2, 3), keepdim=True)
        low_amp_mask = x_aug < (0.5 * mean_amplitude)

        # 只增强部分低幅度区域
        enhance_mask = low_amp_mask & (torch.rand_like(x_aug) < self.enhance_ratio)
        enhance_factors = 1.5 + torch.rand_like(x_aug) * 1.5  # 1.5x-3.0x增强

        x_aug[enhance_mask] = x_aug[enhance_mask] * enhance_factors[enhance_mask]

        return x_aug


class SelfSupervisedLoss(nn.Module):
    """自监督对比学习损失"""

    def __init__(self, temperature=0.1, time_aug=None, freq_aug=None):
        """
        参数:
            temperature: 温度系数
            time_aug: 时间域增强器
            freq_aug: 频率域增强器
        """
        super(SelfSupervisedLoss, self).__init__()

        self.temperature = temperature
        self.time_aug = time_aug or TimeContrastiveAugmentation()
        self.freq_aug = freq_aug or FreqContrastiveAugmentation()

        # 多尺度一致性损失
        self.consistency_margin = 0.5

    def compute_nce_loss(self, features, aug_features):
        """计算归一化温度交叉熵损失"""
        # L2标准化特征
        features = F.normalize(features, dim=1)
        aug_features = F.normalize(aug_features, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, aug_features.T) / self.temperature

        # 对角线是正样本对
        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def compute_time_contrastive_loss(self, features, x_time):
        """计算时间域对比损失"""
        # 应用时间域增强
        x_time_aug = self.time_aug(x_time)

        # 获取增强样本的特征
        _, _, time_aug_embed, _ = features(x_time_aug)

        # 原始样本的特征
        _, _, time_embed, _ = features(x_time)

        # 计算NCE损失
        loss = self.compute_nce_loss(time_embed, time_aug_embed)

        return loss

    def compute_freq_contrastive_loss(self, features, x_freq):
        """计算频率域对比损失"""
        # 获取CWT特征
        batch_size = x_freq.size(0)
        cwt_features = []

        for i in range(batch_size):
            # 对每个样本独立应用CWT
            cwt_result = features.cwt(x_freq[i].mean(dim=0))  # [freq_bins, time_steps]
            cwt_features.append(cwt_result.unsqueeze(0))  # 添加通道维度

        # 合并批次维度
        cwt_features = torch.stack(cwt_features)  # [batch_size, 1, freq_bins, time_steps]

        # 应用频率域增强
        cwt_aug_features = self.freq_aug(cwt_features)

        # 获取增强样本的特征，通过特征提取器的freq_enc部分
        freq_aug_features = features.freq_enc(cwt_aug_features)
        freq_aug_pooled = F.adaptive_avg_pool2d(freq_aug_features, 1).squeeze(-1).squeeze(-1)
        freq_aug_embed = features.freq_projector(freq_aug_pooled)

        # 获取原始样本的频率特征
        _, _, _, freq_embed = features(x_freq)

        # 计算NCE损失
        loss = self.compute_nce_loss(freq_embed, freq_aug_embed)

        return loss

    def compute_multi_scale_consistency(self, time_embed, freq_embed, time_aug_embed, freq_aug_embed):
        """计算多尺度时频一致性损失"""
        # 计算原始嵌入之间的距离
        orig_dist = F.pairwise_distance(time_embed, freq_embed, p=2)

        # 计算包含增强嵌入的距离
        aug_time_orig_freq_dist = F.pairwise_distance(time_aug_embed, freq_embed, p=2)
        orig_time_aug_freq_dist = F.pairwise_distance(time_embed, freq_aug_embed, p=2)
        aug_time_aug_freq_dist = F.pairwise_distance(time_aug_embed, freq_aug_embed, p=2)

        # 三元组损失：原始距离应小于增强距离减去边界
        loss_1 = F.relu(orig_dist - aug_time_orig_freq_dist + self.consistency_margin)
        loss_2 = F.relu(orig_dist - orig_time_aug_freq_dist + self.consistency_margin)
        loss_3 = F.relu(orig_dist - aug_time_aug_freq_dist + self.consistency_margin)

        # 合并损失
        loss = (loss_1 + loss_2 + loss_3) / 3.0

        return loss.mean()

    def forward(self, features, x):
        """
        计算自监督损失

        参数:
            features: 特征提取器
            x: 输入数据 [batch_size, channels, time_steps]

        返回:
            time_loss: 时间域对比损失
            freq_loss: 频率域对比损失
            consistency_loss: 多尺度一致性损失
        """
        # 计算时间域对比损失
        time_loss = self.compute_time_contrastive_loss(features, x)

        # 计算频率域对比损失
        freq_loss = self.compute_freq_contrastive_loss(features, x)

        # 获取原始和增强的嵌入用于一致性损失
        x_time_aug = self.time_aug(x)
        _, _, time_embed, freq_embed = features(x)
        _, _, time_aug_embed, _ = features(x_time_aug)

        # 获取频率增强的嵌入
        batch_size = x.size(0)
        cwt_features = []

        for i in range(batch_size):
            cwt_result = features.cwt(x[i].mean(dim=0))
            cwt_features.append(cwt_result.unsqueeze(0))

        cwt_features = torch.stack(cwt_features)
        cwt_aug_features = self.freq_aug(cwt_features)

        freq_aug_features = features.freq_enc(cwt_aug_features)
        freq_aug_pooled = F.adaptive_avg_pool2d(freq_aug_features, 1).squeeze(-1).squeeze(-1)
        freq_aug_embed = features.freq_projector(freq_aug_pooled)

        # 计算多尺度一致性损失
        consistency_loss = self.compute_multi_scale_consistency(
            time_embed, freq_embed, time_aug_embed, freq_aug_embed
        )

        return time_loss, freq_loss, consistency_loss