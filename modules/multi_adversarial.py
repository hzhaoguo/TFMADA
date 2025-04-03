import torch
import torch.nn as nn
import torch.nn.functional as F


def grl_hook(coeff):
    """梯度反转函数，用于对抗训练"""

    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class MultiAdversarialLoss(nn.Module):
    """多对抗损失模块"""

    def __init__(self, class_num=31):
        """
        参数:
            class_num: 类别数量
        """
        super(MultiAdversarialLoss, self).__init__()
        self.class_num = class_num
        self.adv_criterion = nn.BCEWithLogitsLoss()

    def forward(self, time_pred, freq_pred, target_label):
        """
        计算多对抗损失

        参数:
            time_pred: 时间域判别器预测 [batch_size, 1]
            freq_pred: 频率域判别器预测 [batch_size, 1]
            target_label: 域标签 [batch_size]

        返回:
            adv_loss: 总对抗损失
        """
        # 计算时间域对抗损失
        time_adv_loss = self.adv_criterion(time_pred.squeeze(), target_label.float())

        # 计算频率域对抗损失
        freq_adv_loss = self.adv_criterion(freq_pred.squeeze(), target_label.float())

        # 合并损失
        adv_loss = (time_adv_loss + freq_adv_loss) / 2.0

        return adv_loss


class DomainAdversarialTrainer:
    """领域对抗训练器"""

    def __init__(self, model, discriminator, alpha_scheduler=None):
        """
        参数:
            model: 特征提取器和分类器
            discriminator: 多判别器
            alpha_scheduler: 梯度反转系数调度函数
        """
        self.model = model
        self.discriminator = discriminator
        self.alpha_scheduler = alpha_scheduler or (lambda p: 2.0 / (1.0 + np.exp(-10 * p)) - 1.0)

        # 对抗损失
        self.adv_loss = MultiAdversarialLoss()

    def compute_adversarial_loss(self, source_features, target_features, progress):
        """
        计算对抗损失

        参数:
            source_features: 源域特征
            target_features: 目标域特征
            progress: 训练进度 [0, 1]

        返回:
            adv_loss: 对抗损失
        """
        # 计算梯度反转系数
        alpha = self.alpha_scheduler(progress)

        # 源域和目标域的标签
        batch_size_source = source_features[0].size(0)
        batch_size_target = target_features[0].size(0)
        source_domain = torch.zeros(batch_size_source).to(source_features[0].device)
        target_domain = torch.ones(batch_size_target).to(target_features[0].device)

        # 连接源域和目标域特征
        time_features = torch.cat([source_features[0], target_features[0]], dim=0)
        freq_features = torch.cat([source_features[1], target_features[1]], dim=0)
        domain_labels = torch.cat([source_domain, target_domain], dim=0)

        # 对抗预测
        time_pred, freq_pred = self.discriminator(time_features, freq_features, alpha)

        # 计算对抗损失
        loss = self.adv_loss(time_pred, freq_pred, domain_labels)

        return loss