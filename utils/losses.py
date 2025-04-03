import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """分类损失"""

    def __init__(self, weight=None, reduction='mean'):
        super(ClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, pred, target):
        return self.criterion(pred, target)


class AdversarialLoss(nn.Module):
    """对抗损失"""

    def __init__(self, reduction='mean'):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, target):
        return self.criterion(pred.squeeze(), target.float())


class MultiScaleConsistencyLoss(nn.Module):
    """多尺度一致性损失"""

    def __init__(self, margin=0.5, reduction='mean'):
        super(MultiScaleConsistencyLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, time_embed, freq_embed, time_aug_embed, freq_aug_embed):
        # 计算原始嵌入之间的距离
        orig_dist = F.pairwise_distance(time_embed, freq_embed, p=2)

        # 计算包含增强嵌入的距离
        aug_time_orig_freq_dist = F.pairwise_distance(time_aug_embed, freq_embed, p=2)
        orig_time_aug_freq_dist = F.pairwise_distance(time_embed, freq_aug_embed, p=2)
        aug_time_aug_freq_dist = F.pairwise_distance(time_aug_embed, freq_aug_embed, p=2)

        # 三元组损失：原始距离应小于增强距离减去边界
        loss_1 = F.relu(orig_dist - aug_time_orig_freq_dist + self.margin)
        loss_2 = F.relu(orig_dist - orig_time_aug_freq_dist + self.margin)
        loss_3 = F.relu(orig_dist - aug_time_aug_freq_dist + self.margin)

        # 合并损失
        loss = (loss_1 + loss_2 + loss_3) / 3.0

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PseudoLabelLoss(nn.Module):
    """伪标签损失"""

    def __init__(self, weight=None, reduction='mean'):
        super(PseudoLabelLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, pred, pseudo_labels, mask):
        """
        参数:
            pred: 模型预测
            pseudo_labels: 伪标签
            mask: 掩码，指示哪些样本被选为伪标签
        """
        if mask.sum() == 0:
            # 如果没有样本被选为伪标签，返回零损失
            return torch.tensor(0.0, device=pred.device)

        # 只对选中的样本计算损失
        return self.criterion(pred[mask], pseudo_labels[mask])