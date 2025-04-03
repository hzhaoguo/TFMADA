import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(nn.Module):
    """教师模型，使用指数移动平均更新参数"""

    def __init__(self, student_model, momentum=0.999):
        """
        参数:
            student_model: 学生模型（目标模型）
            momentum: EMA动量参数
        """
        super(TeacherModel, self).__init__()

        # 创建与学生模型相同的网络结构
        self.feature_extractor = type(student_model.feature_extractor)(student_model.feature_extractor.config)
        self.classifier = type(student_model.classifier)(student_model.classifier.input_dim,
                                                         student_model.classifier.num_classes)

        # 初始化参数
        self.momentum = momentum
        self._initialize(student_model)

        # 确保教师模型不计算梯度
        for param in self.parameters():
            param.detach_()

    def _initialize(self, student_model):
        """初始化教师模型参数"""
        for t_param, s_param in zip(self.feature_extractor.parameters(), student_model.feature_extractor.parameters()):
            t_param.data.copy_(s_param.data)

        for t_param, s_param in zip(self.classifier.parameters(), student_model.classifier.parameters()):
            t_param.data.copy_(s_param.data)

    def update(self, student_model):
        """使用EMA更新教师模型参数"""
        with torch.no_grad():
            for t_param, s_param in zip(self.feature_extractor.parameters(),
                                        student_model.feature_extractor.parameters()):
                t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

            for t_param, s_param in zip(self.classifier.parameters(), student_model.classifier.parameters()):
                t_param.data = t_param.data * self.momentum + s_param.data * (1 - self.momentum)

    def forward(self, x):
        """
        输入:
            x: [batch_size, input_channels, time_steps]
        返回:
            time_pred: 基于时间特征的预测
            freq_pred: 基于频率特征的预测
            fused_pred: 融合的预测
        """
        # 提取特征
        time_features, freq_features, time_embed, freq_embed = self.feature_extractor(x)

        # 分别预测
        time_pred = self.classifier(time_embed)
        freq_pred = self.classifier(freq_embed)

        # 融合预测 (简单平均)
        fused_pred = (time_pred + freq_pred) / 2

        return time_pred, freq_pred, fused_pred


class PseudoLabelGenerator:
    """伪标签生成器"""

    def __init__(self, initial_threshold=0.8, max_threshold=0.95, ramp_factor=0.005):
        """
        参数:
            initial_threshold: 初始置信度阈值
            max_threshold: 最大置信度阈值
            ramp_factor: 阈值增长因子
        """
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.max_threshold = max_threshold
        self.ramp_factor = ramp_factor
        self.kl_threshold = 0.5  # 时频一致性KL散度阈值

    def update_threshold(self, progress):
        """更新阈值"""
        self.current_threshold = min(
            self.initial_threshold + (self.max_threshold - self.initial_threshold) * progress,
            self.max_threshold
        )

    def generate_labels(self, time_pred, freq_pred, fused_pred, class_counts=None):
        """
        生成伪标签

        参数:
            time_pred: 基于时间特征的预测 [batch_size, num_classes]
            freq_pred: 基于频率特征的预测 [batch_size, num_classes]
            fused_pred: 融合的预测 [batch_size, num_classes]
            class_counts: 类别计数，用于平衡阈值

        返回:
            pseudo_labels: 生成的伪标签 [batch_size]
            mask: 指示哪些样本被选为伪标签 [batch_size]
        """
        batch_size = fused_pred.size(0)
        num_classes = fused_pred.size(1)

        # 计算置信度和预测类别
        confidence, predictions = torch.max(fused_pred, dim=1)

        # 计算时间和频率预测之间的KL散度
        kl_time_freq = F.kl_div(
            F.log_softmax(time_pred, dim=1),
            F.softmax(freq_pred, dim=1),
            reduction='none'
        ).sum(dim=1)

        # 根据类别频率调整阈值
        thresholds = torch.ones_like(confidence) * self.current_threshold

        if class_counts is not None:
            max_count = max(class_counts.values())
            for class_idx in range(num_classes):
                if class_idx in class_counts:
                    class_mask = (predictions == class_idx)
                    ratio = class_counts[class_idx] / max_count
                    # 对稀有类别降低阈值
                    thresholds[class_mask] *= max(0.9, ratio)

        # 生成伪标签掩码：满足置信度阈值且时频一致性高
        mask = (confidence >= thresholds) & (kl_time_freq <= self.kl_threshold)

        # 只对掩码中的样本生成标签
        pseudo_labels = torch.zeros_like(predictions)
        pseudo_labels[mask] = predictions[mask]

        return pseudo_labels, mask