import torch
import torch.nn as nn
import torch.nn.functional as F
from models.feature_extractor import TFFeatureExtractor
from models.discriminators import MultiDiscriminator
from models.classifier import Classifier
from modules.teacher_model import TeacherModel, PseudoLabelGenerator
import numpy as np


class TFMADA(nn.Module):
    """时间-频率多对抗自适应框架"""

    def __init__(self, config):
        """
        参数:
            config: 配置对象，包含模型参数
        """
        super(TFMADA, self).__init__()

        # 特征提取器
        self.feature_extractor = TFFeatureExtractor(config)

        # 分类器
        self.classifier = Classifier(config.shared_dim, config.num_classes)

        # 多判别器
        self.discriminator = MultiDiscriminator(
            config.shared_dim,
            config.shared_dim,
            {'time': config.disc_hidden_dims.get('time'),
             'freq': config.disc_hidden_dims.get('freq')}
        )

        # 保存配置
        self.config = config

    def forward(self, x):
        """
        输入:
            x: [batch_size, input_channels, time_steps]
        返回:
            pred: 分类预测
            domain_time: 时间域的域预测
            domain_freq: 频率域的域预测
        """
        # 提取特征
        time_features, freq_features, time_embed, freq_embed = self.feature_extractor(x)

        # 域判别
        domain_time, domain_freq = self.discriminator(time_embed, freq_embed)

        # 分类预测（使用时间和频率嵌入的平均）
        shared_embed = (time_embed + freq_embed) / 2
        pred = self.classifier(shared_embed)

        return pred, domain_time, domain_freq, time_embed, freq_embed

    def get_parameters(self):
        """获取模型参数分组，用于优化器"""
        feature_params = list(self.feature_extractor.parameters())
        classifier_params = list(self.classifier.parameters())
        discriminator_params = list(self.discriminator.parameters())

        return {
            'feature_extractor': feature_params,
            'classifier': classifier_params,
            'discriminator': discriminator_params
        }


class TFMADATrainer:
    """TFMADA训练器"""

    def __init__(self, model, config):
        """
        参数:
            model: TFMADA模型
            config: 训练配置
        """
        self.model = model
        self.config = config

        # 创建教师模型
        self.teacher = TeacherModel(model, momentum=config.teacher_momentum)

        # 创建伪标签生成器
        self.pseudo_generator = PseudoLabelGenerator(
            initial_threshold=config.initial_threshold,
            max_threshold=config.max_threshold,
            ramp_factor=config.ramp_factor
        )

        # 分类损失
        self.cls_criterion = nn.CrossEntropyLoss()

        # 域判别损失
        self.adv_criterion = nn.BCEWithLogitsLoss()

        # 设备
        self.device = next(model.parameters()).device

    def compute_classification_loss(self, pred, labels):
        """计算分类损失"""
        return self.cls_criterion(pred, labels)

    def compute_adversarial_loss(self, domain_time, domain_freq, domain_labels):
        """计算对抗损失"""
        # 整合时间和频率域判别器的损失
        time_loss = self.adv_criterion(domain_time.squeeze(), domain_labels.float())
        freq_loss = self.adv_criterion(domain_freq.squeeze(), domain_labels.float())

        return (time_loss + freq_loss) / 2.0

    def compute_pseudo_label_loss(self, pred, pseudo_labels, mask):
        """计算伪标签损失"""
        if mask.sum() == 0:
            return torch.tensor(0.0).to(self.device)

        # 只对选中的样本计算损失
        return self.cls_criterion(pred[mask], pseudo_labels[mask])

    def get_alpha(self, current_step, total_steps, gamma=10.0):
        """计算梯度反转层的alpha参数"""
        p = current_step / total_steps
        return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0

    def update_teacher(self):
        """更新教师模型"""
        self.teacher.update(self.model)

    def train_step(self, source_data, source_labels, target_data, current_step, total_steps):
        """执行一步训练"""
        self.model.train()

        batch_size = source_data.size(0)
        target_batch_size = target_data.size(0)

        # 准备域标签
        source_domain = torch.zeros(batch_size).to(self.device)
        target_domain = torch.ones(target_batch_size).to(self.device)

        # 计算梯度反转的alpha
        alpha = self.get_alpha(current_step, total_steps)

        # 源域前向传播
        source_pred, source_domain_time, source_domain_freq, _, _ = self.model(source_data)

        # 目标域前向传播
        target_pred, target_domain_time, target_domain_freq, _, _ = self.model(target_data)

        # 分类损失（仅源域）
        cls_loss = self.compute_classification_loss(source_pred, source_labels)

        # 对抗损失
        # 时间域判别损失
        time_adv_loss = self.compute_adversarial_loss(
            torch.cat([source_domain_time, target_domain_time]),
            torch.cat([source_domain, target_domain])
        )

        # 频率域判别损失
        freq_adv_loss = self.compute_adversarial_loss(
            torch.cat([source_domain_freq, target_domain_freq]),
            torch.cat([source_domain, target_domain])
        )

        # 总对抗损失
        adv_loss = (time_adv_loss + freq_adv_loss) / 2.0

        # 使用教师模型生成伪标签
        with torch.no_grad():
            self.teacher.eval()
            time_pred, freq_pred, fused_pred = self.teacher(target_data)

            # 统计目标域各类别样本数
            class_counts = {}
            for c in range(self.config.num_classes):
                class_counts[c] = (torch.argmax(fused_pred, dim=1) == c).sum().item()

            # 生成伪标签
            pseudo_labels, mask = self.pseudo_generator.generate_labels(
                time_pred, freq_pred, fused_pred, class_counts
            )

        # 计算伪标签损失
        pseudo_loss = self.compute_pseudo_label_loss(target_pred, pseudo_labels, mask)

        # 更新伪标签阈值
        self.pseudo_generator.update_threshold(current_step / total_steps)

        # 总损失
        total_loss = cls_loss + adv_loss + self.config.pseudo_weight * pseudo_loss

        # 更新教师模型
        self.update_teacher()

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'adv_loss': adv_loss,
            'pseudo_loss': pseudo_loss,
            'pseudo_ratio': mask.float().mean().item()
        }

    def test(self, test_loader):
        """模型测试"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # 模型预测
                pred, _, _, _, _ = self.model(data)

                # 计算准确率
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy