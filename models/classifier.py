import torch
import torch.nn as nn


class Classifier(nn.Module):
    """分类器"""

    def __init__(self, input_dim, num_classes, hidden_dim=None):
        """
        参数:
            input_dim: 输入特征维度
            num_classes: 类别数量
            hidden_dim: 隐藏层维度，默认为input_dim的2倍
        """
        super(Classifier, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim * 2

        self.input_dim = input_dim
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        输入:
            x: [batch_size, input_dim] 特征向量
        返回:
            logits: [batch_size, num_classes] 预测logits
        """
        return self.layers(x)