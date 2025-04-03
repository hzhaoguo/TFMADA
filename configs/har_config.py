from configs.default_config import Config


class HARConfig(Config):
    """人类活动识别数据集配置"""

    def __init__(self):
        super(HARConfig, self).__init__()

        # 数据参数
        self.batch_size = 32
        self.num_workers = 4

        # 模型参数
        self.input_channels = 113
        self.hidden_dim = 32
        self.shared_dim = 64
        self.num_classes = 4

        # 时间特征提取器参数
        self.time_kernels = [5, 3, 3]
        self.use_attention = False

        # 判别器参数
        self.disc_hidden_dims = {'time': 64, 'freq': 64}

        # 训练参数
        self.lr = 0.0005
        self.epochs = 100