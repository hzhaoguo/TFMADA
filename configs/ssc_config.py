from configs.default_config import Config


class SSCConfig(Config):
    """睡眠阶段分类数据集配置"""

    def __init__(self):
        super(SSCConfig, self).__init__()

        # 数据参数
        self.batch_size = 32
        self.num_workers = 4

        # 模型参数
        self.input_channels = 1
        self.hidden_dim = 64
        self.shared_dim = 128
        self.num_classes = 5

        # 时间特征提取器参数
        self.time_kernels = [9, 5, 3]
        self.use_attention = True

        # 判别器参数
        self.disc_hidden_dims = {'time': 64, 'freq': 64}

        # 训练参数
        self.lr = 0.001
        self.epochs = 100