from configs.default_config import Config


class MFDConfig(Config):
    """机器故障诊断数据集配置"""

    def __init__(self):
        super(MFDConfig, self).__init__()

        # 数据参数
        self.batch_size = 64
        self.num_workers = 4

        # 模型参数
        self.input_channels = 8
        self.hidden_dim = 64
        self.shared_dim = 128
        self.num_classes = 3

        # 时间特征提取器参数
        self.time_kernels = [7, 5, 3]
        self.use_attention = False

        # 判别器参数
        self.disc_hidden_dims = {'time': 128, 'freq': 128}

        # 训练参数
        self.lr = 0.001
        self.epochs = 100