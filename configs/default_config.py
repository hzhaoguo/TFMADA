class Config:
    """TFMADA默认配置"""

    def __init__(self):
        # 数据参数
        self.batch_size = 64
        self.num_workers = 4

        # 模型参数
        self.input_channels = 1
        self.hidden_dim = 64
        self.shared_dim = 128
        self.num_classes = 3

        # CWT参数
        self.cwt_scales = 32
        self.wavelet = 'morlet'

        # 时间特征提取器参数
        self.time_kernels = [7, 5, 3]
        self.use_attention = False

        # 频率特征提取器参数
        self.freq_kernels = [(3, 5), (3, 3)]

        # 判别器参数
        self.disc_hidden_dims = {'time': 128, 'freq': 128}

        # 自监督学习参数
        self.temperature = 0.1
        self.consistency_margin = 0.5

        # 教师模型参数
        self.teacher_momentum = 0.999
        self.initial_threshold = 0.8
        self.max_threshold = 0.95
        self.ramp_factor = 0.005

        # 训练参数
        self.lr = 0.001
        self.weight_decay = 3e-4
        self.epochs = 100
        self.pseudo_weight = 0.005


def get_config(dataset_name):
    """根据数据集名称获取配置"""
    config = Config()

    if dataset_name.upper() == 'MFD':
        # 机器故障诊断数据集配置
        config.input_channels = 8
        config.hidden_dim = 64
        config.shared_dim = 128
        config.num_classes = 3
        config.time_kernels = [7, 5, 3]
        config.use_attention = False
        config.disc_hidden_dims = {'time': 128, 'freq': 128}

    elif dataset_name.upper() == 'HAR':
        # 人类活动识别数据集配置
        config.input_channels = 113
        config.hidden_dim = 32
        config.shared_dim = 64
        config.num_classes = 4
        config.time_kernels = [5, 3, 3]
        config.use_attention = False
        config.disc_hidden_dims = {'time': 64, 'freq': 64}

    elif dataset_name.upper() == 'SSC':
        # 睡眠阶段分类数据集配置
        config.input_channels = 1
        config.hidden_dim = 64
        config.shared_dim = 128
        config.num_classes = 5
        config.time_kernels = [9, 5, 3]
        config.use_attention = True
        config.disc_hidden_dims = {'time': 64, 'freq': 64}

    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    return config