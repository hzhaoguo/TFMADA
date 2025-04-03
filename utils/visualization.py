import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_features(model, source_loader, target_loader, device, save_path=None, n_samples=1000):
    """可视化源域和目标域的特征分布

    参数:
        model: 训练好的模型
        source_loader: 源域数据加载器
        target_loader: 目标域数据加载器
        device: 设备
        save_path: 保存路径
        n_samples: 采样数量
    """
    model.eval()

    # 收集源域和目标域的特征与标签
    source_features = []
    source_labels = []
    target_features = []
    target_labels = []

    # 提取源域特征
    for data, labels in source_loader:
        if len(source_features) * data.size(0) >= n_samples:
            break

        data = data.to(device)

        with torch.no_grad():
            _, _, _, time_embed, freq_embed = model(data)
            # 使用时间和频率嵌入的平均作为特征
            features = (time_embed + freq_embed) / 2

        source_features.append(features.cpu().numpy())
        source_labels.append(labels.numpy())

    # 提取目标域特征
    for data, labels in target_loader:
        if len(target_features) * data.size(0) >= n_samples:
            break

        data = data.to(device)

        with torch.no_grad():
            _, _, _, time_embed, freq_embed = model(data)
            # 使用时间和频率嵌入的平均作为特征
            features = (time_embed + freq_embed) / 2

        target_features.append(features.cpu().numpy())
        target_labels.append(labels.numpy())

    # 合并批次
    source_features = np.vstack(source_features)
    source_labels = np.concatenate(source_labels)
    target_features = np.vstack(target_features)
    target_labels = np.concatenate(target_labels)

    # 随机采样，避免过多样本
    if len(source_features) > n_samples:
        indices = np.random.choice(len(source_features), n_samples, replace=False)
        source_features = source_features[indices]
        source_labels = source_labels[indices]

    if len(target_features) > n_samples:
        indices = np.random.choice(len(target_features), n_samples, replace=False)
        target_features = target_features[indices]
        target_labels = target_labels[indices]

    # 合并源域和目标域特征用于t-SNE
    all_features = np.vstack([source_features, target_features])
    domain_labels = np.concatenate([np.zeros(len(source_features)), np.ones(len(target_features))])
    all_class_labels = np.concatenate([source_labels, target_labels])

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
    tsne_features = tsne.fit_transform(all_features)

    # 分离源域和目标域的t-SNE特征
    source_tsne = tsne_features[:len(source_features)]
    target_tsne = tsne_features[len(source_features):]

    # 可视化t-SNE结果
    plt.figure(figsize=(12, 5))

    # 按领域分
    plt.subplot(1, 2, 1)
    plt.scatter(source_tsne[:, 0], source_tsne[:, 1], c='r', marker='o', alpha=0.6, label='源域')
    plt.scatter(target_tsne[:, 0], target_tsne[:, 1], c='b', marker='x', alpha=0.6, label='目标域')
    plt.title('按领域的特征分布')
    plt.legend()

    # 按类别分
    plt.subplot(1, 2, 2)
    unique_classes = np.unique(all_class_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

    # 源域类别（实心标记）
    for i, cls in enumerate(unique_classes):
        plt.scatter(source_tsne[source_labels == cls, 0], source_tsne[source_labels == cls, 1],
                    c=[colors[i]], marker='o', alpha=0.6, label=f'源-类别{cls}')

    # 目标域类别（空心标记）
    for i, cls in enumerate(unique_classes):
        plt.scatter(target_tsne[target_labels == cls, 0], target_tsne[target_labels == cls, 1],
                    c=[colors[i]], marker='x', alpha=0.6, label=f'目标-类别{cls}')

    plt.title('按类别的特征分布')
    # plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_loss_curves(losses, save_path=None):
    """绘制损失曲线

    参数:
        losses: 损失字典，键为损失名称，值为损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    for loss_name, loss_values in losses.items():
        plt.plot(loss_values, label=loss_name)

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()


def plot_cwt_spectrogram(cwt_coeffs, scales, save_path=None):
    """绘制CWT谱图

    参数:
        cwt_coeffs: CWT系数 [scales, time_steps]
        scales: 尺度列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    # 绘制小波谱图
    plt.imshow(np.abs(cwt_coeffs), aspect='auto', cmap='jet',
               extent=[0, cwt_coeffs.shape[1], scales[-1], scales[0]])

    plt.colorbar(label='Magnitude')
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.title('Continuous Wavelet Transform Spectrogram')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()