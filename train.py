import torch
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from torch.utils.data import DataLoader
from models.tfmada import TFMADA, TFMADATrainer
from utils.data_loader import get_datasets
from utils.visualization import visualize_features
from configs.default_config import get_config


def train(args):
    """训练TFMADA模型"""
    # 加载配置
    config = get_config(args.dataset)
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.epochs = args.epochs

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    source_train, source_test, target_train, target_test = get_datasets(
        args.dataset, args.source_domain, args.target_domain
    )

    # 创建数据加载器
    source_train_loader = DataLoader(
        source_train, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    target_train_loader = DataLoader(
        target_train, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    source_test_loader = DataLoader(
        source_test, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    target_test_loader = DataLoader(
        target_test, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    # 创建模型
    model = TFMADA(config).to(device)

    # 创建训练器
    trainer = TFMADATrainer(model, config)

    # 创建优化器
    parameters = model.get_parameters()
    optimizer = optim.Adam([
        {'params': parameters['feature_extractor'], 'lr': config.lr},
        {'params': parameters['classifier'], 'lr': config.lr},
    ], weight_decay=config.weight_decay)

    discriminator_optimizer = optim.Adam(
        parameters['discriminator'], lr=config.lr, weight_decay=config.weight_decay
    )

    # 训练循环
    total_steps = len(source_train_loader) * config.epochs
    best_acc = 0.0

    # 创建结果目录
    result_dir = f"results/{args.dataset}/{args.source_domain}_to_{args.target_domain}"
    os.makedirs(result_dir, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()

        # 创建源域和目标域数据迭代器
        source_iter = iter(source_train_loader)
        target_iter = iter(target_train_loader)

        num_batches = min(len(source_train_loader), len(target_train_loader))

        # 每个epoch的总损失
        epoch_loss = {
            'total_loss': 0.0,
            'cls_loss': 0.0,
            'adv_loss': 0.0,
            'pseudo_loss': 0.0,
            'pseudo_ratio': 0.0
        }

        for batch_idx in range(num_batches):
            current_step = epoch * num_batches + batch_idx

            # 获取批次数据
            try:
                source_data, source_labels = next(source_iter)
            except StopIteration:
                source_iter = iter(source_train_loader)
                source_data, source_labels = next(source_iter)

            try:
                target_data, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_train_loader)
                target_data, _ = next(target_iter)

            # 移至设备
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)

            # 清除梯度
            optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # 训练步骤
            loss_dict = trainer.train_step(
                source_data, source_labels, target_data, current_step, total_steps
            )

            # 反向传播和优化
            loss_dict['total_loss'].backward()
            optimizer.step()
            discriminator_optimizer.step()

            # 累积损失
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    epoch_loss[k] += v.item()
                else:
                    epoch_loss[k] += v

        # 平均损失
        for k in epoch_loss:
            epoch_loss[k] /= num_batches

        # 测试模型
        source_acc = trainer.test(source_test_loader)
        target_acc = trainer.test(target_test_loader)

        # 打印进度
        print(f"Epoch [{epoch + 1}/{config.epochs}] - "
              f"Loss: {epoch_loss['total_loss']:.4f} - "
              f"Cls Loss: {epoch_loss['cls_loss']:.4f} - "
              f"Adv Loss: {epoch_loss['adv_loss']:.4f} - "
              f"Pseudo Loss: {epoch_loss['pseudo_loss']:.4f} - "
              f"Pseudo Ratio: {epoch_loss['pseudo_ratio']:.2f} - "
              f"Source Acc: {source_acc:.2f}% - "
              f"Target Acc: {target_acc:.2f}%")

        # 保存最佳模型
        if target_acc > best_acc:
            best_acc = target_acc
            model_path = os.path.join(result_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

        # 可视化特征（每5个epoch）
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            visualize_features(
                model, source_test_loader, target_test_loader, device,
                os.path.join(result_dir, f'features_epoch_{epoch + 1}.png')
            )

    # 加载最佳模型进行最终测试
    model.load_state_dict(torch.load(os.path.join(result_dir, 'best_model.pth')))
    final_target_acc = trainer.test(target_test_loader)

    print(f"Training completed. Best target accuracy: {final_target_acc:.2f}%")

    return final_target_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TFMADA Training')
    parser.add_argument('--dataset', type=str, default='MFD', choices=['MFD', 'HAR', 'SSC'],
                        help='Dataset (MFD, HAR, SSC)')
    parser.add_argument('--source_domain', type=str, required=True,
                        help='Source domain')
    parser.add_argument('--target_domain', type=str, required=True,
                        help='Target domain')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    start_time = time.time()
    train(args)
    elapsed_time = time.time() - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")