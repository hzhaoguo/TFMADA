import argparse
import numpy as np
import os
import torch
from train import train


def run_cross_domain_experiments(args):
    """运行跨域实验"""
    results = {}

    # 获取域名称
    if args.dataset == 'MFD':
        domains = ['H', 'I', 'J', 'K']
    elif args.dataset == 'HAR':
        domains = ['A', 'B', 'C', 'D']
    elif args.dataset == 'SSC':
        domains = ['EDF', 'SH1', 'SH2']
    else:
        raise ValueError(f"未知数据集: {args.dataset}")

    # 运行所有跨域组合
    for source in domains:
        for target in domains:
            if source != target:  # 不同域间才进行迁移
                experiment_name = f"{source}->{target}"
                print(f"\n{'=' * 50}")
                print(f"开始实验: {experiment_name}")
                print(f"{'=' * 50}\n")

                # 设置参数
                args.source_domain = source
                args.target_domain = target

                # 运行训练
                accuracy = train(args)
                results[experiment_name] = accuracy

    # 打印所有结果
    print("\n\n最终结果汇总:")
    print(f"{'实验':15s} | {'准确率 (%)':10s}")
    print("-" * 30)

    total_acc = 0.0
    for exp_name, acc in results.items():
        print(f"{exp_name:15s} | {acc:.2f}")
        total_acc += acc

    avg_acc = total_acc / len(results)
    print(f"\n平均准确率: {avg_acc:.2f}%")

    # 保存结果
    result_dir = f"results/{args.dataset}/summary"
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, 'cross_domain_results.txt'), 'w') as f:
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"批次大小: {args.batch_size}, 学习率: {args.lr}, 轮次: {args.epochs}\n\n")
        f.write(f"{'实验':15s} | {'准确率 (%)':10s}\n")
        f.write("-" * 30 + "\n")

        for exp_name, acc in results.items():
            f.write(f"{exp_name:15s} | {acc:.2f}\n")

        f.write(f"\n平均准确率: {avg_acc:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description='TFMADA: 时间-频率多对抗自适应框架')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='MFD', choices=['MFD', 'HAR', 'SSC'],
                        help='数据集选择 (MFD, HAR, SSC)')
    parser.add_argument('--source_domain', type=str, default=None,
                        help='源域名称，如果为None，将运行所有跨域组合')
    parser.add_argument('--target_domain', type=str, default=None,
                        help='目标域名称，如果为None，将运行所有跨域组合')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 模型参数
    parser.add_argument('--cwt_scales', type=int, default=32,
                        help='连续小波变换的尺度数量')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='隐藏层维度')

    # 实验控制
    parser.add_argument('--run_all', action='store_true',
                        help='运行所有跨域组合实验')

    args = parser.parse_args()

    # 设置全局随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 运行实验
    if args.run_all or (args.source_domain is None or args.target_domain is None):
        run_cross_domain_experiments(args)
    else:
        train(args)


if __name__ == '__main__':
    main()