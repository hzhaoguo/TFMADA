import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def accuracy(y_true, y_pred):
    """计算准确率"""
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred, average='macro'):
    """计算精确率"""
    return precision_score(y_true, y_pred, average=average)


def recall(y_true, y_pred, average='macro'):
    """计算召回率"""
    return recall_score(y_true, y_pred, average=average)


def f1(y_true, y_pred, average='macro'):
    """计算F1分数"""
    return f1_score(y_true, y_pred, average=average)


def calculate_metrics(model, data_loader, device):
    """计算模型在给定数据加载器上的指标"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)

            # 模型预测
            pred, _, _, _, _ = model(data)
            _, predicted = torch.max(pred.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    acc = accuracy(all_labels, all_preds)
    prec = precision(all_labels, all_preds)
    rec = recall(all_labels, all_preds)
    f1_score = f1(all_labels, all_preds)

    # 计算每个类别的准确率
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1_score,
        'per_class_accuracy': per_class_acc
    }

    return metrics