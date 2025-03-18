import torch
from sklearn import metrics
from torch import Tensor
import numpy as np


def diceCoeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    input = (input > 0.5).float()  # 将概率转换为二值掩码
    target = (target > 0.5).float()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        dice = 0
        for i in range(input.shape[0]):
            dice += diceCoeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def mean_iou(input: Tensor, target: Tensor, num_classes=4, epsilon=1e-6):
    assert input.size() == target.size()
    iou_per_class = []

    # 对于每一个类别计算IoU
    for cls in range(num_classes):
        # 计算每个类的交集（TP）
        intersection = torch.sum((input == cls) & (target == cls))
        # 计算每个类的并集
        union = torch.sum(input == cls) + torch.sum(target == cls) - intersection

        # 避免除零
        iou = (intersection + epsilon) / (union + epsilon)
        iou_per_class.append(iou)

    # 计算所有类别的平均IoU
    return torch.mean(torch.tensor(iou_per_class))


def compute_sensitivity(input: Tensor, target: Tensor, epsilon=1e-6):
    assert input.size() == target.size()
    if input.dim() == 2:
        input = (input > 0.5).float()
        input = input.cpu().numpy()
        target = target.cpu().numpy()
        true_positive = np.sum((input == 1) & (target == 1))
        actual_positive = np.sum(target == 1)
        sensitivity = true_positive / (actual_positive + epsilon)
        return sensitivity

    else:
        sensitivity = 0
        for i in range(input.shape[0]):
            sensitivity += compute_sensitivity(input[i, ...], target[i, ...])
        return sensitivity / input.shape[0]


def compute_acc(input, target):
    if input.dim() > 2:
        # 如果输入维度大于2（可能包含多个样本），则遍历每个样本计算准确率
        acc = 0
        for i in range(input.shape[0]):
            acc += compute_acc(input[i, ...], target[i, ...])  # 递归计算每个样本的准确率
        return acc / input.shape[0]
    else:
        input = input.cpu().numpy()
        target = target.cpu().numpy()
        # 计算准确率
        correct = (input == target).sum()
        total = input.size
        acc = correct / total
        return acc

def compute_acc_nobg(input, target):
    if input.dim() > 2:
        # 如果输入维度大于2（可能包含多个样本），则遍历每个样本计算准确率
        acc = 0
        for i in range(input.shape[0]):
            acc += compute_acc(input[i, ...], target[i, ...])  # 递归计算每个样本的准确率
        return acc / input.shape[0]
    else:
        input = input.cpu().numpy()
        target = target.cpu().numpy()
        # 创建非背景像素的掩码
        non_background_mask = target != 0
        # 只计算非背景像素的准确率
        correct = ((input == target) & non_background_mask).sum()
        total = non_background_mask.sum()
        acc = correct / total if total > 0 else 0
        return acc

def compute_kappa(input: Tensor, target: Tensor, num_classes=4, epsilon=1e-6):
    assert input.size() == target.size()
    hw = int(input.shape[0])  # 假设宽度和高度为 512
    po = 0.0  # 观察到的一致性，确保为浮动类型
    pe = 0.0  # 预期一致性，确保为浮动类型

    # 计算 Po 和 Pe
    for cls in range(num_classes):
        # 计算每个类的交集（TP）
        intersection = torch.sum((input == cls) & (target == cls))
        # 计算每个类的预测概率和标签概率
        a = torch.sum(input == cls)
        b = torch.sum(target == cls)
        po += intersection
        pe += (a * b)

    # 计算Po和Pe
    po /= (hw * hw)  # 将po转换为Float类型以便于计算
    pe = (pe + epsilon) / (hw * hw * hw * hw)  # 使用正则化避免分母为0

    # 计算Kappa值
    return (po - pe + epsilon) / (1 - pe + epsilon)


def compute_specificity(input: Tensor, target: Tensor, epsilon=1e-6):
    assert input.size() == target.size()
    if input.dim() == 2:
        input = (input > 0.5).float()
        specificity = 0
        true_negatives = torch.sum((input == 0) * (target == 0)).float()
        false_positives = torch.sum((input == 1) * (target == 0)).float()
        specificity = true_negatives / (true_negatives + false_positives + epsilon)
        return specificity
    else:
        specificity = 0
        for i in range(input.shape[0]):
            specificity += compute_specificity(input[i, ...], target[i, ...])
        return specificity / input.shape[0]
