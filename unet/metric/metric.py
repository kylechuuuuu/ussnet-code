import os
import torch
import cv2
import numpy as np
from skimage import morphology

# 定义计算 F1 分数的函数
def computeF1(pred, gt):
    """
    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: segmentation metric
    """
    # 1, h, w
    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score * 100, precision * 100, recall * 100

# 定义计算 Topo 指标的函数
def computeTopo(pred, gt):
    """
    :param pred: prediction, tensor
    :param gt: gt, tensor
    :return: Topo metric
    """
    pred = pred[0].detach().cpu().numpy().astype(int)  # float data does not support bit_and and bit_or
    gt = gt[0].detach().cpu().numpy().astype(int)
    
    pred = morphology.skeletonize(pred >= 0.5)  # 骨架提取函数，输入输出都是二值图像
    gt = morphology.skeletonize(gt >= 0.5)

    cor_intersection = gt & pred
    com_intersection = gt & pred

    cor_tp = np.sum(cor_intersection)
    com_tp = np.sum(com_intersection)

    sk_pred_sum = np.sum(pred)
    sk_gt_sum = np.sum(gt)

    smooth = 1e-7
    correctness = cor_tp / (sk_pred_sum + smooth)
    completeness = com_tp / (sk_gt_sum + smooth)
    quality = cor_tp / (sk_pred_sum + sk_gt_sum - com_tp + smooth)

    return torch.tensor(correctness * 100), torch.tensor(completeness * 100), torch.tensor(quality * 100)

def evaluate_folder(pred_folder, gt_folder):
    """
    计算文件夹中所有图像的 F1, Precision, Recall, Quality, Correctness, Completeness 指标

    :param pred_folder: 预测图像文件夹路径
    :param gt_folder: 目标（Ground Truth）图像文件夹路径
    """
    pred_files = sorted(os.listdir(pred_folder))
    gt_files = sorted(os.listdir(gt_folder))

    assert len(pred_files) == len(gt_files), "预测文件夹和目标文件夹中的图像数量不匹配"

    f1_scores = []
    precisions = []
    recalls = []
    correctnesses = []
    completenesses = []
    qualities = []

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, gt_file)

        # 读取图像并转换为张量
        pred_image = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像是二值图像
        _, pred_binary = cv2.threshold(pred_image, 127, 255, cv2.THRESH_BINARY)
        _, gt_binary = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)

        pred_tensor = torch.tensor(pred_binary).unsqueeze(0).float() / 255
        gt_tensor = torch.tensor(gt_binary).unsqueeze(0).float() / 255

        # 计算各项指标
        f1, precision, recall = computeF1(pred_tensor, gt_tensor)
        correctness, completeness, quality = computeTopo(pred_tensor, gt_tensor)

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        correctnesses.append(correctness)
        completenesses.append(completeness)
        qualities.append(quality)

        print(f"Image: {pred_file} | F1: {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | Correctness: {correctness:.2f} | Completeness: {completeness:.2f} | Quality: {quality:.2f}")

    # 汇总输出所有图像的平均指标
    print("\nOverall Metrics:")
    print(f"Average F1 Score: {torch.mean(torch.stack(f1_scores)):.2f}")
    print(f"Average Precision: {torch.mean(torch.stack(precisions)):.2f}")
    print(f"Average Recall: {torch.mean(torch.stack(recalls)):.2f}")
    print(f"Average Correctness: {torch.mean(torch.stack(correctnesses)):.2f}")
    print(f"Average Completeness: {torch.mean(torch.stack(completenesses)):.2f}")
    print(f"Average Quality: {torch.mean(torch.stack(qualities)):.2f}")

if __name__ == "__main__":
    # 设置预测图像和目标图像文件夹路径
    pred_folder = "test1"  # 替换为你的预测图像文件夹路径
    gt_folder = "mask2"  # 替换为你的目标图像文件夹路径
    
    evaluate_folder(pred_folder, gt_folder)
