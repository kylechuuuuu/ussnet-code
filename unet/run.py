import os
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F
import numpy as np
from skimage import morphology
import glob
import traceback

## model
from model.reseg6 import reseg

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def pad_to_multiple(tensor, divisor):
    _, _, h, w = tensor.size()
    pad_h = (divisor - h % divisor) if h % divisor != 0 else 0
    pad_w = (divisor - w % divisor) if w % divisor != 0 else 0
    padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded_tensor, pad_h, pad_w

def computeF1(pred, gt):
    pred = pred.float()
    gt = gt.float()

    tp = (gt * pred).sum().to(torch.float32)
    tn = ((1 - gt) * (1 - pred)).sum().to(torch.float32)
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1_score * 100, precision * 100, recall * 100

def computeTopo(pred, gt):
    pred = pred[0].detach().cpu().numpy().astype(int)
    gt = gt[0].detach().cpu().numpy().astype(int)
    pred = morphology.skeletonize(pred >= 0.5)
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

def evaluate_segmentation(segmentation, gt_tensor):
    f1, precision, recall = computeF1(segmentation, gt_tensor)
    correctness, completeness, quality = computeTopo(segmentation, gt_tensor)
    return f1, precision, recall, correctness, completeness, quality

def process_images(input_folder, model_path, gt_folder):
    net = reseg().cuda()
    # net = torch.nn.DataParallel(net)  # Use multiple GPUs
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    transform = Compose([ToTensor()])
    
    f1_scores_segmentation = []
    precisions_segmentation = []
    recalls_segmentation = []
    correctnesses_segmentation = []
    completenesses_segmentation = []
    qualities_segmentation = []

    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).cuda()
            img_tensor, pad_h, pad_w = pad_to_multiple(img_tensor, 16)
            
            with torch.no_grad():
                # Get all segmentation outputs from the model
                segmentation_outputs = net(img_tensor)
            
            # Use the last segmentation output for evaluation
            final_segmentation = segmentation_outputs[-1]
            
            # Apply sigmoid and threshold to the final segmentation output
            final_segmentation = torch.sigmoid(final_segmentation)
            final_segmentation = (final_segmentation >= 0.5).float()
            final_segmentation = final_segmentation[:, :, :-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]

            gt_path = os.path.join(gt_folder, img_name)
            gt_tensor = transform(Image.open(gt_path).convert('L')).unsqueeze(0).cuda()
            gt_tensor = (gt_tensor >= 0.5).float()
            
            # Evaluate segmentation metrics
            f1_seg, precision_seg, recall_seg, correctness_seg, completeness_seg, quality_seg = evaluate_segmentation(final_segmentation, gt_tensor)
            f1_scores_segmentation.append(f1_seg)
            precisions_segmentation.append(precision_seg)
            recalls_segmentation.append(recall_seg)
            correctnesses_segmentation.append(correctness_seg)
            completenesses_segmentation.append(completeness_seg)
            qualities_segmentation.append(quality_seg)

    avg_f1_seg = torch.mean(torch.stack(f1_scores_segmentation))
    avg_precision_seg = torch.mean(torch.stack(precisions_segmentation))
    avg_recall_seg = torch.mean(torch.stack(recalls_segmentation))
    avg_correctness_seg = torch.mean(torch.stack(correctnesses_segmentation))
    avg_completeness_seg = torch.mean(torch.stack(completenesses_segmentation))
    avg_quality_seg = torch.mean(torch.stack(qualities_segmentation))

    return (avg_f1_seg, avg_precision_seg, avg_recall_seg, avg_correctness_seg, avg_completeness_seg, avg_quality_seg)

def evaluate_models(weight_folder, input_folder, gt_folder, output_file):
    model_files = glob.glob(os.path.join(weight_folder, '*.pth'))
    results = []

    if not model_files:
        print(f"No .pth files found in {weight_folder}")
        return

    try:
        with open(output_file, 'w') as f:
            print(f"Opening file for writing: {output_file}")
            f.write("Model File | Avg F1 (Segmentation) | Avg Precision (Segmentation) | Avg Recall (Segmentation) | Avg Correctness (Segmentation) | Avg Completeness (Segmentation) | Avg Quality (Segmentation)\n")
            f.write("-" * 180 + "\n")

            for model_path in model_files:
                print(f"Evaluating {model_path}...")
                try:
                    (avg_f1_seg, avg_precision_seg, avg_recall_seg, avg_correctness_seg, 
                     avg_completeness_seg, avg_quality_seg) = process_images(input_folder, model_path, gt_folder)
                    
                    result_line = f"{os.path.basename(model_path)} | {avg_f1_seg:.2f} | {avg_precision_seg:.2f} | {avg_recall_seg:.2f} | {avg_correctness_seg:.2f} | {avg_completeness_seg:.2f} | {avg_quality_seg:.2f}\n"
                    print(f"Writing result: {result_line.strip()}")
                    f.write(result_line)
                    f.flush()  # Ensure immediate write to file
                except Exception as e:
                    print(f"Error processing {model_path}: {str(e)}")
                    traceback.print_exc()
    
        if results:
            best_model = max(results, key=lambda x: x[1])  # Sort by F1 score
            print(f"\nBest model: {os.path.basename(best_model[0])}")
            print(f"Best F1 Score: {best_model[1]:.2f}")
        else:
            print("No results were obtained.")
    except Exception as e:
        print(f"Error writing to file {output_file}: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    weight_folder = 'weight/test23/'
    input_folder = 'datasets/DRIVE/val/image'
    gt_folder = 'datasets/DRIVE/val/annotation_mask'
    output_file = 'model_evaluation_results.txt'

    print(f"Starting evaluation...")
    print(f"Weight folder: {weight_folder}")
    print(f"Input folder: {input_folder}")
    print(f"Ground truth folder: {gt_folder}")
    print(f"Output file: {output_file}")

    evaluate_models(weight_folder, input_folder, gt_folder, output_file)
    
    if os.path.exists(output_file):
        print(f"Results have been saved to {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print(f"Error: {output_file} was not created.")
