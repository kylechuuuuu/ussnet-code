import os
import torch
import yaml
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
from PIL import Image
import torch.nn.functional as F
import numpy as np
from skimage import morphology

# Import the reseg model
from model.unet import Ubase

threshold = 0.5


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def pad_to_multiple(tensor, divisor):
    _, _, h, w = tensor.size()
    pad_h = (divisor - h % divisor) if h % divisor != 0 else 0
    pad_w = (divisor - w % divisor) if w % divisor != 0 else 0
    padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
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
    return (
        torch.tensor(correctness * 100),
        torch.tensor(completeness * 100),
        torch.tensor(quality * 100),
    )


def evaluate_segmentation(segmentation, gt_tensor):
    f1, precision, recall = computeF1(segmentation, gt_tensor)
    correctness, completeness, quality = computeTopo(segmentation, gt_tensor)
    return f1, precision, recall, correctness, completeness, quality


def process_images(input_folder, output_folder, model_path, gt_folder=None):
    net = Ubase().cuda()
    # net = torch.nn.DataParallel(net)  # Use multiple GPUs
    net.load_state_dict(torch.load(model_path))
    net.eval()

    transform = Compose(
        [
            ToTensor(),
        ]
    )

    os.makedirs(output_folder, exist_ok=True)

    f1_scores = []
    precisions = []
    recalls = []
    correctnesses = []
    completenesses = []
    qualities = []

    for img_name in sorted(os.listdir(input_folder)):
        if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, img_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).cuda()
            img_tensor, pad_h, pad_w = pad_to_multiple(img_tensor, 16)

            with torch.no_grad():
                segmentation_outputs = net(img_tensor)

            final_segmentation = segmentation_outputs

            final_segmentation = torch.sigmoid(final_segmentation)
            final_segmentation = (final_segmentation >= threshold).float()
            final_segmentation = final_segmentation[
                :, :, : -pad_h if pad_h > 0 else None, : -pad_w if pad_w > 0 else None
            ]

            if gt_folder is not None:
                # Check for mask file with the same extension
                gt_path = None
                for ext in (".png", ".jpg", ".jpeg"):
                    potential_gt_path = os.path.join(gt_folder, img_name.rsplit('.', 1)[0] + ext)
                    if os.path.exists(potential_gt_path):
                        gt_path = potential_gt_path
                        break

                if gt_path is None:
                    print(f"No matching GT found for {img_name}, skipping.")
                    continue

                gt_tensor = (
                    transform(Image.open(gt_path).convert("L")).unsqueeze(0).cuda()
                )
                gt_tensor = (gt_tensor >= 0.5).float()

                f1, precision, recall, correctness, completeness, quality = (
                    evaluate_segmentation(final_segmentation, gt_tensor)
                )

                f1_scores.append(f1)
                precisions.append(precision)
                recalls.append(recall)
                correctnesses.append(correctness)
                completenesses.append(completeness)
                qualities.append(quality)

                print(
                    f"Image: {img_name}| F1: {f1:.2f}| Precision: {precision:.2f}| Recall: {recall:.2f}| Correctness: {correctness:.2f}| Completeness: {completeness:.2f}| Quality: {quality:.2f}"
                )

            save_image(final_segmentation, os.path.join(output_folder, f"{img_name}"))

    if gt_folder is not None:
        print("\nOverall Metrics:")
        print(f"Average F1 Score: {torch.mean(torch.stack(f1_scores)):.2f}")
        print(f"Average Precision: {torch.mean(torch.stack(precisions)):.2f}")
        print(f"Average Recall: {torch.mean(torch.stack(recalls)):.2f}")
        print(f"Average Correctness: {torch.mean(torch.stack(correctnesses)):.2f}")
        print(f"Average Completeness: {torch.mean(torch.stack(completenesses)):.2f}")
        print(f"Average Quality: {torch.mean(torch.stack(qualities)):.2f}")

        # save metrics to file as well
        if output_folder is not None:
            with open(os.path.join(output_folder, "metrics.txt"), "w") as f:
                f.write(f"model: {model_path}\n")
                f.write(f"input: {input_folder}\n")
                f.write(f"threshold: {threshold}\n")
                f.write(f"n_images: {len(f1_scores)}\n")
                f.write(f"Average F1 Score: {torch.mean(torch.stack(f1_scores)):.4f}\n")
                f.write(f"Average Precision: {torch.mean(torch.stack(precisions)):.4f}\n")
                f.write(f"Average Recall: {torch.mean(torch.stack(recalls)):.4f}\n")
                f.write(f"Average Correctness: {torch.mean(torch.stack(correctnesses)):.4f}\n")
                f.write(f"Average Completeness: {torch.mean(torch.stack(completenesses)):.4f}\n")
                f.write(f"Average Quality: {torch.mean(torch.stack(qualities)):.4f}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.environ.get("TRAIN_CONFIG", "./options/training.yaml")
    )
    parser.add_argument("--gpu", default=os.environ.get("TRAIN_GPU", "0"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    opt = load_config(args.config)
    exp = opt["train"]["experiments"]
    input_folder = opt["datasets"]["val"]["data_image"]  # 测试图片文件夹路径
    gt_folder = opt["datasets"]["val"]["data_annotation_mask"]  # GT图像文件夹路径
    output_folder = args.output or os.path.join("results", exp)  # 输出分割图片文件夹路径
    model_path = args.model or os.path.join("weight", exp, "best_model.pth")  # 模型文件路径

    print(f"config={args.config}")
    print(f"model={model_path}")
    print(f"input={input_folder}")
    print(f"output={output_folder}")
    print("Begin processing images...")
    process_images(input_folder, output_folder, model_path, gt_folder=gt_folder)
    print("Done.")
