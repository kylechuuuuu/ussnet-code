import os
import time
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from datasets.ReSegDataset import ReSegDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from model.unet import Ubase
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

threshold = 0.5


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def markov_chain_loss(segmentation_outputs, mask):
    return nn.BCEWithLogitsLoss()(segmentation_outputs, mask)


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
    fp = ((1 - gt) * pred).sum().to(torch.float32)
    fn = (gt * (1 - pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_score * 100, precision * 100, recall * 100


def train_epoch(model, loader, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0.0
    for i, (image, mask) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        image, mask = image.to(device), mask.to(device)
        optimizer.zero_grad()
        segmentation_outputs = model(image)
        loss = markov_chain_loss(segmentation_outputs, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + i)
            # logger.info(f'Epoch [{epoch}], Step [{i}/{len(loader)}], Loss: {loss.item():.4f}')
    average_loss = total_loss / len(loader)
    logger.info(f"Epoch [{epoch}] completed. Average Loss: {average_loss:.4f}")
    return average_loss


def validate(model, val_data_path, device, writer, epoch):
    model.eval()
    transform = Compose(
        [
            ToTensor(),
        ]
    )

    f1_scores = []
    precisions = []
    recalls = []

    img_extensions = ['.png', '.jpg', '.jpeg']

    with torch.no_grad():
        for img_name in os.listdir(val_data_path["images"]):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(val_data_path["images"], img_name)

                mask_name = os.path.splitext(img_name)[0]
                mask_path = None
                for ext in img_extensions:
                    potential_path = os.path.join(val_data_path["masks"], mask_name + ext)
                    if os.path.exists(potential_path):
                        mask_path = potential_path
                        break

                if mask_path is None:
                    logger.error(f"No mask file found for image: {img_name}")
                    continue

                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                image_tensor = transform(image).unsqueeze(0).to(device)
                image_tensor, pad_h, pad_w = pad_to_multiple(image_tensor, 16)
                mask_tensor = transform(mask).unsqueeze(0).to(device)

                segmentation_outputs = model(image_tensor)
                final_segmentation = segmentation_outputs
                final_segmentation = torch.sigmoid(final_segmentation)

                preds = (final_segmentation >= threshold).float()
                preds = preds[
                    :,
                    :,
                    : -pad_h if pad_h > 0 else None,
                    : -pad_w if pad_w > 0 else None,
                ]

                f1, precision, recall = computeF1(preds, mask_tensor)

                f1_scores.append(f1)
                precisions.append(precision)
                recalls.append(recall)

                writer.add_image(f"Validation/Prediction/{img_name}", preds[0], epoch)
                # writer.add_image(f"Validation/Mask/{img_name}", mask_tensor[0], epoch)

    average_f1 = torch.mean(torch.stack(f1_scores)).item()
    average_precision = torch.mean(torch.stack(precisions)).item()
    average_recall = torch.mean(torch.stack(recalls)).item()

    writer.add_scalar("F1/val", average_f1, epoch)
    writer.add_scalar("Precision/val", average_precision, epoch)
    writer.add_scalar("Recall/val", average_recall, epoch)

    logger.info(f"\nValidation Results - Epoch: {epoch}")
    logger.info(f"Average F1 Score: {average_f1:.2f}")
    logger.info(f"Average Precision: {average_precision:.2f}")
    logger.info(f"Average Recall: {average_recall:.2f}")

    return average_f1



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=os.environ.get("TRAIN_CONFIG", "./options/training.yaml")
    )
    parser.add_argument("--gpu", default=os.environ.get("TRAIN_GPU", "0"))
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    opt = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = opt["train"]["epochs"]
    logger.info(
        f"config={args.config} experiments={opt['train']['experiments']} gpu={args.gpu} device={device}"
    )

    torch.manual_seed(opt["train"]["seed"])
    torch.cuda.manual_seed_all(opt["train"]["seed"])

    trainset = ReSegDataset(
        data_images=opt["datasets"]["train"]["data_image"],
        data_annotation_mask=opt["datasets"]["train"]["data_annotation_mask"],
        train_size=opt["train"]["train_size"],
    )

    train_loader = DataLoader(
        trainset,
        batch_size=opt["train"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    net = Ubase().to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        net = nn.DataParallel(net)

    optimizer = optim.AdamW(
        net.parameters(),
        lr=opt["train"]["lr"],
        weight_decay=opt["train"]["weight_decay"],
        betas=(0.9, 0.999),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=opt["train"]["epochs"], eta_min=1e-7)

    experiment_dir = os.path.join("weight", opt["train"]["experiments"])
    os.makedirs(experiment_dir, exist_ok=True)
    results_file_path = os.path.join(experiment_dir, "training_results.txt")

    writer = SummaryWriter(log_dir=os.path.join("./tf_logs", opt["train"]["experiments"]))
    start_time = time.time()

    best_f1 = -float("inf")

    with open(results_file_path, "a") as file:
        for epoch in range(1, total_epochs + 1):
            epoch_start_time = time.time()

            train_loss = train_epoch(
                net, train_loader, optimizer, device, writer, epoch
            )

            checkpoint_path = os.path.join(
                experiment_dir, f"model_epoch_{epoch + 1}.pth"
            )
            # torch.save(net.state_dict(), checkpoint_path)

            do_validation = (epoch + 1) % opt["train"]["val_interval_epochs"] == 0 or (
                epoch + 1
            ) == total_epochs

            if do_validation:
                val_data_path = {
                    "images": opt["datasets"]["val"]["data_image"],
                    "masks": opt["datasets"]["val"]["data_annotation_mask"],
                }
                average_f1 = validate(net, val_data_path, device, writer, epoch)

                if average_f1 > best_f1:
                    best_f1 = average_f1
                    best_model_path = os.path.join(experiment_dir, "best_model.pth")
                    torch.save(net.state_dict(), best_model_path)
                    logger.info(
                        f"New best F1 score: {average_f1:.4f}. Model saved to {best_model_path}"
                    )

                file.write(f"Epoch: {epoch}\n")
                file.write(f"Average F1 Score: {average_f1:.2f}\n")
                file.write("-" * 30 + "\n")

            scheduler.step()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            remaining_time = (total_epochs - epoch - 1) * epoch_duration
            hours, remainder = divmod(remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            logger.info(
                f"Estimated Time Remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s"
            )

    writer.close()

    total_training_time = time.time() - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Total Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")


if __name__ == "__main__":
    main()
