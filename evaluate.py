"""
evaluate.py - U-Net 이진 물 감지기 검증셋 정량 평가 스크립트

계산 메트릭:
- IoU (Intersection over Union)
- Dice Coefficient (F1-Score)
- Precision (정밀도)
- Recall (재현율)
- OA (Overall Accuracy, 전체 정확도)
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from src.dataset import WaterFringeDataset
from src.model import UNet

def calculate_metrics(preds, targets, smooth=1e-6):
    """
    preds, targets: (B, 1, H, W) PyTorch Tensor (0 or 1)
    """
    preds = preds.byte()
    targets = targets.byte()

    # True Positives, False Positives, False Negatives, True Negatives
    tp = (preds & targets).sum().float().item()
    fp = (preds & ~targets).sum().float().item()
    fn = (~preds & targets).sum().float().item()
    tn = (~preds & ~targets).sum().float().item()

    # 계산
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    total_pixels = tp + fp + fn + tn
    oa = (tp + tn) / total_pixels if total_pixels > 0 else 0.0

    return iou, precision, recall, oa

def main():
    CHECKPOINT = "checkpoint/best_sam_unet_model.pth"
    IMG_DIR = "data/processed/val/images"
    LABEL_DIR = "data/processed/val/labels"
    WATER_THRESHOLD = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    if not os.path.exists(CHECKPOINT):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)

    # 데이터 로더 셋업 (배치 사이즈 4)
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_dataset = WaterFringeDataset(IMG_DIR, LABEL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    print(f"Validation Dataset Size: {len(val_dataset)} images")

    # 모델 로드
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print("[OK] Model weights loaded successfully.")

    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_oa = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device) # (B, 1, H, W)

            with torch.cuda.amp.autocast():
                outputs = model(images) # logits
                probs = torch.sigmoid(outputs)
                preds = (probs > WATER_THRESHOLD).float()

            for p, t in zip(preds, masks):
                iou, precision, recall, oa = calculate_metrics(p, t)
                total_iou += iou
                total_precision += precision
                total_recall += recall
                total_oa += oa
                count += 1

    avg_iou = total_iou / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count
    avg_oa = total_oa / count

    print("\n" + "="*40)
    print("         QUANTITATIVE EVALUATION RESULTS")
    print("="*40)
    print(f"- Evaluated Images: {count}")
    print(f"- Mean IoU ( 수역 ): {avg_iou*100:.2f}%")
    print(f"- Precision ( 정밀도 ): {avg_precision*100:.2f}%")
    print(f"- Recall ( 재현율 ): {avg_recall*100:.2f}%")
    print(f"- Overall Accuracy ( OA ): {avg_oa*100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
