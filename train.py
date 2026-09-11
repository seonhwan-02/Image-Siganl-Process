"""
train.py - U-Net 이진 물 감지(Water Binary Segmentation) 학습 스크립트

[설계 의도]
  물과 숲은 항공 영상에서 색이 비슷해 SAM만으로 구분하기 어렵습니다.
  따라서 U-Net을 먼저 사용해 "이 픽셀이 물(수변)인가 아닌가"를 이진 분류합니다.
  학습된 가중치(best_unet_model.pth)는 predict_sam_supervised_clean.py에서
  SAM + 텍스처 분석과 결합되어 최종 4분류(물/숲/모래사장/기타)를 수행합니다.

[파이프라인]
  train.py -> best_unet_model.pth (water probability map)
      |
      v
  predict_sam_supervised_clean.py
      -> 물(파랑) / 숲(초록) / 모래사장·나지(주황) / 기타(회색) 4분류 출력
"""
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.dataset import WaterFringeDataset
from src.model import UNet


def main():
    # ===== 설정 =====
    BATCH_SIZE    = 2
    EPOCHS        = 30
    LEARNING_RATE = 0.0001
    IMG_SIZE      = 512          
    RESUME_CHECKPOINT = "checkpoint/latest_unet_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # ===== 데이터 변환 =====
    train_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # ===== 데이터셋 / 데이터로더 =====
    train_dataset = WaterFringeDataset(
        "data/processed/train/images",
        "data/processed/train/labels",
        transform=train_transform
    )
    val_dataset = WaterFringeDataset(
        "data/processed/val/images",
        "data/processed/val/labels",
        transform=val_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Train: {len(train_dataset)} images / Val: {len(val_dataset)} images")

    # ===== 모델 (이진 분류: num_classes=1) =====
    model = UNet(in_channels=3, num_classes=1).to(device)

    # BCEWithLogitsLoss: 이진 물/비물 분류에 적합
    # pos_weight: 물 픽셀이 적으므로 양성(물) 클래스에 가중치 부여
    pos_weight = torch.tensor([3.0]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ===== 체크포인트 재개 =====
    best_val_loss = float("inf")
    START_EPOCH   = 0

    if os.path.exists(RESUME_CHECKPOINT):
        try:
            ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
            # 이전 6-class 체크포인트와 호환되지 않을 수 있으므로 strict=False
            incompatible = model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            if ckpt.get("optimizer_state_dict"):
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            START_EPOCH   = ckpt.get("epoch", -1) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"[OK] Resumed from {RESUME_CHECKPOINT} (Epoch {START_EPOCH})")
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(f"     Note: output layer re-initialized for num_classes=1")
                START_EPOCH   = 0
                best_val_loss = float("inf")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}. Starting fresh.")
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")

    os.makedirs("checkpoint", exist_ok=True)
    scaler = torch.cuda.amp.GradScaler()

    # ===== 학습 루프 =====
    print(f"\n[START] Training U-Net binary water detector ({START_EPOCH+1} ~ {EPOCHS} epochs)")
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        for images, masks in train_bar:
            images = images.to(device)
            masks  = masks.to(device).float()   # (B, 1, H, W) float

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)          # (B, 1, H, W) logits
                loss    = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_dataset)

        # ===== 검증 루프 =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device).float()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss    = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        avg_val_loss = val_loss / len(val_dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # CSV 로그 기록
        log_path   = "checkpoint/training_log.csv"
        file_exists = os.path.exists(log_path)
        with open(log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"])

        # 최신 체크포인트 저장 (재개용)
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss":        best_val_loss,
        }, RESUME_CHECKPOINT)

        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoint/best_unet_model.pth")
            print(f"  -> Best model saved! (Val Loss: {best_val_loss:.4f})")

    print("\n[DONE] Training complete! Use predict_sam_supervised_clean.py for 4-class prediction.")


if __name__ == "__main__":
    main()