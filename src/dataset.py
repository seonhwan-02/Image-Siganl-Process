import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class WaterFringeDataset(Dataset):
    """
    U-Net 이진 물 감지(Water Detection) 학습용 Dataset.
    
    [설계 의도]
    - 물과 숲은 색이 비슷해서 SAM만으로 구분이 어려움.
    - 따라서 U-Net이 먼저 "물인가 아닌가"를 이진 분류(binary)로 판단.
    - 이 확률맵(water probability map)을 predict_sam_supervised_clean.py에서
      SAM + 텍스처 분석과 결합해 물/숲/모래사장/기타 4분류를 수행.
    
    [마스크 생성 방법]
    1. JSON 레이블이 있으면: CODE=50(수역) 폴리곤을 마스크에 버닝(1로 설정)
    2. JSON이 없어도: RGB 색상 + 로컬 질감(texture) 분석으로 물 자동 감지
       - 어두운 RGB + 파랑 계열 + 질감 매끄러움(Local Std < 5.0)
    -> 최종 마스크: 물=1, 비물=0 (이진 float 마스크)
    """
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.upper().endswith((".TIF", ".TIFF"))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, f"{base_name}.json")

        # --- 1. 이미지 로드 ---
        try:
            with rasterio.open(img_path) as src:
                image = src.read([1, 2, 3])
                image = np.moveaxis(image, 0, -1).astype(np.uint8)
                geo_transform = src.transform
                height, width = src.height, src.width
        except Exception as e:
            print(f"[SKIP] Unreadable file: {img_name}")
            next_idx = (idx + 1) % len(self)
            if next_idx == idx:
                return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512)
            return self.__getitem__(next_idx)

        # --- 2. 이진 물 마스크 생성 ---
        water_mask = np.zeros((height, width), dtype=np.uint8)

        # 2a. JSON 레이블 기반: CODE=50 (수역) 폴리곤 버닝
        if os.path.exists(label_path):
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                features = meta.get("annotation", {}).get("features", [])
                water_shapes = []
                for feat in features:
                    geo = feat.get("geometry", {})
                    code = str(feat.get("properties", {}).get("CODE", "")).strip()
                    # CODE=50: 수역 (물), CODE=20/40/511: 수변초목도 포함
                    if code in ["50", "20", "40", "511"] and geo.get("coordinates"):
                        try:
                            poly_obj = shape(geo)
                            water_shapes.append((poly_obj, 1))
                        except Exception:
                            continue
                if water_shapes:
                    water_mask = rasterize(
                        water_shapes,
                        out_shape=(height, width),
                        transform=geo_transform,
                        fill=0,
                        dtype=np.uint8
                    )
            except Exception as e:
                print(f"[WARN] Label load failed for {img_name}: {str(e)[:80]}")

        # 2b. 색상 + 질감 기반 자동 물 감지 (JSON 없거나 보완용)
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mean_img = cv2.boxFilter(gray, -1, (31, 31))
        sq_mean  = cv2.boxFilter(gray**2, -1, (31, 31))
        local_std = np.sqrt(np.clip(sq_mean - mean_img**2, 0, None))

        # 물 조건: 어둡고 + 파랑/초록 계열 + 질감 매끄러움
        color_water = (r < 85) & (g < 110) & (b < 110) & (g > r) & (local_std < 5.0)
        color_water = color_water.astype(np.uint8)

        # 모폴로지 노이즈 제거
        kernel = np.ones((9, 9), np.uint8)
        color_water = cv2.morphologyEx(color_water, cv2.MORPH_OPEN, kernel)

        # JSON 마스크와 색상 마스크를 OR로 합침 (둘 다 활용)
        water_mask = np.clip(water_mask + color_water, 0, 1).astype(np.uint8)

        # --- 3. Albumentations 변환 적용 ---
        if self.transform:
            augmented = self.transform(image=image, mask=water_mask)
            image_t = augmented["image"]
            mask_t  = augmented["mask"]
        else:
            base_transform = A.Compose([
                A.Resize(512, 512),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            augmented = base_transform(image=image, mask=water_mask)
            image_t = augmented["image"]
            mask_t  = augmented["mask"]

        # 마스크를 float32, (1, H, W) 형태로 변환 (BCEWithLogitsLoss 용)
        mask_t = mask_t.float().unsqueeze(0)
        return image_t, mask_t


if __name__ == "__main__":
    print("Dataset pipeline check...")
    ds = WaterFringeDataset(
        img_dir="data/processed/train/images",
        label_dir="data/processed/train/labels"
    )
    if len(ds) > 0:
        img, msk = ds[0]
        print(f"[OK] Image shape: {img.shape}, Mask shape: {msk.shape}")
        print(f"     Mask unique values: {msk.unique().tolist()}")
    else:
        print("[ERROR] No images found.")