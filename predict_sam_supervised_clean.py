"""
predict_sam_supervised_clean.py - Water + Forest 동시 감지 및 Clean Single 경계선 시각화

- 물(Water)  : U-Net (질감 기반 학습 모델) → 파란 경계선 (단일 연속 외곽선)
- 숲(Forest) : 질감 거칠기 + 초록 채널 우세 → 초록 경계선 (단일 연속 외곽선)
- 마스크 없이 경계선(Contour)만 원본 위에 표시
- 3번째 패널에 단일 숲 경계선과 단일 물 경계선만 깔끔하게 노란/초록 및 노란/파란 이중 경계선으로 오버레이

사용법:
  python predict_sam_supervised_clean.py AP_HR_2021_0179_01
"""

import os
import sys
import torch
import numpy as np
import cv2
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import shutil
from scipy import ndimage

sys.path.append(os.path.abspath('.'))
from src.model import UNet

# ── 설정 ─────────────────────────────────────────────────────────
IMG_SIZE     = 512
CHECKPOINT   = "checkpoint/best_sam_unet_model.pth"
ARTIFACT_DIR = r"C:\Users\kimse\.gemini\antigravity\brain\831c6100-a054-4b10-9032-e60e39028194"

# 물 판별 임계값
WATER_THRESHOLD = 0.5

# 숲 판별 임계값 (질감 기반)
FOREST_STD_MIN      = 6.0
FOREST_GREEN_RATIO  = 0.95
FOREST_MIN_AREA_PX  = 500

def find_image(filename):
    for split in ["val", "train"]:
        p = f"data/processed/{split}/images/{filename}"
        if os.path.exists(p):
            return p
    return None

def compute_texture_map(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    def local_std(g, k):
        m  = cv2.boxFilter(g, -1, (k, k))
        s  = cv2.boxFilter(g ** 2, -1, (k, k))
        return np.sqrt(np.clip(s - m ** 2, 0, None))
    return local_std(gray, 7) * 0.4 + local_std(gray, 31) * 0.6

def detect_forest(image_rgb, texture_map):
    r = image_rgb[:, :, 0].astype(np.float32)
    g = image_rgb[:, :, 1].astype(np.float32)
    b = image_rgb[:, :, 2].astype(np.float32)
    cond_rough = texture_map > FOREST_STD_MIN
    cond_green = (g > r * FOREST_GREEN_RATIO) & (g > b * 0.90)
    forest_mask = (cond_rough & cond_green).astype(np.uint8)
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    forest_mask  = cv2.morphologyEx(forest_mask, cv2.MORPH_OPEN,  kernel_open)
    forest_mask  = cv2.morphologyEx(forest_mask, cv2.MORPH_CLOSE, kernel_close)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(forest_mask)
    filtered = np.zeros_like(forest_mask)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= FOREST_MIN_AREA_PX:
            filtered[labels == lbl] = 1
    return filtered.astype(bool)

def predict_single(image_path, model, device, transform):
    filename = os.path.basename(image_path)
    base     = os.path.splitext(filename)[0]
    print(f"\n[예측] {filename}")

    # 1. 이미지 로드
    with rasterio.open(image_path) as src:
        img = src.read([1, 2, 3])
        img = np.moveaxis(img, 0, -1)
    img_512 = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # 2. 질감 맵 계산
    texture_map = compute_texture_map(img_512)

    # 3. 물 마스크 예측 및 단일 영역 추출
    aug  = transform(image=img_512)
    inp  = aug['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.sigmoid(logits)[0, 0].cpu().numpy()
    water_mask = (probs > WATER_THRESHOLD).astype(np.uint8)
    water_mask = ndimage.binary_fill_holes(water_mask).astype(np.uint8)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN,  k7, iterations=2)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k7, iterations=3)
    water_mask = ndimage.binary_fill_holes(water_mask).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(water_mask)
    if n > 1:
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        water_mask = (lbl == biggest).astype(np.uint8)

    # 3.5 모래사장 및 도로 마스크 추출
    r_s = img_512[:,:,0].astype(float)
    g_s = img_512[:,:,1].astype(float)
    b_s = img_512[:,:,2].astype(float)
    brightness = (r_s + g_s + b_s) / 3.0
    
    # 아스팔트 조건: R≈G≈B (회색톤, diff<20), 밝기 40~160, 질감 매끄러움 (texture_map < 8.0)
    color_diff = np.abs(r_s - g_s) + np.abs(g_s - b_s) + np.abs(r_s - b_s)
    asphalt_cond = (color_diff < 20) & (brightness > 40) & (brightness < 160) & (texture_map < 8.0)

    sand_cond = (
        (brightness > 115) &
        (r_s > b_s * 0.85)
    ) | asphalt_cond
    sand_mask = sand_cond.astype(np.uint8)
    sand_mask[water_mask == 1] = 0
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) # 좁은 도로 보존을 위한 작은 커널
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    sand_mask = cv2.morphologyEx(sand_mask, cv2.MORPH_OPEN,  k3,  iterations=1)
    sand_mask = cv2.morphologyEx(sand_mask, cv2.MORPH_CLOSE, k15, iterations=1)

    # 4. 숲 마스크 및 단일 영역 추출 (모래사장 영역 제외)
    forest_mask = detect_forest(img_512, texture_map)
    forest_mask = forest_mask & ~water_mask.astype(bool)
    forest_mask = forest_mask & ~sand_mask.astype(bool)

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(forest_mask.astype(np.uint8))
    forest_single = np.zeros_like(forest_mask, dtype=np.uint8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = areas.max()
        keep_ids = [i+1 for i, a in enumerate(areas) if a >= max_area * 0.10]
        forest_clean = np.zeros_like(forest_mask, dtype=np.uint8)
        for kid in keep_ids:
            forest_clean[lbl == kid] = 1
        k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)) # 512 해상도 맞춤 커널크기 20
        forest_merged = cv2.dilate(forest_clean, k_merge, iterations=1)
        forest_merged[water_mask == 1] = 0
        forest_merged[sand_mask == 1] = 0
        forest_merged = ndimage.binary_fill_holes(forest_merged).astype(np.uint8)
        contours, _ = cv2.findContours(forest_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if contours:
            cv2.drawContours(forest_single, [contours[0]], -1, 1, -1)
    else:
        forest_single = forest_mask.copy().astype(np.uint8)
        contours, _ = cv2.findContours(forest_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

    forest_single[water_mask == 1] = 0
    forest_single[sand_mask == 1] = 0

    main_forest_contour = contours[0] if contours else None
    smooth_forest = None
    if main_forest_contour is not None:
        epsilon = 0.0008 * cv2.arcLength(main_forest_contour, True)
        smooth_forest = cv2.approxPolyDP(main_forest_contour, epsilon, True)

    water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    water_contours = sorted(water_contours, key=cv2.contourArea, reverse=True)
    main_water_contour = water_contours[0] if water_contours else None
    smooth_water = None
    if main_water_contour is not None:
        epsilon_w = 0.0005 * cv2.arcLength(main_water_contour, True)
        smooth_water = cv2.approxPolyDP(main_water_contour, epsilon_w, True)

    # 모래사장 경계 추출
    sand_contours, _ = cv2.findContours(sand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    smooth_sands = []
    for cnt in sand_contours:
        if cv2.contourArea(cnt) > 200: # 의미 있는 크기의 모래사장 구역만 추출
            epsilon_s = 0.001 * cv2.arcLength(cnt, True)
            smooth_cnt = cv2.approxPolyDP(cnt, epsilon_s, True)
            smooth_sands.append(smooth_cnt)

    # 5. 원본 위에 노란색 외곽 + 초록/파란색/주황색 내선 이중 경계선 그리기
    img_contour = img_512.copy()
    if smooth_forest is not None:
        cv2.drawContours(img_contour, [smooth_forest], -1, (255, 220, 0), 3) # 노란 외곽
        cv2.drawContours(img_contour, [smooth_forest], -1, (30, 180, 30), 2)  # 초록 내선
    if smooth_water is not None:
        cv2.drawContours(img_contour, [smooth_water], -1, (255, 220, 0), 3)  # 노란 외곽
        cv2.drawContours(img_contour, [smooth_water], -1, (30, 100, 200), 2) # 파란 내선
    for smooth_sand in smooth_sands:
        cv2.drawContours(img_contour, [smooth_sand], -1, (255, 220, 0), 3)   # 노란 외곽
        cv2.drawContours(img_contour, [smooth_sand], -1, (30, 140, 230), 2)  # 주황 내선 (모래사장)
    img_contour = img_contour.astype(np.float32) / 255.0

    # 6. 비율 계산
    total_px    = IMG_SIZE * IMG_SIZE
    water_pct   = water_mask.sum()        / total_px * 100
    forest_pct  = forest_single.sum()     / total_px * 100
    sand_pct    = sand_mask.sum()         / total_px * 100
    other_pct   = 100 - water_pct - forest_pct - sand_pct
    print(f"  [결과 비율] Water: {water_pct:.1f}%, Forest: {forest_pct:.1f}%, Sand: {sand_pct:.1f}%, Other: {other_pct:.1f}%")

    # 7. 시각화 피겨 렌더링
    img_show = img_512.astype(np.float32) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes:
        ax.set_facecolor('#0f0f1a')

    # 패널 1: 원본
    axes[0].imshow(img_show)
    axes[0].set_title("Original Image", fontsize=13, fontweight='bold', color='white', pad=12)
    axes[0].axis('off')

    # 패널 2: U-Net 물 확률 맵
    im = axes[1].imshow(probs, cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.set_label('Water Probability (0.0 ~ 1.0)', color='white', fontsize=8)
    axes[1].set_title("U-Net Water Map", fontsize=13, fontweight='bold', color='white', pad=12)
    axes[1].axis('off')

    # 패널 3: 원본 + 단일 클린 경계선
    axes[2].imshow(img_contour)
    w_line = mpatches.Patch(edgecolor=(0.12, 0.39, 0.78), facecolor='none', linewidth=2, label=f'Water Boundary  {water_pct:.1f}%')
    f_line = mpatches.Patch(edgecolor=(0.12, 0.70, 0.12), facecolor='none', linewidth=2, label=f'Forest Boundary {forest_pct:.1f}%')
    s_line = mpatches.Patch(edgecolor=(0.85, 0.60, 0.30), facecolor='none', linewidth=2, label=f'Sand/Road Boundary   {sand_pct:.1f}%')
    o_line = mpatches.Patch(facecolor='#555', label=f'Other           {other_pct:.1f}%')
    axes[2].legend(handles=[w_line, f_line, s_line, o_line], loc='lower right', fontsize=10, facecolor='#0f0f1a', edgecolor='white', labelcolor='white')
    axes[2].set_title("Water + Forest Boundary  (no mask)", fontsize=13, fontweight='bold', color='white', pad=12)
    axes[2].axis('off')

    fig.suptitle(
        f"SAM-SUPERVISED  |  Texture-Based  |  Water & Forest Detection  |  {filename}",
        fontsize=13, fontweight='bold', color='white', y=1.01
    )
    plt.tight_layout()

    # 이미지 저장
    os.makedirs("result_images", exist_ok=True)
    out_path = f"result_images/sam_pred_{base}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> 저장 완료: {out_path}")

    # 아티팩트 복사
    os.makedirs(os.path.join(ARTIFACT_DIR, "images"), exist_ok=True)
    shutil.copy(out_path, os.path.join(ARTIFACT_DIR, "images", f"sam_pred_{base}.png"))
    shutil.copy(out_path, os.path.join(ARTIFACT_DIR, f"sam_pred_{base}.png"))
    return water_pct, forest_pct, other_pct

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"예측 디바이스: {device}")

    if not os.path.exists(CHECKPOINT):
        print(f"[오류] 가중치 없음: {CHECKPOINT}")
        sys.exit(1)

    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print("[OK] U-Net 모델 로드 완료")

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    target = sys.argv[1] if len(sys.argv) > 1 else "AP_HR_2021_0179_01"
    fname = target if target.endswith('.tif') else f"{target}.tif"
    img_path = find_image(fname)

    if img_path:
        predict_single(img_path, model, device, transform)
    else:
        print(f"[경고] 파일을 찾을 수 없습니다: {fname}")

if __name__ == "__main__":
    main()
