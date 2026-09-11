import os
import matplotlib.pyplot as plt
import numpy as np

# 데이터 설정
metrics = ['Mean IoU\n(Water)', 'Recall', 'Precision', 'F1-Score\n(Dice)']
values = [58.27, 61.67, 93.43, 61.06]
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'] # 파랑, 주황, 빨강, 초록 색상군

# 디자인 테마 설정 (어두운 고급형 테마)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')

# 막대 그리기
bars = ax.bar(metrics, values, color=colors, width=0.55, edgecolor='none', alpha=0.85)

# 축 및 레이블 스타일링
ax.set_ylim(0, 115)
ax.set_ylabel('Performance Score (%)', fontsize=12, fontweight='bold', color='#a0a0c0', labelpad=10)
ax.set_title('U-Net Binary Water Segmentation Performance', fontsize=15, fontweight='bold', color='white', pad=20)
ax.tick_params(colors='#a0a0c0', labelsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.2, color='#a0a0c0')

# 막대 위에 값 레이블링
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),  # 6 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='white')

# 테두리 제거
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# 저장 경로들 설정
out_paths = [
    r"C:\Users\kimse\water_fringe_detection - 복사본\result_images\evaluation_metrics.png",
    r"C:\Users\kimse\water_fringe_detection\result_images\evaluation_metrics.png",
    r"C:\Users\kimse\.gemini\antigravity\brain\831c6100-a054-4b10-9032-e60e39028194\images\evaluation_metrics.png",
    r"C:\Users\kimse\.gemini\antigravity\brain\831c6100-a054-4b10-9032-e60e39028194\evaluation_metrics.png"
]

for p in out_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {p}")

plt.close()
