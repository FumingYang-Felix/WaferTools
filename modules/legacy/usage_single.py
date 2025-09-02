
import cv2
from pathlib import Path
from section_counter import (
    process_image_with_sam,
    process_image_with_auto_generator,
)


# ---------- 简易可视化 ----------
import numpy as np
import cv2
from pycocotools import mask as mask_utils

def draw_masks(image_bgr, masks, alpha=0.6):
    """将 masks 叠加到 BGR 图片上并返回新图像。"""
    overlay = image_bgr.copy()

    for mask in masks:
        # 1️⃣ 取得二值掩膜 m_disp ------------------------------------------------
        if isinstance(mask, dict) and "segmentation" in mask:
            seg = mask["segmentation"]
            if isinstance(seg, dict):          # COCO RLE
                m_disp = mask_utils.decode(seg)
            else:                              # COCO Polygon list[list[float]]
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                m_disp = np.zeros(image_bgr.shape[:2], np.uint8)
                cv2.fillPoly(m_disp, [poly], 1)
        else:                                  # 已经是 np.ndarray
            m_disp = mask.astype(np.uint8)

        # 2️⃣ 颜色叠加 -----------------------------------------------------------
        overlay[m_disp > 0] = cv2.addWeighted(
            image_bgr[m_disp > 0], 1 - alpha,
            np.full_like(image_bgr[m_disp > 0], (0, 255, 0)),  # 绿色
            alpha, 0
        )

        # 3️⃣ 画轮廓 -------------------------------------------------------------
        cnts, _ = cv2.findContours(m_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)

    return overlay      


# ========= 配置 =========
IMAGE_PATH = "w06_250nm_1.0mms_60sections_RGB.png_files.png"
CHECKPOINT  = "sam_vit_l.pth"
MODEL_TYPE  = "vit_l"          # vit_b / vit_l / vit_h
USE_AUTO    = True             # True ➜ SamAutomaticMaskGenerator；False ➜ 改进网格点
OUT_DIR     = Path("out_single")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ========= 检测 =========
if USE_AUTO:
    masks = process_image_with_auto_generator(
        IMAGE_PATH,
        model_type=MODEL_TYPE,
        # checkpoint=CHECKPOINT,   # ★ 只对自动模式必传
        # points_per_side=32,      # ← 以下皆可覆写默认超参
        # pred_iou_thresh=0.8,
        # min_mask_region_area=500,
    )
else:
    masks = process_image_with_sam(
        IMAGE_PATH,
        model_type=MODEL_TYPE,
        checkpoint=CHECKPOINT,   # 同样支持手动指定
        use_auto_generator=False,
        points_per_side=32,      # 网格点密度
    )

print(f"[INFO] 得到 {len(masks)} 个 masks")

# ========= 可视化 =========
image_bgr = cv2.imread(IMAGE_PATH)
overlay   = draw_masks(image_bgr, masks, alpha=0.6)
cv2.imwrite(str(OUT_DIR / "overlay.png"), overlay)
print(f"[INFO] overlay.png -> {OUT_DIR.absolute()}")
