"""
对 DeblurGAN-v2 去模糊结果进行彩色图像增强微调
================================================
利用经典图像处理技术（参考 pic_exec 项目）对 submit/ 目录的输出做轻量级增强：

1. Lab 空间 CLAHE 自适应对比度增强（温和参数）
2. Unsharp Masking 轻微锐化
3. 灰度世界偏色矫正（自动白平衡）
4. HSV 空间饱和度微调
5. 双边滤波去噪

用法：
    python enhance_submit.py [input_dir] [output_dir]
    python enhance_submit.py                          # 默认 submit/ -> submit_enhanced/
    python enhance_submit.py submit/ submit_enhanced/
"""

import os
import sys
import cv2
import numpy as np


# ============================================================
# 增强函数（参考 pic_exec 中的技术手段）
# ============================================================

def correct_color_cast(image):
    """
    灰度世界假设矫正偏色（来自 pic_exec/程序1）
    假设图像各通道均值应趋于灰色，调整各通道比例
    """
    result = image.astype(np.float64)
    avg_r = result[:, :, 0].mean()
    avg_g = result[:, :, 1].mean()
    avg_b = result[:, :, 2].mean()
    avg_gray = (avg_r + avg_g + avg_b) / 3.0

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / (avg_r + 1e-6)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / (avg_g + 1e-6)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / (avg_b + 1e-6)), 0, 255)

    return result.astype(np.uint8)


def clahe_enhance_lab(image, clip_limit=0.8, tile_grid=(8, 8)):
    """
    在 Lab 色彩空间对 L 通道做 CLAHE 对比度增强
    只处理亮度通道，保留原始色彩（来自 pic_exec/程序2）

    参数保守：clipLimit=0.8（默认2.0）
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    lab[:, :, 0] = l_enhanced
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def unsharp_mask(image, amount=1.1, sigma=1.5):
    """
    Unsharp Masking 锐化（来自 pic_exec/程序1）
    output = amount * original - (amount - 1) * blurred

    参数保守：amount=1.1（原始用1.8），sigma=1.5
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image, amount, blurred, -(amount - 1), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def adjust_saturation(image, factor=1.05):
    """
    在 HSV 空间调整饱和度（来自 pic_exec/图片处理.md - 皮肤色相/饱和度调整）
    factor > 1.0 增加饱和度，< 1.0 降低饱和度

    参数保守：factor=1.05（+5%）
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def bilateral_denoise(image, d=5, sigma_color=15, sigma_space=15):
    """
    双边滤波去噪（来自 pic_exec/程序2 - 边缘保留平滑）
    在去模糊后做轻微去噪，消除可能的振铃/噪声

    参数保守：d=5, sigmaColor=15, sigmaSpace=15
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# ============================================================
# 主增强流水线
# ============================================================

def enhance_image(img_rgb):
    """
    对单张图片执行完整的增强流水线：

    Step 1: 灰度世界偏色矫正 — 自动白平衡
    Step 2: Lab 空间 CLAHE — 自适应对比度增强
    Step 3: Unsharp Masking — 轻微锐化
    Step 4: 饱和度微调 — 让色彩更鲜活
    Step 5: 双边滤波 — 边缘保留去噪
    """
    print("  [1/5] 灰度世界偏色矫正...")
    step1 = correct_color_cast(img_rgb)

    print("  [2/5] CLAHE 对比度增强...")
    step2 = clahe_enhance_lab(step1, clip_limit=0.8)

    print("  [3/5] Unsharp Masking 锐化...")
    step3 = unsharp_mask(step2, amount=1.1, sigma=1.5)

    print("  [4/5] 饱和度微调...")
    step4 = adjust_saturation(step3, factor=1.05)

    print("  [5/5] 双边滤波去噪...")
    step5 = bilateral_denoise(step4, d=5, sigma_color=15, sigma_space=15)

    return step5


def main(input_dir='submit/', output_dir='submit_enhanced/'):
    """
    批量处理 input_dir 下的所有图片，输出到 output_dir
    """
    input_dir = os.path.normpath(input_dir)
    output_dir = os.path.normpath(output_dir)

    if not os.path.isdir(input_dir):
        print(f"[错误] 输入目录不存在: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片格式
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(extensions)]

    if not files:
        print(f"[警告] 输入目录中没有图片: {input_dir}")
        sys.exit(0)

    print(f"找到 {len(files)} 张图片待处理")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 50)

    for filename in files:
        filepath = os.path.join(input_dir, filename)
        print(f"\n处理: {filename}")

        # 读取图片（BGR -> RGB）
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            print(f"  [跳过] 无法读取: {filepath}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 执行增强
        result_rgb = enhance_image(img_rgb)

        # 保存（RGB -> BGR）
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"  保存到: {out_path}")

    print("\n" + "=" * 50)
    print(f"全部完成！增强后的图片保存在: {output_dir}")


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
