from predict import Predictor
import cv2
import numpy as np
import os

imgs = [
    'test_img/000201.png',
    'test_img/kohler_blur_1.png',
    'test_img/kohler_blur_2.png',
    'test_img/kohler_blur_3.png',
    'test_img/kohler_blur_4.png',
]

predictor = Predictor(weights_path='fpn_inception.h5')
os.makedirs('submit_deblur', exist_ok=True)

for f in imgs:
    basename = os.path.basename(f)
    img = cv2.imread(f)
    if img is None:
        print(f'SKIP (cannot read): {f}')
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = predictor(img_rgb, None)
    pred_bgr = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    out = os.path.join('submit_deblur', basename)
    cv2.imwrite(out, pred_bgr)
    print(f'Done: {out} ({pred_bgr.shape})')
