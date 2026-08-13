"""
Phase 3: Classical OpenCV Visual Analysis (Fast Parallelized Supplementary Signal).
Performs visual feature extraction on downloaded catalogue images without replacing CLIP or FAISS.

Features extracted per image:
- visual_border_detected: 'detected' / 'not_detected' / 'unknown'
- visual_contrast_border: 'detected' / 'not_detected' / 'unknown'
- visual_zari_detected: 'detected' / 'not_detected' / 'unknown'
- visual_decorative_work: 'detected' / 'not_detected' / 'unknown'
- visual_texture_score: float (Laplacian variance)
"""
import os
import io
import requests
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import config

VISUAL_PARQUET_PATH = os.path.join(config.DATA_DIR, "visual_features.parquet")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def analyze_image_opencv(img: Image.Image) -> dict:
    """Analyze a single PIL image with OpenCV classical algorithms."""
    img_rgb = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # Define border region (outer 15% strip) vs center body (inner 70%)
    border_mask = np.ones((h, w), dtype=bool)
    border_mask[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)] = False
    body_mask = ~border_mask

    # 1. Edge density (Canny)
    edges = cv2.Canny(gray, 50, 150)
    border_edge_density = np.mean(edges[border_mask]) if np.any(border_mask) else 0
    body_edge_density = np.mean(edges[body_mask]) if np.any(body_mask) else 0

    border_detected = "detected" if border_edge_density > 15.0 or (border_edge_density > body_edge_density * 1.25) else "not_detected"

    # 2. Region color contrast (HSV difference between border and body)
    h_border = img_hsv[border_mask]
    h_body = img_hsv[body_mask]
    if len(h_border) > 0 and len(h_body) > 0:
        mean_border_hsv = np.mean(h_border, axis=0)
        mean_body_hsv = np.mean(h_body, axis=0)
        color_diff = np.linalg.norm(mean_border_hsv - mean_body_hsv)
        contrast_border = "detected" if color_diff > 35.0 else "not_detected"
    else:
        contrast_border = "unknown"

    # 3. Metallic/Zari sheen detection
    gold_mask = cv2.inRange(img_hsv, np.array([15, 60, 140]), np.array([35, 255, 255]))
    silver_mask = cv2.inRange(img_hsv, np.array([0, 0, 180]), np.array([180, 30, 255]))
    metallic_pixels = np.sum(gold_mask > 0) + np.sum(silver_mask > 0)
    metallic_ratio = metallic_pixels / (h * w)
    zari_detected = "detected" if metallic_ratio > 0.05 else "not_detected"

    # 4. Texture / Decorative complexity (Laplacian variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    decorative_work = "detected" if laplacian_var > 300.0 else "not_detected"

    return {
        "visual_border_detected": border_detected,
        "visual_contrast_border": contrast_border,
        "visual_zari_detected": zari_detected,
        "visual_decorative_work": decorative_work,
        "visual_texture_score": round(laplacian_var, 2),
    }


def process_row(row_dict):
    url = row_dict["image_url"]
    sku = row_dict.get("sku", "")
    res = {
        "image_url": url,
        "sku": sku,
        "visual_border_detected": "unknown",
        "visual_contrast_border": "unknown",
        "visual_zari_detected": "unknown",
        "visual_decorative_work": "unknown",
        "visual_texture_score": 0.0,
    }
    try:
        resp = requests.get(url, timeout=10, headers=HEADERS)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content))
            res.update(analyze_image_opencv(img))
    except Exception:
        pass
    return res


def main():
    meta = pd.read_parquet(config.METADATA_PATH)
    print(f"Running Fast Parallel OpenCV Visual Analysis across all {len(meta)} catalogue images...")

    records = meta.to_dict("records")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        futures = [executor.submit(process_row, r) for r in records]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(f.result())

    vis_df = pd.DataFrame(results)
    vis_df.to_parquet(VISUAL_PARQUET_PATH, index=False)
    print(f"\nVisual features successfully written to {VISUAL_PARQUET_PATH}")
    print("Border detected count:", (vis_df["visual_border_detected"] == "detected").sum())
    print("Zari detected count:", (vis_df["visual_zari_detected"] == "detected").sum())
    print("Contrast border count:", (vis_df["visual_contrast_border"] == "detected").sum())


if __name__ == "__main__":
    main()
