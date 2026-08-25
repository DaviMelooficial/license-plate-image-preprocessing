import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.restoration import estimate_sigma

CCPD_PATTERN = re.compile(
    r"^(?P<area>\d+)-(?P<tilt_h>\d+)_(?P<tilt_v>\d+)-"
    r"(?P<x1>\d+)&(?P<y1>\d+)_(?P<x2>\d+)&(?P<y2>\d+)-"
    r"(?P<v1x>\d+)&(?P<v1y>\d+)_(?P<v2x>\d+)&(?P<v2y>\d+)_"
    r"(?P<v3x>\d+)&(?P<v3y>\d+)_(?P<v4x>\d+)&(?P<v4y>\d+)-"
    r"(?P<plate_code>[\d_]+)-(?P<brightness_ccpd>\d+)-(?P<blur_ccpd>\d+)$"
)

# Limiares fixos calibrados a partir de mean +/- 0.5*std das 100 imagens
# analisadas em eda_pre_process.ipynb (mesma logica de z-score da EDA,
# congelada em valores absolutos para permitir testar UMA imagem isolada).
BRIGHTNESS_LOW = 57.4
BRIGHTNESS_HIGH = 105.5
CONTRAST_LOW = 36.0
SHARPNESS_LOW = 61.1
NOISE_HIGH = 0.225
PLATE_AREA_LOW = 0.0183


def parse_ccpd_bbox(filename_stem):
    m = CCPD_PATTERN.match(filename_stem)
    if m is None:
        return None
    g = m.groupdict()
    return (int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"]))


def _clamp_bbox(x1, y1, x2, y2, w, h):
    x1, x2 = sorted((max(0, min(x1, w - 1)), max(0, min(x2, w - 1))))
    y1, y2 = sorted((max(0, min(y1, h - 1)), max(0, min(y2, h - 1))))
    if x2 <= x1:
        x2 = min(x1 + 1, w - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, h - 1)
    return x1, y1, x2, y2


def compute_metrics(img_bgr, bbox=None):
    h, w = img_bgr.shape[:2]
    plate_area_ratio = None

    if bbox is not None:
        x1, y1, x2, y2 = _clamp_bbox(*bbox, w, h)
        region = img_bgr[y1:y2, x1:x2]
        plate_area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
    else:
        region = img_bgr

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    noise_sigma = estimate_sigma(gray, average_sigmas=True, channel_axis=None)

    return {
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise_estimate": float(noise_sigma),
        "plate_area_ratio": plate_area_ratio,
    }


def diagnose(metrics):
    flags = {
        "baixa_luz": metrics["brightness_mean"] < BRIGHTNESS_LOW,
        "estourada": metrics["brightness_mean"] > BRIGHTNESS_HIGH,
        "baixo_contraste": metrics["contrast_std"] < CONTRAST_LOW,
        "desfocada": metrics["sharpness_laplacian_var"] < SHARPNESS_LOW,
        "ruidosa": metrics["noise_estimate"] > NOISE_HIGH,
        "placa_pequena": False,
    }
    if metrics["plate_area_ratio"] is not None:
        flags["placa_pequena"] = metrics["plate_area_ratio"] < PLATE_AREA_LOW
    return flags


def auto_gamma_correction(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mean_norm = np.clip(v.mean() / 255.0, 1e-3, 1 - 1e-3)
    gamma = np.log(0.5) / np.log(mean_norm)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    hsv[:, :, 2] = cv2.LUT(v, lut)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise_nlmeans(img_bgr):
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )


def unsharp_mask(img_bgr, sigma=3, amount=1.5):
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    return cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)


def upscale_lanczos(img_bgr, scale=1.5):
    h, w = img_bgr.shape[:2]
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)


def process_image(img_bgr, bbox=None):
    metrics = compute_metrics(img_bgr, bbox=bbox)
    flags = diagnose(metrics)

    processed = img_bgr
    corrections = []

    if flags["ruidosa"]:
        processed = denoise_nlmeans(processed)
        corrections.append("ruidosa")
    if flags["baixa_luz"] or flags["estourada"]:
        processed = auto_gamma_correction(processed)
        corrections.append("baixa_luz" if flags["baixa_luz"] else "estourada")
    if flags["baixo_contraste"]:
        processed = apply_clahe(processed)
        corrections.append("baixo_contraste")
    if flags["desfocada"]:
        processed = unsharp_mask(processed)
        corrections.append("desfocada")
    if flags["placa_pequena"]:
        processed = upscale_lanczos(processed)
        corrections.append("placa_pequena")

    return processed, corrections, flags, metrics


def _scale_bbox_for_upscale(bbox, corrections_applied, scale=1.5):
    if bbox is None or "placa_pequena" not in corrections_applied:
        return bbox
    x1, y1, x2, y2 = bbox
    return (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))


def process_folder(images_dir, output_dir, db_path):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(images_dir.glob("*.jpg")):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue

        bbox = parse_ccpd_bbox(path.stem)
        before_metrics = compute_metrics(img_bgr, bbox=bbox)
        processed, corrections, flags, _ = process_image(img_bgr, bbox=bbox)

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), processed)

        after_bbox = _scale_bbox_for_upscale(bbox, corrections)
        after_metrics = compute_metrics(processed, bbox=after_bbox)

        record = {
            "original_filename": path.name,
            "processed_filename": path.name,
            "original_path": str(path),
            "processed_path": str(out_path),
            **{f"flag_{k}": v for k, v in flags.items()},
            "corrections_applied": ",".join(corrections) if corrections else "",
            "brightness_mean_before": before_metrics["brightness_mean"],
            "brightness_mean_after": after_metrics["brightness_mean"],
            "contrast_std_before": before_metrics["contrast_std"],
            "contrast_std_after": after_metrics["contrast_std"],
            "sharpness_laplacian_var_before": before_metrics["sharpness_laplacian_var"],
            "sharpness_laplacian_var_after": after_metrics["sharpness_laplacian_var"],
            "noise_estimate_before": before_metrics["noise_estimate"],
            "noise_estimate_after": after_metrics["noise_estimate"],
            "plate_area_ratio_before": before_metrics["plate_area_ratio"],
            "plate_area_ratio_after": after_metrics["plate_area_ratio"],
        }
        records.append(record)

    df = pd.DataFrame(records)

    conn = sqlite3.connect(db_path)
    df.to_sql("image_mapping", conn, if_exists="replace", index=True, index_label="id")
    conn.close()

    return df
