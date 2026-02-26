import os
import cv2
import numpy as np
import easyocr
import traceback
from ultralytics import YOLO

# Initialize EasyOCR Reader (English)
print("Initializing EasyOCR reader...")
ocr_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'easyocr')
os.makedirs(ocr_cache_dir, exist_ok=True)
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=ocr_cache_dir)

# Load trained YOLOv8 model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'plate_detector.pt')
if not os.path.exists(model_path):
    print(f"Warning: Model not found at {model_path}. Loading default yolov8n.pt for demonstration.")
    yolo_model = YOLO("yolov8n.pt")
else:
    yolo_model = YOLO(model_path)

# Hardcoded State Dictionary
STATE_CODES = {
    "AP": "Andhra Pradesh",   "AR": "Arunachal Pradesh", "AS": "Assam",
    "BR": "Bihar",            "CG": "Chhattisgarh",       "GA": "Goa",
    "GJ": "Gujarat",          "HR": "Haryana",            "HP": "Himachal Pradesh",
    "JH": "Jharkhand",        "KA": "Karnataka",          "KL": "Kerala",
    "MP": "Madhya Pradesh",   "MH": "Maharashtra",        "MN": "Manipur",
    "ML": "Meghalaya",        "MZ": "Mizoram",            "NL": "Nagaland",
    "OD": "Odisha",           "PB": "Punjab",             "RJ": "Rajasthan",
    "SK": "Sikkim",           "TN": "Tamil Nadu",         "TS": "Telangana",
    "TR": "Tripura",          "UP": "Uttar Pradesh",      "UK": "Uttarakhand",
    "WB": "West Bengal",      "AN": "Andaman & Nicobar",  "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli", "DD": "Daman and Diu", "DL": "Delhi",
    "JK": "Jammu & Kashmir",  "LA": "Ladakh",             "LD": "Lakshadweep",
    "PY": "Puducherry",
}

# Confidence below which multi-pass OCR voting is triggered
OCR_CONFIDENCE_THRESHOLD = 0.70


def clean_ocr_text(text: str) -> str:
    """Remove spaces and special characters from OCR text output."""
    return ''.join(e for e in text if e.isalnum()).upper()


def identify_state(plate_text: str) -> str:
    """Identify Indian state based on first two characters."""
    if len(plate_text) >= 2:
        return STATE_CODES.get(plate_text[:2], "Unknown / Invalid")
    return "Unknown"


# ─────────────────────────────────────────────────────────────────
# Novelty 1: Adaptive Multi-Stage Preprocessing Pipeline
# ─────────────────────────────────────────────────────────────────

def _generate_preprocessing_variants(gray: np.ndarray) -> list:
    """
    Generate multiple enhanced versions of a grayscale plate crop.

    Each variant targets a different image quality scenario:
      v0 — Denoised + CLAHE + Otsu threshold  (main novel pipeline)
      v1 — CLAHE only + Otsu threshold         (no denoising)
      v2 — Adaptive threshold (Gaussian)       (handles uneven lighting)
      v3 — Raw Otsu threshold                  (original baseline)

    Returns:
        List of (label, binary_image) tuples, ordered by expected quality.
    """
    # Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE — locally boosts contrast in small tiles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_denoised = clahe.apply(denoised)
    clahe_only = clahe.apply(gray)

    # Binary images
    _, v0 = cv2.threshold(clahe_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, v1 = cv2.threshold(clahe_only, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    _, v3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        ("clahe+denoise+otsu", v0),
        ("clahe+otsu",         v1),
        ("adaptive_gaussian",  v2),
        ("raw_otsu",           v3),
    ]


# ─────────────────────────────────────────────────────────────────
# Novelty 2: Multi-Pass OCR Voting
# ─────────────────────────────────────────────────────────────────

def _ocr_single_pass(image: np.ndarray) -> tuple[str, float]:
    """
    Run EasyOCR on one image variant.

    Returns:
        (clean_text, mean_confidence) — confidence is 0.0 if no text found.
    """
    try:
        ocr_results = reader.readtext(image)
        if not ocr_results:
            return "", 0.0

        raw_text = "".join([r[1] for r in ocr_results])
        mean_conf = float(np.mean([r[2] for r in ocr_results]))
        return clean_ocr_text(raw_text), mean_conf
    except Exception as e:
        print(f"OCR single-pass error: {e}")
        return "", 0.0


def _multi_pass_ocr(variants: list) -> tuple[str, float]:
    """
    Run OCR across all preprocessing variants and select the best result.

    Strategy:
      1. Run the primary variant (v0 — CLAHE+denoise+Otsu).
      2. If its confidence exceeds OCR_CONFIDENCE_THRESHOLD, return immediately.
      3. Otherwise run all remaining variants, collect (text, confidence) pairs.
      4. Return the (text, confidence) pair with the highest confidence.

    Args:
        variants: List of (label, image) tuples from _generate_preprocessing_variants()

    Returns:
        (best_text, best_confidence)
    """
    best_text, best_conf = "", 0.0
    results = []

    for label, img in variants:
        text, conf = _ocr_single_pass(img)
        if not text:
            continue
        results.append((text, conf, label))

        # Early exit if primary pass is confident enough
        if label == variants[0][0] and conf >= OCR_CONFIDENCE_THRESHOLD:
            return text, conf

    if not results:
        return "", 0.0

    # Pick variant with max confidence
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0], results[0][1]


# ─────────────────────────────────────────────────────────────────
# Main inference entry point
# ─────────────────────────────────────────────────────────────────

def process_image(img_array: np.ndarray) -> list:
    """
    Takes an OpenCV BGR image array.

    Returns:
        List of detected plates, each containing:
          bbox        — [x1, y1, x2, y2]
          confidence  — YOLO bounding-box confidence
          text        — Cleaned OCR plate text
          state       — Resolved Indian state / UT
          ocr_confidence — Mean OCR confidence across recognised characters
    """
    results = yolo_model.predict(source=img_array, save=False, augment=False, conf=0.25)

    detected_plates = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            plate_crop = img_array[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            # ── Novelty 1: generate preprocessing variants ──────────────────
            variants = _generate_preprocessing_variants(gray)

            # ── Novelty 2: multi-pass OCR voting ───────────────────────────
            try:
                clean_text, ocr_conf = _multi_pass_ocr(variants)

                if not clean_text:
                    continue

                state = identify_state(clean_text)

                detected_plates.append({
                    "bbox":           [x1, y1, x2, y2],
                    "confidence":     conf,
                    "text":           clean_text,
                    "state":          state,
                    "ocr_confidence": ocr_conf,
                })
            except Exception as e:
                print(f"OCR Error: {e}")
                traceback.print_exc()

    return detected_plates
