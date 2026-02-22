import os
import cv2
import easyocr
import traceback
from ultralytics import YOLO

# Initialize EasyOCR Reader (English, using CPU or GPU as available)
print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=False) # Fallback to CPU to avoid CUDA OOM if YOLO uses GPU

# Load trained YOLOv8 model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'plate_detector.pt')
if not os.path.exists(model_path):
    print(f"Warning: Model not found at {model_path}. Loading default yolov8n.pt for demonstration.")
    yolo_model = YOLO("yolov8n.pt")
else:
    yolo_model = YOLO(model_path)

# Hardcoded State Dictionary
STATE_CODES = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman & Nicobar",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "JK": "Jammu & Kashmir",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "PY": "Puducherry"
}

def clean_ocr_text(text):
    """Remove spaces and special characters from OCR text output."""
    return ''.join(e for e in text if e.isalnum()).upper()

def identify_state(plate_text):
    """Identify Indian state based on first two characters."""
    if len(plate_text) >= 2:
        state_code = plate_text[:2]
        return STATE_CODES.get(state_code, "Unknown / Invalid")
    return "Unknown"

def process_image(img_array):
    """
    Takes an OpenCV BGR image array.
    Returns:
        - List of detected plates with coordinates, raw text, and state.
        - Annotated image array (optional)
    """
    results = yolo_model.predict(source=img_array, save=False, augment=False, conf=0.25)
    
    detected_plates = []
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Crop the bounding box
            plate_crop = img_array[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            # Convert to Grayscale for better OCR
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run EasyOCR
            try:
                ocr_results = reader.readtext(binary)
                if ocr_results:
                    # Combine all text detected in the plate
                    raw_text = "".join([res[1] for res in ocr_results])
                    clean_text = clean_ocr_text(raw_text)
                    state = identify_state(clean_text)
                    
                    detected_plates.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "text": clean_text,
                        "state": state
                    })
            except Exception as e:
                print(f"OCR Error: {e}")
                traceback.print_exc()

    return detected_plates
