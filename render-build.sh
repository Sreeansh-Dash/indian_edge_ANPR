#!/bin/bash
# Install dependencies
pip install -r requirements.txt

# Create cache directories
mkdir -p models/easyocr

# Download YOLOv8n weights directly so Ultralytics doesn't download it at runtime
if [ ! -f "models/plate_detector.pt" ]; then
    echo "Downloading default YOLOv8n weights for edge inference..."
    curl -L https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -o models/plate_detector.pt
fi

# Pre-download EasyOCR models (detection & recognition) so it doesn't happen at API startup
echo "Caching EasyOCR models..."
python -c "import os; os.makedirs('models/easyocr', exist_ok=True); import easyocr; easyocr.Reader(['en'], gpu=False, model_storage_directory='models/easyocr')"

echo "Build complete."
