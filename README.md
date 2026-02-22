# Real-time Indian ANPR Edge System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal)

A high-performance, real-time Automatic Number Plate Recognition (ANPR) system optimized for edge devices. This system utilizes a lightweight YOLOv8-nano model to detect Indian license plates, processes the crops using EasyOCR for text extraction, and uses a hardcoded ruleset to determine the Indian state based on the first two alphanumeric characters.

## Features

- **Edge Deployment Optimized**: Uses `yolov8n.pt` and `EasyOCR` on CPU for maximum compatibility across devices without heavy GPUs.
- **Real-time Processing**: Stream video directly from a camera to the backend API via HTTP.
- **State Recognition**: Hardcoded Indian regional RTO codes recognize the state of registration (e.g. `DL` -> Delhi, `MH` -> Maharashtra).
- **Premium Web UI**: Responsive, dark-mode glassmorphism interface built with vanilla HTML/CSS/JS.

## Project Structure

```text
/
├── src/
│   ├── data_prep.py     # Parses PASCAL VOC XML from @license_plates to YOLO txt
│   ├── train.py         # YOLOv8-nano training pipeline
│   ├── inference.py     # Inference pipeline (YOLO detect -> Crop -> EasyOCR -> State Check)
│   └── app.py           # FastAPI Web Server
├── models/              # Contains the exported edge weights (`plate_detector.pt`)
├── data/                # Processed train/validation splits for YOLO
├── web/                 
│   ├── index.html       # Web UI
│   ├── style.css        
│   └── script.js        
├── docs/                # YOLO training performance metrics & graphs
└── requirements.txt     # Python dependencies
```

## Setup Instructions

### 1. Installation

Ensure you have Python 3.9+ installed.

```bash
# Clone the repository
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data & Train (Optional)

If you haven't processed the `@license_plates` dataset yet:

```bash
# Convert XML to YOLO format
python src/data_prep.py

# Train the custom edge model
python src/train.py
```

*Note: The best weights will automatically save to `/models/plate_detector.pt` and graphs will save to `/docs/`.*

### 3. Running the System

Start the FastAPI application:

```bash
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

Once the server is running, simply double click `web/index.html` in your browser (or serve it with a static web server like Live Server) to access the real-time inference dashboard.

## Dashboard Usage
- **Image Upload**: Click the dashed upload zone to upload an image of a car from the dataset. The backend will draw bounding boxes and read the plate.
- **Live Camera**: Click 'Start Stream' to process your webcam feed at ~1 FPS and see live plate detection.
