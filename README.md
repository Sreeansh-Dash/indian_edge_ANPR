# NeuralPlate | Real-time Indian ANPR Edge System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal)

## Executive Summary

**NeuralPlate** is a high-performance, real-time Automatic Number Plate Recognition (ANPR) system specifically engineered for Indian license plates and optimized for edge deployment. The system combines modern deep learning (YOLOv8) for object detection with robust OCR (EasyOCR) and a custom regional recognition engine.

### Key Highlights
- **Edge Efficiency**: Designed to run on consumer-grade CPUs with ~600-900ms latency, making it ideal for standard workstations and edge gateways.
- **Adaptive Multi-Stage Preprocessing**: Generates multiple image variants (CLAHE, Denoising, Adaptive Thresholding) to dynamically handle poor lighting and motion blur.
- **Multi-Pass OCR Voting**: Fault-tolerant text extraction that re-runs OCR on alternate image variants if confidence drops below 70%, voting on the best outcome.
- **RTO Format Validator**: A dedicated rule engine that validates extracted text against structural grammar (e.g., `MH12AB1234`) and flags invalid/partial reads.
- **Detection History Log**: In-memory rolling audit trail of the session's detections with a live frontend feed via the `/history` endpoint.
- **Premium Interface**: A glassmorphism dashboard providing real-time visual feedback, bounding boxes, OCR confidence, plate validity badges, and performance analytics.

---

## File-by-File Documentation

### 🚀 Backend Components (`/src`)

#### [`app.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/app.py) - The API Gateway
- **Purpose**: Serves as the interface between the AI models and the web dashboard.
- **Key Logic**: 
    - Uses **FastAPI** to handle asynchronous image uploads via the `/detect` endpoint.
    - Exposes a `/history` endpoint (GET/DELETE) to serve the in-memory session detection log.
    - Orchestrates the call to the inference module, validity engine, and log manager.
    - Returns structured JSON responses (Bounding boxes, Text, State, OCR Confidence, Validity).

#### [`inference.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/inference.py) - The Intelligence Engine
- **Purpose**: The core pipeline that transforms raw pixels into text and state information.
- **Key Logic**:
    - **Plate Detection**: Uses `yolov8n.pt` (Nano) to locate Indian license plates within an image.
    - **Adaptive Preprocessing**: Generates up to 4 enhanced variants (CLAHE, Denoising, etc.) of the plate crop.
    - **Multi-Pass OCR Voting**: Uses **EasyOCR** (English) across the variants and selects the read with the highest confidence.
    - **State Check**: Maps the first two characters of the plate to a dictionary of 35+ Indian states/UTs.

#### [`validator.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/validator.py) - RTO Format Validator *(Novelty)*
- **Purpose**: Validates the structural grammar of the extracted plate text.
- **Key Logic**: Checks text against regex rules for Standard (e.g., MH12AB1234), Short, and BH-series Indian plates. Returns `VALID`, `PARTIAL`, or `INVALID`.

#### [`log_manager.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/log_manager.py) - Memory Log *(Novelty)*
- **Purpose**: Maintains a thread-safe circular buffer for the active detection session.
- **Key Logic**: Stores up to 50 of the most recent detections, recording timestamp, plate text, state, and validity for the frontend history feed.

#### [`train.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/train.py) - Model Training Pipeline
- **Purpose**: Automates the fine-tuning of YOLOv8 for specific license plate datasets.
- **Key Logic**:
    - Configures training parameters (epochs, image size) and stores results in `/runs`.
    - Automatically exports the best-performing weights (`best.pt`) to the `/models` folder.
    - Generates and saves performance visualizations (Metrics & Confusion Matrix) to `/docs` for the dashboard.

#### [`data_prep.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/data_prep.py) - Dataset Preprocessing
- **Purpose**: Prepares raw datasets for YOLOv8 training.
- **Key Logic**:
    - Parses **PASCAL VOC XML** annotations and converts them to the normalized **YOLO .txt format**.
    - Implements an 80/20 train-validation split.
    - Generates the `data.yaml` configuration file required by the Ultralytics framework.

### 🎨 Frontend Components (`/web`)

#### [`index.html`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/web/index.html) - Dashboard Structure
- A responsive, glassmorphism-themed UI structure.
- Features dual-views: **Live Inference** (interactive workspace with a real-time Session History panel) and **Model Analytics** (training metrics).
- Uses Google Fonts (Outfit) and Phosphor Icons for a premium aesthetic.

#### [`script.js`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/web/script.js) - Interface Logic
- Handles the **Webcam Stream** (1 FPS) and **Drag-and-Drop** file uploads.
- Communicates with the FastAPI backend endpoints (`/detect`, `/history`) via `fetch`.
- Dynamically draws bounding boxes and text labels on a `<canvas>` overlay.
- Renders the color-coded validity badges and live history feed.

#### [`style.css`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/web/style.css) - Design System
- Implements a futuristic dark theme using HSL color tokens.
- Defines glassmorphism effects (`backdrop-filter`) and smooth micro-animations for card entries.

---

## Project Structure

```text
/
├── src/                 # System logic & API
├── models/              # Exported AI weights (plate_detector.pt)
├── data/                # Processed YOLO dataset (images/labels)
├── web/                 # Dashboard source code (HTML/CSS/JS)
├── docs/                # Auto-generated training graphs & metrics
├── license_plates/      # Raw raw dataset (XML/Images)
└── requirements.txt     # Environment dependencies
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
