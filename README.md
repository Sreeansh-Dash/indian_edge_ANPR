# NeuralPlate | Real-time Indian ANPR Edge System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal)

## Executive Summary

**NeuralPlate** is a high-performance, real-time Automatic Number Plate Recognition (ANPR) system specifically engineered for Indian license plates and optimized for edge deployment. The system combines modern deep learning (YOLOv8) for object detection with robust OCR (EasyOCR) and a custom regional recognition engine.

### Key Highlights
- **Edge Efficiency**: Designed to run on consumer-grade CPUs with ~600-900ms latency, making it ideal for standard workstations and edge gateways.
- **Region Intelligence**: Automatically identifies the Indian state/union territory (e.g., `DL` → Delhi, `KA` → Karnataka) using hardcoded RTO logic.
- **Premium Interface**: A glassmorphism dashboard providing real-time visual feedback, bounding boxes, and performance analytics.

---

## File-by-File Documentation

### 🚀 Backend Components (`/src`)

#### [`app.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/app.py) - The API Gateway
- **Purpose**: Serves as the interface between the AI models and the web dashboard.
- **Key Logic**: 
    - Uses **FastAPI** to handle asynchronous image uploads via the `/detect` endpoint.
    - Mounts the `/docs` directory as a static file server to display training charts in the UI.
    - Orchestrates the call to the inference module and returns structured JSON responses (Bounding boxes, Text, State).

#### [`inference.py`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/src/inference.py) - The Intelligence Engine
- **Purpose**: The core pipeline that transforms raw pixels into text and state information.
- **Key Logic**:
    - **Plate Detection**: Uses `yolov8n.pt` (Nano) to locate Indian license plates within an image.
    - **Preprocessing**: Crops detected plates and applies Grayscale + Otsu's Thresholding to improve OCR accuracy.
    - **OCR**: Uses **EasyOCR** (English) to read the alphanumeric characters.
    - **State Check**: Maps the first two characters of the plate to a dictionary of 30+ Indian states/UTs.

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
- Features a dual-view system: **Live Inference** (interactive workspace) and **Model Analytics** (training metrics).
- Uses Google Fonts (Outfit) and Phosphor Icons for a premium aesthetic.

#### [`script.js`](file:///c:/Users/Sreeansh%20Dash/OneDrive/Desktop/Projects/ai%20proj/web/script.js) - Interface Logic
- Handles the **Webcam Stream** (captured at 1 FPS to balance load) and **Drag-and-Drop** file uploads.
- Communicates with the FastAPI backend using `fetch` API.
- Dynamically draws bounding boxes and text labels on a `<canvas>` overlay.

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
