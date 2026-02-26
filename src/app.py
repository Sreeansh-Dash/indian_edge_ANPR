from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
import uvicorn
import os
from typing import List, Optional

# Core modules
from src.inference import process_image
from src.validator import validate_plate
from src.log_manager import log_detection, get_history, clear_history

app = FastAPI(title="NeuralPlate — Indian ANPR System")

# Mount training charts directory
docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
app.mount("/charts", StaticFiles(directory=docs_path), name="charts")

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────────

class PlateResult(BaseModel):
    bbox:           List[int]
    confidence:     float
    text:           str
    state:          str
    ocr_confidence: float    # NEW: OCR mean character confidence
    validity:       str      # NEW: VALID | PARTIAL | INVALID
    validity_details: str    # NEW: human-readable explanation

class DetectionResponse(BaseModel):
    success: bool
    plates:  List[PlateResult]
    error:   Optional[str] = None

class HistoryEntry(BaseModel):
    timestamp:      str
    plate_text:     str
    state:          str
    confidence:     float
    ocr_confidence: float
    validity:       str

class HistoryResponse(BaseModel):
    count:   int
    entries: List[HistoryEntry]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/detect", response_model=DetectionResponse)
async def detect_plates(file: UploadFile = File(...)):
    """
    Upload an image to detect and read Indian license plates.

    Returns bounding boxes, OCR text, state, OCR confidence, and
    plate format validity for every detected plate.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        raw_results = process_image(img)

        enriched = []
        for plate in raw_results:
            # ── Novelty 3: Plate Format Validation ──────────────────────────
            validation = validate_plate(plate["text"])

            result = PlateResult(
                bbox           = plate["bbox"],
                confidence     = plate["confidence"],
                text           = plate["text"],
                state          = plate["state"],
                ocr_confidence = plate.get("ocr_confidence", 0.0),
                validity       = validation["validity"],
                validity_details = validation["details"],
            )
            enriched.append(result)

            # ── Novelty 4: Log to detection history ─────────────────────────
            log_detection(
                plate_text     = plate["text"],
                state          = plate["state"],
                confidence     = plate["confidence"],
                ocr_confidence = plate.get("ocr_confidence", 0.0),
                validity       = validation["validity"],
            )

        return DetectionResponse(success=True, plates=enriched)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "plates": [], "error": str(e)}
        )


@app.get("/history", response_model=HistoryResponse)
async def detection_history(limit: int = Query(default=20, ge=1, le=50)):
    """
    Retrieve the most recent plate detections logged this session.

    Args:
        limit: Number of entries to return (1–50, default 20)
    """
    entries = get_history(limit=limit)
    return HistoryResponse(count=len(entries), entries=entries)


@app.delete("/history")
async def clear_detection_history():
    """Clear all detection history from the in-memory log."""
    clear_history()
    return {"success": True, "message": "Detection history cleared."}


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "NeuralPlate ANPR"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
