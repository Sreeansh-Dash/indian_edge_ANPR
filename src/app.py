from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import uvicorn
from typing import List

from fastapi.staticfiles import StaticFiles

# Import our inference module
from src.inference import process_image

app = FastAPI(title="Indian ANPR System")

# Mount charts directory
import os
docs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
app.mount("/charts", StaticFiles(directory=docs_path), name="charts")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlateResult(BaseModel):
    bbox: List[int]
    confidence: float
    text: str
    state: str

class DetectionResponse(BaseModel):
    success: bool
    plates: List[PlateResult]
    error: str = None

@app.post("/detect", response_model=DetectionResponse)
async def detect_plates(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = process_image(img)
        
        return DetectionResponse(
            success=True,
            plates=results
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "plates": [], "error": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
