"""
log_manager.py — In-Memory Detection History Log

Maintains a thread-safe circular buffer of the last N plate detections
for the current server session. Exposed via the GET /history API endpoint.
"""

import threading
from collections import deque
from datetime import datetime, timezone
from typing import List

# Maximum number of detections to keep in memory
MAX_LOG_SIZE = 50

_lock = threading.Lock()
_log: deque = deque(maxlen=MAX_LOG_SIZE)


def log_detection(plate_text: str, state: str, confidence: float,
                  ocr_confidence: float, validity: str) -> None:
    """
    Append a detection event to the in-memory log.

    Args:
        plate_text: Cleaned OCR text of the detected plate
        state: Resolved Indian state / UT name
        confidence: YOLO bounding-box confidence (0–1)
        ocr_confidence: OCR text confidence (0–1)
        validity: One of 'VALID', 'PARTIAL', 'INVALID'
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plate_text": plate_text,
        "state": state,
        "confidence": round(confidence, 4),
        "ocr_confidence": round(ocr_confidence, 4),
        "validity": validity
    }
    with _lock:
        _log.appendleft(entry)   # Most-recent first


def get_history(limit: int = MAX_LOG_SIZE) -> List[dict]:
    """
    Return the most recent detections, newest first.

    Args:
        limit: Maximum number of entries to return (capped at MAX_LOG_SIZE)

    Returns:
        List of detection dicts
    """
    with _lock:
        return list(_log)[:min(limit, MAX_LOG_SIZE)]


def clear_history() -> None:
    """Clear all detection history from memory."""
    with _lock:
        _log.clear()
