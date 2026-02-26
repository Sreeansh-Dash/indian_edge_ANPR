"""
validator.py — Indian RTO License Plate Format Validator

Indian vehicle registration plates follow a strict format defined by the
Ministry of Road Transport and Highways (MoRTH):

  <STATE_CODE (2 uppercase letters)>
  <DISTRICT_NUMBER (2 digits)>
  <SERIES (1–2 uppercase letters)>
  <REGISTRATION_NUMBER (4 digits)>

  Example: MH12AB1234 | DL3CAB1234 | KA05MG5678

Special cases handled:
  - BH (Bharat series, 2021+): YY BH NNNN CC  e.g. 22BH1234AB
  - Older plates may omit series letters (e.g., MH121234)

Validity categories:
  VALID   — Matches full Indian plate grammar
  PARTIAL — State code is a known RTO code but full format doesn't match
  INVALID — Does not begin with a known state code at all
"""

import re

# All valid Indian RTO state/UT codes
VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH",
    "KA", "KL", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB",
    "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "WB",
    # Union Territories
    "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD", "PY"
}

# Standard plate: SS DD [A-Z]{1,2} [0-9]{4}
_STANDARD_PLATE_RE = re.compile(
    r'^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{4})$'
)

# Bharat series: DD BH NNNN [A-Z]{2}
_BH_PLATE_RE = re.compile(
    r'^(\d{2})(BH)(\d{4})([A-Z]{2})$'
)

# Short / district-only plate (no series): SS DD NNNN
_SHORT_PLATE_RE = re.compile(
    r'^([A-Z]{2})(\d{2})(\d{4})$'
)


def validate_plate(plate_text: str) -> dict:
    """
    Validate an extracted plate string against Indian RTO plate grammar.

    Args:
        plate_text: Cleaned, uppercased plate string (alphanumeric only)

    Returns:
        dict with keys:
          - validity: "VALID" | "PARTIAL" | "INVALID"
          - plate_type: "STANDARD" | "BH_SERIES" | "SHORT" | "UNKNOWN"
          - state_code: detected 2-letter code or None
          - details: human-readable explanation
    """
    text = plate_text.upper().strip()

    # ── Bharat series check ──────────────────────────────────────────────────
    bh_match = _BH_PLATE_RE.match(text)
    if bh_match:
        year, _, number, series = bh_match.groups()
        return {
            "validity": "VALID",
            "plate_type": "BH_SERIES",
            "state_code": "BH",
            "details": f"Bharat Series plate — Year: 20{year}, Number: {number}, Class: {series}"
        }

    # ── Standard plate check ──────────────────────────────────────────────────
    std_match = _STANDARD_PLATE_RE.match(text)
    if std_match:
        state_code, district, series, number = std_match.groups()
        if state_code in VALID_STATE_CODES:
            return {
                "validity": "VALID",
                "plate_type": "STANDARD",
                "state_code": state_code,
                "details": f"Valid — State: {state_code}, District: {district}, Series: {series}, Reg: {number}"
            }
        else:
            return {
                "validity": "PARTIAL",
                "plate_type": "STANDARD",
                "state_code": state_code,
                "details": f"Format OK but '{state_code}' is not a recognised RTO state code"
            }

    # ── Short plate check (no series letters) ────────────────────────────────
    short_match = _SHORT_PLATE_RE.match(text)
    if short_match:
        state_code, district, number = short_match.groups()
        if state_code in VALID_STATE_CODES:
            return {
                "validity": "PARTIAL",
                "plate_type": "SHORT",
                "state_code": state_code,
                "details": f"Short format (no series) — State: {state_code}, District: {district}, Reg: {number}"
            }

    # ── Partial: at least starts with valid 2-letter state code ─────────────
    if len(text) >= 2 and text[:2] in VALID_STATE_CODES:
        return {
            "validity": "PARTIAL",
            "plate_type": "UNKNOWN",
            "state_code": text[:2],
            "details": f"Starts with valid state code '{text[:2]}' but full format doesn't match standard grammar"
        }

    # ── Completely invalid ───────────────────────────────────────────────────
    return {
        "validity": "INVALID",
        "plate_type": "UNKNOWN",
        "state_code": None,
        "details": f"'{text}' does not match any known Indian plate format"
    }
