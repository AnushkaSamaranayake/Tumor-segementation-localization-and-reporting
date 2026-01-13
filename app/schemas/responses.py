from pydantic import BaseModel
from typing import Dict, List

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class Arrow(BaseModel):
    start: Dict[str, float]
    end: Dict[str, float]

class LocalizeOutput(BaseModel):
    bounding_box: BoundingBox
    distances_mm: Dict[str, float]
    arrows: Dict[str, Arrow]

class ContourPoint(BaseModel):
    x: int
    y: int

class ContourResponse(BaseModel):
    contours: Dict[str, List[ContourPoint]]

class PredictionResponse(BaseModel):
    mask_base64: str

class FullResponse(BaseModel):
    prediction: PredictionResponse
    localization: LocalizeOutput
    contours: ContourResponse