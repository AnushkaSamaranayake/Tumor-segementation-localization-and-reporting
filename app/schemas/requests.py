from pydantic import BaseModel
from typing import List

class ImageRequest(BaseModel):
    image_base64: str

class MaskRequest(BaseModel):
    mask : List[List[int]]