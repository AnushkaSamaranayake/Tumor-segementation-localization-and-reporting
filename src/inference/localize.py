import cv2 as cv
import numpy as np

def tumor_localization(mask: np.ndarray, PIXEL_SPACING: float):

    tumor_pixels = np.argwhere(mask == 1)

    if tumor_pixels.size == 0:
        raise ValueError("No tumor detected in the mask.")
    
    y_coords = tumor_pixels[:, 0]
    x_coords = tumor_pixels[:, 1]

    x_min, x_max = int(x_coords.min()), int(x_coords.max())
    y_min, y_max = int(y_coords.min()), int(y_coords.max())

    h, w = mask.shape

    distance_px = {
        "left": x_min,
        "right": w - x_max,
        "top": y_min,
        "bottom": h - y_max
    }

    distance_mm = {
        key: round(value * PIXEL_SPACING) for key, value in distance_px.items()
    }

    arrows = {
        "left" : {
            "start": { "x": 0, "y": (y_min + y_max) // 2 },
            "end": { "x": x_min, "y": (y_min + y_max) // 2 }
        },
        "right" : {
            "start" : { "x": w, "y": (y_min + y_max) // 2 },
            "end": { "x": x_max, "y": (y_min + y_max) // 2 }
        },
        "top" : {
            "start": { "x": (x_min + x_max) // 2, "y": 0 },
            "end": { "x": (x_min + x_max) // 2, "y": y_min }
        },
        "bottom" : {
            "start": { "x": (x_min + x_max) // 2, "y": h },
            "end": { "x": (x_min + x_max) // 2, "y": y_max }
        }
    }

    outputs = {
        "bounding_box": {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min
        },
        "distances_mm": distance_mm,
        "arrows": arrows
    }

    return outputs



"""
Returns:
{
  "bounding_box": {
    "x": 42,
    "y": 51,
    "width": 37,
    "height": 29
  },
  "distances_mm": {
    "left": 10.92,
    "right": 46.02,
    "top": 13.26,
    "bottom": 45.76
  },
  "arrows": {
    "left": {
      "start": { "x": 0, "y": 65 },
      "end": { "x": 42, "y": 65 }
    }
  }
}
"""