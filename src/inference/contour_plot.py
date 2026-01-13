import cv2 as cv
import numpy as np

def plot_contour_boundry(predicted_mask: np.ndarray):

    contours, _ = cv.findContours(predicted_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contour_paths = []

    for contour in contours:
        path = [
            {"x":int(point[0][0]), "y":int(point[0][0])} for point in contour
        ]
        contour_paths.append(path)

    return contour_paths


"""
Returns:
{
  "contours": [
    [
      { "x": 120, "y": 45 },
      { "x": 121, "y": 46 },
      { "x": 123, "y": 49 }
    ]
  ]
}
"""