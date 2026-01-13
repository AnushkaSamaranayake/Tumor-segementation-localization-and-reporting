import torch
import numpy as np
from PIL import Image
import base64
import io

class Prediction:
    def __init__(self, model, device, image, transform = None):
        self.model = model
        self.device = device
        self.image = image
        self.transform = transform


    def predict_mask(self):

        #set model on evaluate mode
        self.model.eval()

        if self.transform:
            image = self.transform(image)

        input_image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_image)
            output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        return output_mask

    def output_image(self, mask: np.ndarray) -> str:

        img = Image.fromarray((mask * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        
        return base64.b64decode(buf.getvalue()).decode("utf-8")
