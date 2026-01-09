import logging
from zenml import step
import torch

@step
def evaluate_model(model, test_loader, criterion):
    """Evaluates the segmentation model.

    Args:
        model: The trained segmentation model.
        test_loader: DataLoader for the test dataset.
        criterion: Loss function.

    Returns:
        test_loss: The average loss on the test dataset.
    """
    logging.info("Starting model evaluation")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    logging.info(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss