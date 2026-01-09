import logging
from zenml import step
import torch

## Define the model

@step
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs: int) -> None:
    """Trains a segmentation model.

    Args:
        model: The segmentation model to be trained.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        The trained model.
    """
    logging.info("Starting model training")
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #     for images, masks in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, masks)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()

    #     avg_loss = running_loss / len(train_loader)
    #     logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    #     # Validation step
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for images, masks in val_loader:
    #             outputs = model(images)
    #             loss = criterion(outputs, masks)
    #             val_loss += loss.item()

    #     avg_val_loss = val_loss / len(val_loader)
    #     logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # logging.info("Model training completed")
    # return model
    pass
