import logging
from zenml import pipeline
from steps.ingest_data import ingest_data, create_data_loaders
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def train_pipeline(image_dir: str, mask_dir: str):
    """Pipeline for training and evaluating a segmentation model.

    Args:
        image_dir (str): Directory containing the input images.
        mask_dir (str): Directory containing the corresponding masks.
    """
    # Ingest and preprocess data
    dataset = ingest_data(image_dir=image_dir, mask_dir=mask_dir)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(dataset=dataset)

    # Define model, criterion, optimizer, and number of epochs
    model = None  # Define your segmentation model here
    criterion = None  # Define your loss function here
    optimizer = None  # Define your optimizer here
    num_epochs = None  # Set the number of epochs

    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs
    )

    # Evaluate the model
    evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        criterion=criterion
    )

