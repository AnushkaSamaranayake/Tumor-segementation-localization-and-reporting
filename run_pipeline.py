from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Define paths to image and mask directories
    image_dir = "Data/images"
    mask_dir = "Data/masks"

    # Run the training pipeline
    train_pipeline(image_dir=image_dir, mask_dir=mask_dir)