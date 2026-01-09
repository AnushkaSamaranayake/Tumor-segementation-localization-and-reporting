from zenml import step
import logging
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Annotated
import os

# Data Augmentation
image_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

mask_transform = transforms.Compose([
    transforms.Resize((256,256), interpolation=Image.NEAREST),
    transforms.ToTensor()
])

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

@step
def ingest_data(image_dir: str, mask_dir: str) -> SegmentationDataset:
    """Ingests and preprocesses the image and mask data for segmentation tasks.

    Args:
        image_dir (str): Directory containing the input images.
        mask_dir (str): Directory containing the corresponding masks.

    Returns:
        SegmentationDataset: A dataset object containing preprocessed images and masks.
    """
    logging.info(f"Ingesting data from {image_dir} and {mask_dir}")
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    logging.info(f"Dataset contains {len(dataset)} samples.")
    return dataset

@step
def create_data_loaders(dataset: SegmentationDataset) -> Tuple[
    Annotated[DataLoader, "train_loader"],
    Annotated[DataLoader, "val_loader"],
    Annotated[DataLoader, "test_loader"]
]:
    """Creates training and validation data loaders from the dataset.

    Args:
        dataset (SegmentationDataset): The dataset containing images and masks.

    Returns:
        train_loader, val_loader and test_loader: Data loaders for training, validation, and testing with ratio 70:15:15.
    """
    logging.info("Creating data loaders")

    dataset_size = len(dataset)

    train_size = int(0.70 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - (train_size + val_size)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader