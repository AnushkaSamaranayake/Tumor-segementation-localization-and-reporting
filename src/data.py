import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Annotated
from PIL import Image
import os
from torchvision import transforms


class DataAugmentation:

    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.mask_transform = transforms.Compose([
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


def create_dataset(image_dir: str, mask_dir: str) -> SegmentationDataset:
    """Creates a segmentation dataset.

    Args:
        image_dir (str): Directory containing the input images.
        mask_dir (str): Directory containing the corresponding masks.

    Returns:
        SegmentationDataset: A dataset object containing preprocessed images and masks.
    """
    data_augmentation = DataAugmentation()
    dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_transform=data_augmentation.image_transform,
        mask_transform=data_augmentation.mask_transform
    )
    return dataset

def create_data_loaders(dataset: SegmentationDataset) -> Tuple[
    Annotated[DataLoader, "Training DataLoader"],
    Annotated[DataLoader, "Validation DataLoader"],
    Annotated[DataLoader, "Test DataLoader"]
]:
    """Splits the dataset into training, validation, and test sets and creates DataLoaders.

    Args:
        dataset (SegmentationDataset): The segmentation dataset.
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and test sets.
    """

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader
