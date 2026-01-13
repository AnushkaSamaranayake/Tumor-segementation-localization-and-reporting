import os
import mlflow
import torch

from src.data import SegmentationDataset, DataAugmentation, DataLoading
from src.train import Trainer
from src.model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_dir = "Data/images"
mask_dir = "Data/masks"

data_augmentation = DataAugmentation()

dataset = SegmentationDataset(
    image_dir=img_dir,
    mask_dir=mask_dir,
    image_transform=data_augmentation.image_transform,
    mask_transform=data_augmentation.mask_transform
)

train_loader, val_loader, test_loader = DataLoading.data_loaders(dataset)

params = {
    "epochs": 50,
    "learning_rate":1e-4,
    "batch_size":8
}

model = UNet().to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

mlflow.set_experiment("Tumor segmentation with UNet")

trainer = Trainer(
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=params["epochs"]
)

trainer.train(params)

