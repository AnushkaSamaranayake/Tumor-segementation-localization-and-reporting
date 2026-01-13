import os
import torch
import mlflow
from evaluate import evaluate

evaluation_metrics = evaluate

class Trainer:
    def __init__(self, device,train_loader, val_loader, model, criterion, optimizer, num_epochs=25):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def train(self, params):

        with mlflow.start_run():
            mlflow.log_params(params)

            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss,running_dice = 0.0, 0.0
                num_batches = 0  #number of trained batches = len of train_loader
                best_dice = 0.0

                for images, masks in self.train_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(images)

                    loss = self.criterion(outputs, masks)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item()
                    running_dice += evaluation_metrics(outputs, masks).dice_score().item()
                    num_batches += 1

                epoch_train_loss = running_loss / num_batches
                epoch_train_dice = running_dice / num_batches
                epoch_val_loss, epoch_val_dice = self.validate()

                if epoch_val_dice > best_dice:
                    best_dice = epoch_val_dice
                    torch.save(self.model.state_dict(), os.path.join("Tumor-segementation-localization-and-reporting/model","best_model.pth"))

                mlflow.log_metric({
                    "train_loss": epoch_train_loss,
                    "train_dice": epoch_train_dice,
                    "val_loss": epoch_val_loss,
                    "val_dice": epoch_val_dice
                }, step=epoch)

                print(f"Epoch [{epoch+1}/{self.num_epochs}], Train dice: {epoch_train_dice:.4f}, Val dice: {epoch_val_dice:.4f}")
            print("Training complete.")
            mlflow.pytorch.log_model(self.model, "model")

    def validate(self):
        self.model.eval()
        running_loss, running_dice = 0.0, 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                running_loss += loss.item()
                running_dice += evaluation_metrics(outputs, masks).dice_score().item()
                num_batches += 1

        val_loss = running_loss / num_batches
        val_dice = running_dice / num_batches
        return val_loss, val_dice
