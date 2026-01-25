import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from typing import Dict, Tuple


class UNet3DSegmenter:
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        self.model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
        ).to(self.device)

        self.criterion = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )

        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )

        self.optimizer = None
        self.scheduler = None

    def setup_training(self, lr: float = 1e-4):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def forward(self, images):
        return self.model(images)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()

        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(images)

        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            dice = self.dice_metric(preds, labels)

        return {
            "loss": loss.item(),
            "dice": dice.mean().item(),
        }

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()

        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1, keepdim=True)
            dice = self.dice_metric(preds, labels)

        return {
            "loss": loss.item(),
            "dice": dice.mean().item(),
        }

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        if image.ndim == 4:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy().squeeze(), probs.cpu().numpy().squeeze()
class SegmentationTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_epoch(self):
        loss, dice = 0, 0
        for batch in self.train_loader:
            m = self.model.train_step(batch)
            loss += m["loss"]
            dice += m["dice"]
        return loss / len(self.train_loader), dice / len(self.train_loader)

    def validate_epoch(self):
        loss, dice = 0, 0
        for batch in self.val_loader:
            m = self.model.validate_step(batch)
            loss += m["loss"]
            dice += m["dice"]
        return loss / len(self.val_loader), dice / len(self.val_loader)
def main():
    model = UNet3DSegmenter()
    print("UNet3D model created successfully")


if __name__ == "__main__":
    main()
