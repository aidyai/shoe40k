from typing import Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import get_cosine_schedule_with_warmup
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import timm


class Shoe40kClassificationModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 3,
        image_size: int = 384,
        learning_rate: float = 1e-4,
        model_name: str = "vit_base_patch16_384.augreg_in21k_ft_in1k",
        optimizer: str = "adam",
        lr: float = 1e-2,
        pretrained: bool = False,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        n_classes: int = 6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr = lr
        self.pretrained = pretrained
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes

        # Create the image classification model using timm library
        self.model = timm.create_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_classes=self.n_classes,
        )

        # Initialize metrics for evaluation
        self.accuracy = MulticlassAccuracy(num_classes=self.n_classes)
        self.recall = MulticlassRecall(num_classes=self.n_classes)
        self.precision = MulticlassPrecision(num_classes=self.n_classes)
        self.f1_score = MulticlassF1Score(num_classes=self.n_classes)

    def forward(self, x):
        # Forward pass of the model
        return self.model(x)

    def _compute_metrics(self, batch, split):
        # Compute metrics (loss, accuracy, f1_score, recall, precision) for a given batch and split
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)

        metrics = {
            f"{split}_Loss": loss,
            f"{split}_Acc": self.accuracy(preds=preds, target=y),
            f"{split}_f1_score": self.f1_score(preds=preds, target=y),
            f"{split}_recall": self.recall(preds=preds, target=y),
            f"{split}_precision": self.precision(preds=preds, target=y),
        }

        return loss, metrics

    def training_step(self, batch, batch_idx):
        # Training step
        loss, metrics = self._compute_metrics(batch, "train")
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        _, metrics = self._compute_metrics(batch, "val")
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return metrics

    def configure_optimizers(self):
        # Configure optimizer and learning rate scheduler
        optimizer_params = {
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
        }

        if self.optimizer == "adam":
            optimizer = Adam(self.parameters(), **optimizer_params)
        elif self.optimizer == "adamw":
            optimizer = AdamW(self.parameters(), **optimizer_params)
        elif self.optimizer == "sgd":
            optimizer = SGD(self.parameters(), **optimizer_params, momentum=self.momentum)
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.max_steps,
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available scheduler. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
