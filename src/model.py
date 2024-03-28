

"""Lighning Module
From: https://github.com/bwconrad/vit-finetune/blob/main/src/model.py
"""


from typing import List, Optional, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR



#from torchmetrics import MetricCollection
#from torchmetrics.functional import f1_score
#from torchmetrics.classification.accuracy import Accuracy           #Confusion Matrix, FI Score
#from torchmetrics.classification.stat_scores import StatScores
#from pytorch_lightning.metrics.functional import accuracy

from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassF1Score

from transformers import AutoConfig, AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup




MODEL_DICT = {
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
}


class Shoe40kClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit-l15-224-in21k",
        optimizer: str = "sgd",
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        n_classes: int = 10,
        label_smoothing: float = 0.0,
        image_size: int = 224,
        weights: Optional[str] = None,
        training_mode: str = "full",
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: List[str] = ["query", "value"],
        lora_dropout: float = 0.0,
        lora_bias: str = "none",
    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
            lora_r: Dimension of LoRA update matrices
            lora_alpha: LoRA scaling factor
            lora_target_modules: Names of the modules to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            lora_bias: Whether to train biases during LoRA. One of ['none', 'all' or 'lora_only']
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing
        self.image_size = image_size
        self.weights = weights
        self.training_mode = training_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias

        # Initialize network
        try:
            model_path = MODEL_DICT[self.model_name]
        except:
            raise ValueError(
                f"{model_name} is not an available model. Should be one of {[k for k in MODEL_DICT.keys()]}"
            )

        # Initialize with pretrained weights
        self.net = AutoModelForImageClassification.from_pretrained(
            model_path,
            num_labels=self.n_classes,
            ignore_mismatched_sizes=True,
            image_size=self.image_size,
        )

        # Load checkpoint weights
        if self.weights:
            print(f"Loaded weights from {self.weights}")
            ckpt = torch.load(self.weights)["state_dict"]

            # Remove prefix from key names
            new_state_dict = {}
            for k, v in ckpt.items():
                if k.startswith("net"):
                    k = k.replace("net" + ".", "")
                    new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)


        #if self.training_mode == "lora":
            # Wrap in LoRA model
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            modules_to_save=["classifier"],
        )
        self.net = get_peft_model(self.net, config)
        

    def forward(self, x):
        return self.net(pixel_values=x).logits


    def _evaluate(self, batch, batch_idx, stage):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)
        f1 = f1_score(preds, y, num_classes=self.num_classes, average='macro')
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_acc', acc, prog_bar=True)
        self.log(f'{stage}_f1', f1, prog_bar=True)

        return loss, acc, f1

    def training_step(self, batch, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        loss, acc, f1 = self._evaluate(batch, "train")
        return loss

    def validation_step(self, batch, _):
        return self._evaluate(batch, "val")
    

    def configure_optimizers(self):
        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.net.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.net.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate scheduler
        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
