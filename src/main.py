import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from .dataset import Shoe40kDataModule
from .model import Shoe40kClassificationModel
from .image_logger import ImagePredictionLogger

def train(batch_size: int, 
          epochs: int,
          csv_path: str,
          dataset_path: str,
          ):
    
    """Trains a PyTorch Lightning model using Weights and Biases"""

    # Instantiate the model
    model = Shoe40kClassificationModel()

    # Instantiate the LightningDataModule
    data_module = Shoe40kDataModule(csv_path, dataset_path, batch_size=batch_size)

    # Load the DataLoaders for both the train and validation datasets
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(val_loader))

    # Create the configuration of the current run
    wandb_config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'dataset': 'shoe40k',
        'dataset_train_size': len(data_module.train_dataloader()),
        'dataset_val_size': len(data_module.val_dataloader()),
        'input_shape': '[3,344,344]',
        'channels_last': False,
        'criterion': 'CrossEntropyLoss',
        'optimizer': 'Adam'
    }

    # Init the PyTorch Lightning WandbLogger (you need to `wandb login` first!)
    wandb_logger = WandbLogger(project='shoe40k', job_type='train', config=wandb_config)

    # Initialize Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")

    trainer = pl.Trainer(enable_checkpointing=True,
                     enable_model_summary=True,
                     devices=[0],
                     callbacks=[early_stop_callback,
                                ImagePredictionLogger(val_samples)],
                     max_epochs=50,
                     min_epochs=1,
                     logger=wandb_logger,
                     accelerator="gpu"
          )



    trainer.fit(model, train_loader, val_loader)

    # Close wandb run
    wandb.finish()

