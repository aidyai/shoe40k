import os
import yaml
import gc
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.dataset import Shoe40kDataModule
from src.model import Shoe40kClassificationModel
from src.image_logger import ImagePredictionLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint




def load_config_from_yaml(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Resolve relative paths if necessary
    config['csv_path'] = os.path.abspath(config['csv_path'])
    config['dataset_path'] = os.path.abspath(config['dataset_path'])

    return config


def train(config_file_path: str):

    # Load experiment configuration from YAML file
    wandb_config = load_config_from_yaml(config_file_path)

    # Instantiate LightningDataModule to determine dataset sizes
    data_module = Shoe40kDataModule(wandb_config['csv_path'], wandb_config['dataset_path'], batch_size=wandb_config['batch_size'])

    # DataLoaders for both the train and validation datasets
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(val_loader))
    
    # dataset related configuration parameters
    wandb_config['train_batches'] =  len(train_loader)
    wandb_config['val_batches'] = len(val_loader)
    wandb_config['dataset_train_size'] = len(train_loader.dataset)
    wandb_config['dataset_val_size'] = len(val_loader.dataset)

    # Initialize the Lightning module
    model = Shoe40kClassificationModel()

    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_f1_score")
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{val_f1_score:.2f}",
        monitor="val_f1_score",
        mode="max",
        verbose=True,
        save_top_k=1,
    )

    # Download the checkpoint artifact and resume training
    resume_run_id = True #wandb_config.get('resume_run_id', '')
    if resume_run_id:
        api = wandb.Api()
        artifact = api.artifact(f"{wandb_config['project']}/model-{wandb_config['id']}:latest", type="model")
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, "model.ckpt")
        model = Shoe40kClassificationModel.load_from_checkpoint(checkpoint_path)

        # Initialize WandbLogger for resume training
        wandb_logger = WandbLogger(
            #entity=wandb_config['entity'],
            id=wandb_config['id'],
            project=wandb_config['project'],
            job_type='train',
            config=wandb_config,
            log_model="all",
            #resume="allow"
        )

        trainer = pl.Trainer(
            enable_checkpointing=True,
            enable_model_summary=True,
            callbacks=[early_stop_callback, checkpoint_callback],
            #resume_from_checkpoint=checkpoint_path,
            accelerator=wandb_config['accelerator'],
            logger=wandb_logger
        )

        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

    else:
        # Initialize WandbLogger for new training run
        wandb_logger = WandbLogger(
            entity=wandb_config['entity'],
            project=wandb_config['project'],
            job_type='train',
            config=wandb_config,
            log_model="all"
        )

        trainer = pl.Trainer(
            enable_checkpointing=True,
            enable_model_summary=True,
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=wandb_config['max_epochs'],
            min_epochs=wandb_config['min_epochs'],
            accelerator=wandb_config['accelerator'],
            logger=wandb_logger
        )

        trainer.fit(model, train_loader, val_loader)


    
    # Close wandb run
    wandb.finish()

   
if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()  
    config_file_path = '/content/shoe40k/config.yaml'  
    train(config_file_path)
