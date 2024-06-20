import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

def checkpoint_callback():
    return ModelCheckpoint(monitor="val_loss", mode="min", every_n_epochs=1, save_top_k=5, save_last=True,
                           filename="model-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.pth")


def early_stop():
    return EarlyStopping(monitor="val_loss", mode="min", patience=5)