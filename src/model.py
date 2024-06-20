        








class LitViTModel(pl.LightningModule):
    def __init__(self, num_classes=10, lr=1e-3):
        super(LitViTModel, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes
        
        # Load pre-trained ViT model
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_classes)
        
        # Define metrics
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(num_classes=n_classes, top_k=1, task='multiclass')
        self.f1score = torchmetrics.F1Score(num_classes=n_classes, top_k=1, task='multiclass')
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x) 
        
        
    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, 'train')
        
        accuracy = self.accuracy(y_hat.argmax(1), y)
        f1_score = self.f1score(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1score': f1_score}, prog_bar=True,
                     on_step=False, on_epoch=True)
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx, 'valid')
        accuracy = self.accuracy(y_hat, y)
        f1_score = self.f1score(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1score': f1_score}, prog_bar=True,
                     on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_accuracy': accuracy, 'val_f1score': f1_score}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    
    def _common_step(self, batch, batch_idx, split='train'):
        x, y = batch
        if self.current_epoch == 0:
            self.logger.log_image(f'{split}_grid', images=[x])
        
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y
        
    def predict(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        preds = torch.argmax(y_hat, dim=1)
        return y_hat, preds

    def on_train_epoch_end(self) -> None:
        print("\n")
        
        
        
        
        
        
        
        
