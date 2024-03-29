import os
import torch

from PIL import Image
import pandas as pd
from torch.nn.modules import transformer
import torchvision.transforms as T

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import LightningDataModule




class Shoe40kTransforms(T.Compose):
    def __init__(self, phase):
        self.phase = phase
        self.transforms = {
            'train': [
                T.Resize((32, 32)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ],
            'val': [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ],
            'test': [
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        }
        
        super().__init__(self.transforms[self.phase])




class Shoe40kDataset(Dataset):
    def __init__(self, df, path, phase):
        self.df = df
        self.path = path
        self.file_names = df['file_name'].values
        self.labels = df['Label'].values
        self.phase = phase
        self.transform = Shoe40kTransforms(phase=phase)

        
    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.path, file_name)
        
        # Load the image using PIL.Image.open()
        with Image.open(file_path) as image:
            # Convert the image to RGB
            image = image.convert('RGB')
        
        
        # Convert the PIL image to a PyTorch tensor
        tensor_image = transforms.ToTensor()(image)

        # Apply the transformation (if any) to the tensor
        if self.transform:
            tensor_image = self.transform(tensor_image)
                        
        # Convert the label to a PyTorch tensor
        label = torch.tensor(self.labels[idx]).long()

        return tensor_image, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
    
    

class Shoe40kDataModule(LightningDataModule):
    
    def __init__(self, csv_path, dataset_path, batch_size):
        super().__init__()
        self.csv_path = csv_path
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        
        # Load your CSV file containing image filenames and labels
        df = pd.read_csv(self.csv_path)
        
        df['Label'] = df['Label'].apply(lambda x: x[0:4]).astype('category').cat.codes

        # Split dataset into training and testing sets with stratified sampling
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
 
        self.train_dataset = Shoe40kDataset(df=train_df, path=self.dataset_path, phase='train')
        self.val_dataset = Shoe40kDataset(df=val_df, path=self.dataset_path, phase='val')

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size, 
                          num_workers=12, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size, 
                          num_workers=12, shuffle=False)

