import gc
import torch
from src.main import train

if __name__ == '__main__':
    # Free up gpu vRAM from memory leaks.
    torch.cuda.empty_cache()
    gc.collect()
    
    # Define variables
    batch_size = 12
    epochs = 10
    csv_path = '/content/FOOT40K.csv'
    dataset_path = '/content/FOOT40k'

    # Call the train function
    train(batch_size, epochs, csv_path, dataset_path)
