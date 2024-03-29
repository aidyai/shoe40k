from src.main import train

if __name__ == '__main__':
    # Define variables
    batch_size = 32
    epochs = 10
    csv_path = '/content/FOOT40K.csv'
    dataset_path = '/content/FOOT40k'

    # Call the train function
    train(batch_size, epochs, csv_path, dataset_path)
