from src.main import train

if __name__ == '__main__':
    # Define variables
    batch_size = 32
    epochs = 10
    csv_path = ''
    dataset_path = ''

    # Call the train function
    train(batch_size, epochs, csv_path, dataset_path)
