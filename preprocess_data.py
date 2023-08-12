# Import Libaries
from torch.utils.data import DataLoader, TensorDataset

# Map target and label data together in a tuple format
def preprocess_data(label_X, target_y):
    preprocessed= TensorDataset(label_X, target_y)
    return preprocessed

# Create data loaders
def dataloader(dataset, batch_size, shuffle, num_workers):
    dataloader= DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle= shuffle,
                           num_workers=num_workers,
                           )
    return (dataloader)

