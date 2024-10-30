import torch
import pandas as pd
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import warnings

from src.data.preparation.prepare_datasets import prepare_datasets
from src.data.datasets.s1_dataset import S1Dataset
from src.data.transforms.custom_transform import CustomTransform
from src.training.trainer import train_model

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Load data paths
    try:
        df_train_paths = pd.read_csv("data/csv/df_train_paths_new.csv")
        df_valid_paths = pd.read_csv("data/csv/df_valid_paths.csv")
    except FileNotFoundError:
        prepare_datasets("data/train", "data/val")
        df_train_paths = pd.read_csv("data/csv/df_train_paths_new.csv")
        df_valid_paths = pd.read_csv("data/csv/df_valid_paths.csv")

    # Set custom transform for data augmentation
    custom_transform = CustomTransform(width=256, height=256)

    # Initialize datasets and dataloaders for training and validation
    train_set = S1Dataset(df_train_paths, flood_label=True, transform=custom_transform)
    validation_set = S1Dataset(df_valid_paths, flood_label=True)
    train_dataloader = DataLoader(train_set, batch_size=64, pin_memory=True, shuffle=False, num_workers=0)
    validation_dataloader = DataLoader(validation_set, batch_size=64, pin_memory=True, shuffle=False, num_workers=0)

    # Set device and model configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=2, encoder_weights=None).to(device)

    # Train the model and validate
    trained_model = train_model(model, train_dataloader, validation_dataloader, device, epochs=20)

    # Save the trained model
    torch.save(trained_model.state_dict(), "model.pt")
