#main_test.py
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
import warnings

from src.data.datasets.s1_dataset import S1Dataset
from src.utils.visualization import plot_prediction
from src.utils.metrics import evaluate_map50, evaluate_map50_95

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# Predict function
def predict(model, device, test_dataloader):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting"):
            image = batch["image"].to(device)
            pred = model(image)
            pred_class = torch.argmax(pred, dim=1).cpu().numpy().astype("uint8")
            predictions.extend(pred_class)

    return np.array(predictions)



def plot_map_results(map_50_values, map_values):
    # Plot para mAP@0.5
    plt.figure(figsize=(10, 5))
    plt.plot(map_50_values, label="mAP@0.5", color="blue")
    plt.xlabel("Batch")
    plt.ylabel("mAP@0.5")
    plt.title("Mean Average Precision (mAP@0.5) per Batch")
    plt.legend()
    plt.savefig("output/map_50_plot.png")  # Guarda el gráfico en la carpeta output
    plt.close()

    # Plot para mAP
    plt.figure(figsize=(10, 5))
    plt.plot(map_values, label="mAP", color="green")
    plt.xlabel("Batch")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision (mAP) per Batch")
    plt.legend()
    plt.savefig("output/map_plot.png")  # Guarda el gráfico en la carpeta output
    plt.close()


if __name__ == "__main__":
    # Load test paths
    df_test_paths = pd.read_csv("data/csv/df_test_paths.csv")

    # Set device and load model weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=2, encoder_weights=None)
    try:
        model.load_state_dict(torch.load("model.pt"))
        model.to(device)
    except FileNotFoundError:
        raise FileNotFoundError("Model weights not found. Please train the model first.")
    
    # Create test dataloader
    test_set = S1Dataset(df_test_paths, flood_label=False)
    test_dataloader = DataLoader(test_set, batch_size=64, pin_memory=True, shuffle=False, num_workers=0)

    # Make predictions of test set
    predictions = predict(model, device, test_dataloader)

    # Plot some predictions of the test set
    plot_prediction(df_test_paths.iloc[1905], predictions[1905])
    plot_prediction(df_test_paths.iloc[2967], predictions[2967])
    plot_prediction(df_test_paths.iloc[9100], predictions[9100])
    plot_prediction(df_test_paths.iloc[1945], predictions[1945])
    plot_prediction(df_test_paths.iloc[6457], predictions[6457])
    
    map50 = evaluate_map50(model, test_dataloader, device)
    map50_95= evaluate_map50_95(model, test_dataloader, device)
    plot_map_results(map50, map50_95)
    
