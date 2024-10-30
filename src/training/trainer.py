import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.metrics import evaluate_model

def train_model(model, train_dataloader, validation_dataloader, device, epochs, lr=0.001):
    """
    Train the model with the specified parameters.

    Parameters:
    - model: PyTorch model to train.
    - train_dataloader: DataLoader for training data.
    - validation_dataloader: DataLoader for validation data.
    - device: Device to use for training (e.g., 'cuda' or 'cpu').
    - epochs: Number of epochs.
    - lr: Learning rate.

    Returns:
    - model: Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    map50_list, map50_95_list = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}:")
        model.train()

        for batch in tqdm(train_dataloader, desc="Training"):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            pred = model(image)
            loss = loss_function(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Evaluating on validation set...")
        map50, map50_95 = evaluate_model(model, validation_dataloader, device)
        print(f"Epoch {epoch + 1}: mAP@50: {map50:.4f}, mAP@50-95: {map50_95:.4f}")
        map50_list.append(map50)
        map50_95_list.append(map50_95)

    plt.figure()
    plt.plot(range(epochs), map50_list, label="mAP50")
    plt.plot(range(epochs), map50_95_list, label="mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP Metrics")
    plt.legend()
    plt.savefig("map_metrics.png")
    plt.show()

    return model
