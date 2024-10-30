import matplotlib.pyplot as plt
import numpy as np
from src.data.utils.image_utils import load_and_normalize_grayscale_image

def plot_prediction(sample, prediction):
    vv_channel = load_and_normalize_grayscale_image(sample["vv_channel"])
    vh_channel = load_and_normalize_grayscale_image(sample["vh_channel"])

    ratio = np.clip(np.nan_to_num(vv_channel / vh_channel, nan=0), 0, 1)
    rgb_original = np.stack((vv_channel, vh_channel, 1 - ratio), axis=2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(rgb_original)
    axs[0].set_title("Original RGB Image")
    axs[1].imshow(prediction)
    axs[1].set_title("Flood Prediction")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
