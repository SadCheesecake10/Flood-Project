import numpy as np
import matplotlib.pyplot as plt
from src.data.utils.image_utils import load_and_normalize_grayscale_image

def plot_maps(sample):
    """
    Plot VV and VH channels, combined ratio, RGB image, water body label, and flood label.

    Parameters:
    - sample (dict): Sample with paths to VV, VH channels and labels.
    """
    vv_channel = load_and_normalize_grayscale_image(sample["vv_channel"])
    vh_channel = load_and_normalize_grayscale_image(sample["vh_channel"])

    ratio = np.clip(np.nan_to_num(vh_channel / vv_channel, nan=0), 0, 1)
    rgb = np.stack((vv_channel, vh_channel, 1 - ratio), axis=2)

    water_body_label = load_and_normalize_grayscale_image(sample["water_body_label"])
    flood_label = load_and_normalize_grayscale_image(sample["flood_label"])

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    axs[0, 0].imshow(vv_channel)
    axs[0, 0].set_title("Normalized VV channel")
    axs[0, 1].imshow(vh_channel)
    axs[0, 1].set_title("Normalized VH channel")
    axs[1, 0].imshow(ratio)
    axs[1, 0].set_title("Combined VV and VH image")
    axs[1, 1].imshow(rgb)
    axs[1, 1].set_title("RGB image")
    axs[2, 0].imshow(water_body_label)
    axs[2, 0].set_title("Water body label")
    axs[2, 1].imshow(flood_label)
    axs[2, 1].set_title("Flood label")

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
