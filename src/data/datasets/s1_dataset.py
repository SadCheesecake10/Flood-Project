import cv2
import numpy as np
from torch.utils.data import Dataset
from src.data.utils.image_utils import load_and_normalize_grayscale_image

class S1Dataset(Dataset):
    def __init__(self, dataset, flood_label, transform=None):
        """
        Dataset class for Sentinel-1 data with optional flood label.
        
        Parameters:
        - dataset (DataFrame): Pandas DataFrame with paths to VV and VH channels and optional flood labels.
        - flood_label (bool): Whether the dataset includes flood labels.
        - transform (callable, optional): Optional transform to be applied on an image and mask.
        
        Returns:
        - image (torch.Tensor): RGB image tensor.
        """
        self.dataset = dataset
        self.flood_label = flood_label
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.iloc[index]
        output = {}

        # Load channels
        vv_channel = load_and_normalize_grayscale_image(sample["vv_channel"])
        vh_channel = load_and_normalize_grayscale_image(sample["vh_channel"])

        # Resize channels if necessary
        if vv_channel.shape != vh_channel.shape:
            vv_channel = cv2.resize(vv_channel, (256, 256))
            vh_channel = cv2.resize(vh_channel, (256, 256))

        # Calculate ratio and create RGB stack
        ratio = np.clip(np.nan_to_num(vv_channel / vh_channel, nan=0), 0, 1)
        rgb = np.stack((vv_channel, vh_channel, 1 - ratio), axis=2).astype("float32")

        if rgb.shape != (256, 256, 3):
            raise ValueError(f"Unexpected shape for rgb: {rgb.shape}")

        # Prepare output without flood label
        if not self.flood_label:
            output["image"] = rgb.transpose((2, 0, 1))
            return output

        # Load flood mask and apply transform if available
        flood_mask = load_and_normalize_grayscale_image(sample["flood_label"])
        if self.transform:
            rgb, flood_mask = self.transform(rgb, flood_mask)

        output["image"] = rgb.transpose((2, 0, 1))
        output["mask"] = flood_mask.astype("int64")

        return output
