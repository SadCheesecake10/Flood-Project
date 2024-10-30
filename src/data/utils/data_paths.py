import os
import numpy as np
from glob import glob

def get_df_paths(path, set_type):
    """
    Retrieve paths of VV and VH channels, flood labels, and water body labels.
    
    Parameters:
    - path (str): Base directory for data.
    - set_type (str): Type of dataset ('train' or 'val').

    Returns:
    - dict: Paths for VV, VH channels, flood labels, and water body labels.
    """
    path_pattern = os.path.join(path, "**", "tiles", "vv", "*.png")
    vv_channel_paths = sorted(glob(path_pattern, recursive=True))
    vv_channel_names = [os.path.split(pth)[1] for pth in vv_channel_paths]
    region_name_dates = ["_".join(name.split("_")[:2]) for name in vv_channel_names]

    flood_labels = [
        os.path.join(path, date, "tiles", "flood_label", name.replace("_vv", ""))
        if set_type == "train" else np.nan
        for date, name in zip(region_name_dates, vv_channel_names)
    ]

    return {
        "region_name": [date.split("_")[0] for date in region_name_dates],
        "vv_channel": vv_channel_paths,
        "vh_channel": [
            os.path.join(path, date, "tiles", "vh", name.replace("vv", "vh"))
            for date, name in zip(region_name_dates, vv_channel_names)
        ],
        "flood_label": flood_labels,
        "water_body_label": [
            os.path.join(path, date, "tiles", "water_body_label", name.replace("_vv", ""))
            for date, name in zip(region_name_dates, vv_channel_names)
        ]
    }
