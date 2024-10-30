import numpy as np
import pandas as pd
from src.data.utils.data_paths import get_df_paths
import os

def prepare_datasets(train_path, test_path):
    """
    Prepare and save training, validation, and test datasets as CSV files.

    Parameters:
    - train_path (str): Path to training data.
    - test_path (str): Path to test/validation data.
    """
    df_train_paths = pd.DataFrame(get_df_paths(train_path, set_type="train"))
    df_test_paths = pd.DataFrame(get_df_paths(test_path, set_type="val"))

    np.random.seed(0)
    region_choice = np.random.choice(df_train_paths["region_name"].unique(), 1)[0]

    df_train_paths_new = df_train_paths[df_train_paths["region_name"] != region_choice]
    df_valid_paths = df_train_paths[df_train_paths["region_name"] == region_choice]

    # Save DataFrames to CSV
    if not os.path.exists("data/csv"):
        os.makedirs("data/csv")
    df_train_paths.to_csv("data/csv/df_train_paths.csv", index=False)
    df_train_paths_new.to_csv("data/csv/df_train_paths_new.csv", index=False)
    df_valid_paths.to_csv("data/csv/df_valid_paths.csv", index=False)
    df_test_paths.to_csv("data/csv/df_test_paths.csv", index=False)
