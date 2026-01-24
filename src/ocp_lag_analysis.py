import matplotlib.pyplot as plt
import polars as pl
import numpy as np
import os
from pathlib import Path

def show_lag_histogram(pairs_path: Path) -> None:
    """
    Generates and saves a histogram of the optimal lags (l_hat) from the pairs data.
    The histogram is styled to match the academic report format.
    """

    # ==========================================
    # 1. SETUP PATHS
    # ==========================================
    # Make sure this matches exactly where your pairs file is located
    output_dir = "data/results_plots"

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading pairs from: {pairs_path}")
    print(f"Saving plot to: {os.path.abspath(output_dir)}")

    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    if not pairs_path.exists():
        raise FileNotFoundError(f"Could not find the file: {pairs_path}")

    # Load the pairs data
    pairs_df = pl.read_parquet(pairs_path).to_pandas()

    # ==========================================
    # 3. GENERATE LAG HISTOGRAM
    # ==========================================
    plt.figure(figsize=(10, 6))

    # We use bins from 0 to 30 because your max_lag is 30



    plt.hist(pairs_df['l_hat'], bins=np.arange(0, 31, 1), 
            color='#2ca02c', edgecolor='black', alpha=0.75, rwidth=0.9)

    # Formatting to match your academic report style
    plt.title('Distribution of Optimal Lags ($\hat{l}$)', fontsize=14)
    plt.xlabel('Estimated Lag (Minutes)', fontsize=12)
    plt.ylabel('Frequency (Count of Pairs)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(np.arange(0, 31, 2))  # Tick marks every 2 minutes for clarity
    plt.tight_layout()

    # ==========================================
    # 4. SAVE PLOT
    # ==========================================
    save_path = f"{output_dir}/lag_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Success! Plot saved to: {save_path}")
    plt.close()