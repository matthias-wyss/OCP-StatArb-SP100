# OCP-StatArb-SP100

Statistical arbitrage strategy based on the **Optimal Causal Path (OCP)** algorithm, applied to high-frequency tick-level data (BBO) of S&P 100 constituents.

## 📌 Project Overview
This project implements a lead-lag relationship detection system between S&P 100 stocks. Using the OCP algorithm, it identifies "Leader" and "Follower" pairs in high-frequency mid-price returns to build a predictive trading strategy.

## 📂 Data Setup
The preprocessing pipeline requires raw tick data in Parquet format.

1. Create a `data/` folder in the project root.
2. Download the raw BBO data (link to be provided).
3. Place the folder `SP100/bbo/` inside `data/raw/`:
   `data/raw/SP100/bbo/*.tar`

## 🚀 Getting Started

### 1. Installation
Install the required dependencies:
pip install -r requirements.txt

### 2. Running the Analysis
The entire workflow is contained within the main Jupyter notebook:
**analysis.ipynb**

### 3. Execution Modes
You can configure the behavior of the notebook using these flags:
* PARTIAL_RUN = True: Runs the analysis on a small subset of tickers (for fast debugging).
* PERFORM_PREPROCESSING = True: Clears previous cache and re-processes all raw data.

## 🛠 Project Structure
* analysis.ipynb: Main execution entry point.
* src/: Core logic modules (preprocessing, OCP engine).
* data/:
    * raw/: Raw tick-level data.
    * extracted/: Extracted raw data.
    * temp/: Temporary preprocessing folder.
    * preprocessed/: Resampled 1-minute returns.
    * selected/: Selected tickers from preprocessed folder for analysis.
    * top_pairs/: Identified lead-lag relationships.
