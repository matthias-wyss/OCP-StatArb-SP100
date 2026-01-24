# OCP-StatArb-SP100

Statistical arbitrage strategy based on the **Optimal Causal Path (OCP)** algorithm, applied to high-frequency tick-level data (BBO) of S&P 100 constituents.

## 📌 Project Overview
This project implements a lead-lag relationship detection system between S&P 100 stocks. Using the OCP algorithm, it identifies "Leader" and "Follower" pairs in high-frequency mid-price returns to build a predictive trading strategy.

## 📂 Data Setup & Directory Structure

### 1. Raw Data Requirements
The preprocessing pipeline requires raw tick data. 
- **Download**: Get the raw dataset here: [Download Raw SP100 BBO Data (ZIP)](TODO)
- **Setup**: Create a `data/raw/SP100/bbo/` directory.
- **Action**: Unzip the downloaded file and place the `.tar` files inside this folder.

### 2. Full Project Hierarchy
The following structure **must exist** in the project root for the scripts to execute correctly. Even if folders like `selected` or `top_pairs` are initially empty, the pipeline expects them to be present:

```text
OCP-StatArb-SP100/
├── data/
│   ├── raw/
│   │   └── SP100/
│   │       └── bbo/       <-- Place downloaded .tar files here
│   ├── extracted/         <-- Mandatory: Extracted raw data archives
│   ├── temp/              <-- Mandatory: Intermediate preprocessing files
│   ├── preprocessed/      <-- Mandatory: Cleaned 1-min mid-price returns
│   ├── selected/          <-- Mandatory: Validated tickers for analysis
│   └── top_pairs/         <-- Mandatory: OCP lead-lag results
├── src/
│   ├── ocp_lag_analysis.py <-- For plotting
│   ├── ocp.py              <-- Core OCP algorithm
│   ├── preprocessing.py    <-- Data cleaning and resampling
│   └── trading.py          <-- Backtesting and strategy execution
├── pdfs/
│   ├── Financial_Big_Data___Project_Proposal.pdf <--- Our project proposal
│   ├── Report.pdf <--- Our report
│   └── Statistical arbitrage with optimal causal paths...pdf <-- Research paper
├── logs/                  <-- Mandatory: Execution logs and debugging info
├── output/                <-- Mandatory: Saved plots and strategy results
├── analysis.ipynb         <-- Main entry point (Run this)
├── requirements.txt       <-- Project dependencies
├── LICENSE                <-- Project license
└── README.md              <-- This file
```

## 🚀 Getting Started

### 1. Installation
Install the required dependencies (ensure you are in your virtual environment): `pip install -r requirements.txt`

### 2. Running the Analysis
The entire workflow is managed via the main Jupyter notebook: ```analysis.ipynb```

### 3. Execution Modes
In the first cells of the notebook, you can set these flags:
- `PARTIAL_RUN = True`: Processes a subset of tickers/days for fast testing.
- `PERFORM_PREPROCESSING = True`: Clears internal folders and re-runs the full preprocessing pipeline from raw data.

