# Golden-Free Runtime Hardware Trojan & Anomaly Detection

A Siamese LSTM-based neural network for detecting hardware trojans and anomalies in runtime data using contrastive learning.

## Setup

### 1. Create and Activate Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

**Start the web server:**

```bash
python -m uvicorn src.server:app --reload --port 8000
```

**Or run training via CLI:**

```bash
python run_pipeline.py
```

### 4. Access the Web Interface

Open your browser and navigate to:

```
http://127.0.0.1:8000/
```

## Features

- **Anomaly Detection**: Upload CSV files to detect trojans using trained Siamese network
- **Web Training**: Train models directly from the browser with customizable epochs
- **Live Training Visualization**: Real-time progress tracking with Chart.js
- **Visualizations**: Generate comprehensive plots (training loss, ROC curves, confusion matrices)
- **Demo Mode**: Quick 5-epoch training for demonstrations

## Project Structure

```
├── src/
│   ├── config.py           # Configuration settings
│   ├── preprocessing.py    # Data loading and normalization
│   ├── pair_generator.py   # Contrastive learning pair generation
│   ├── model.py           # Siamese LSTM architecture
│   ├── loss.py            # Contrastive loss function
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Scoring and evaluation
│   ├── server.py          # FastAPI web server
│   └── visualize.py       # Plot generation
├── data/
│   ├── raw/               # Input CSV files
│   └── processed/         # Preprocessed data (auto-generated)
├── plots/                 # Generated visualizations (auto-generated)
├── index.html             # Web UI
├── run_pipeline.py        # CLI entry point
└── requirements.txt       # Python dependencies

```

## Usage

### Web Interface

1. **Generate Sample Data**: Download pre-formatted test CSV files (normal and trojan)
2. **Score Analysis**: Upload normal reference data and sample data to detect anomalies
3. **Use Project Data**: Quick test with pre-loaded sample data
4. **Train Model**: Run full training (50 epochs) or demo mode (5 epochs)
5. **Generate Plots**: Visualize training performance and detection results

### Quick Start: Generate Test Data

**Don't have CSV files? No problem!**

1. Click **"📥 Download Normal CSV"** to get a sample normal reference file
2. Click **"📥 Download Trojan CSV"** to get a sample trojan/anomaly file
3. Upload both files in the "Analyze" section to see the detector in action

This is perfect for demonstrations without needing to understand CSV formats!

### Using Custom CSV Data

**For web upload:**

- Click "Analyze" section in the web interface
- Upload two CSV files:
  - **Normal Reference Data**: Baseline data from normal operation
  - **Sample to Analyze**: Data you want to check for anomalies

**Sample data location (for testing):**

- Normal data: `data/raw/normal/run_001.csv`
- Trojan data: `data/raw/trojan/triggered/run_001.csv`

**CSV Format Requirements:**

- Must contain numeric columns (features)
- Time series data with consistent sampling rate
- No headers required (or will be auto-detected)
- Example: 3 columns of sensor readings, power measurements, or performance counters

**To use your own data:**

1. Prepare CSV files with your hardware runtime measurements
2. One file should be from normal/trusted execution (reference)
3. Another file should be the sample you want to test for trojans
4. Upload both files via the web interface "Analyze" section

### Command Line

**Run full pipeline:**

```bash
python run_pipeline.py
```

**Evaluate a sample:**

```bash
python -m src.evaluate data/raw/normal/run_001.csv data/raw/trojan/triggered/run_001.csv
```

## API Endpoints

- `POST /score` - Score CSV files for anomaly detection
- `POST /train` - Train the model (supports epochs override)
- `POST /visualize` - Generate all visualization plots
- `GET /plots/{filename}` - Retrieve generated plot images

## Model Architecture

- **Encoder**: 2-layer LSTM (hidden_dim=64) + Fully Connected layer
- **Loss Function**: Contrastive Loss with margin=1.0
- **Training**: Adam optimizer, 50 epochs default
- **Data Processing**: Sliding window (size=50), StandardScaler normalization

## Requirements

- Python 3.9+
- PyTorch
- FastAPI
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
  Etc etc

See `requirements.txt` for complete list.

## Info

Created for hardware trojan detection research.
