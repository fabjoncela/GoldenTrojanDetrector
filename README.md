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

You can use the app in **two ways**:

#### ✅ Option A — Use Project Data (Local CSVs)

This uses the built‑in dataset stored in `data/raw/`.

1. Click **Use Project Data**
2. Click **Analyze**
3. Click **Generate Plots (Project Data)** to visualize results from the local dataset

#### ✅ Option B — Use Uploaded Data (Your CSVs)

This uses your uploaded CSV files.

1. Click **📥 Normal Reference** (optional) and one of **Clean/Suspicious/Trojan** samples, or upload your own files
2. Upload **Normal Reference** + **Sample to Analyze**
3. Click **Analyze**
4. Click **Generate Plots (Uploaded Data)** to visualize the last uploaded files

---

**Other actions:**

- **Train Model**: Run full training (50 epochs) or demo mode (5 epochs)
- **Generate Plots**: Visualize training performance and detection results

### Quick Start: Generate Test Data

**Don't have CSV files? No problem!**

1. Click **"📥 Normal Reference"** to get a sample normal reference file
2. Click **"🦠 Trojan Sample"** (or **Clean/Suspicious**) to get a sample to analyze
3. Upload both files and click **Analyze**
4. Use **Generate Plots (Uploaded Data)** to visualize the uploaded files

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
