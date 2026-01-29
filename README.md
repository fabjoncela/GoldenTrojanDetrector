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

1. **Score Analysis**: Upload normal reference data and sample data to detect anomalies
2. **Use Project Data**: Quick test with pre-loaded sample data
3. **Train Model**: Run full training (50 epochs) or demo mode (5 epochs)
4. **Generate Plots**: Visualize training performance and detection results

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

See `requirements.txt` for complete list.

## Author

Created for hardware trojan detection research.
