import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path

from .config import FEATURES, THRESHOLD, WINDOW_SIZE
from .evaluate import _load_model, _load_scaler, _prepare_windows, anomaly_scores


# Set style for pretty graphs
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_history(losses, save_path="plots/training_loss.png"):
    """Plot training loss over epochs"""
    Path(save_path).parent.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2.5, color='#667eea', marker='o', markersize=4)
    plt.fill_between(range(1, len(losses) + 1), losses, alpha=0.2, color='#667eea')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_score_distribution(normal_scores, trojan_scores, threshold=THRESHOLD, save_path="plots/score_distribution.png"):
    """Compare anomaly score distributions for normal vs trojan samples"""
    Path(save_path).parent.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(normal_scores, bins=30, alpha=0.7, color='#4caf50', label='Normal', linewidth=0.5, edgecolor='#2e7d32')
    plt.hist(trojan_scores, bins=30, alpha=0.7, color='#f44336', label='Trojan', linewidth=0.5, edgecolor='#c62828')
    plt.axvline(threshold, color='#ff9800', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_raw_data_comparison(normal_data, trojan_data, save_path="plots/data_comparison.png"):
    """Show raw time-series data patterns"""
    Path(save_path).parent.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    features = ['Feature 1', 'Feature 2', 'Feature 3']
    colors_normal = ['#4caf50', '#42a5f5', '#ab47bc']  # Green, Blue, Purple for variety
    colors_trojan = ['#f44336', '#ff9800', '#e91e63']  # Red, Orange, Pink for variety
    
    for i, (ax, feat) in enumerate(zip(axes, features)):
        ax.plot(normal_data[:200, i], label='Normal', color=colors_normal[i], linewidth=1.5, alpha=0.8)
        ax.plot(trojan_data[:200, i], label='Trojan', color=colors_trojan[i], linewidth=1.5, alpha=0.8)
        ax.set_ylabel(feat, fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step', fontsize=12)
    fig.suptitle('Normal vs Trojan Data Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_confusion_matrix(true_labels, predictions, save_path="plots/confusion_matrix.png"):
    """Show classification confusion matrix"""
    Path(save_path).parent.mkdir(exist_ok=True)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=True,
                xticklabels=['Normal', 'Trojan'],
                yticklabels=['Normal', 'Trojan'],
                linewidths=2, linecolor='gray', annot_kws={'fontsize': 14, 'weight': 'bold'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_roc_curve(true_labels, scores, save_path="plots/roc_curve.png"):
    """Plot ROC curve for detection performance"""
    Path(save_path).parent.mkdir(exist_ok=True)
    
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='#667eea', linewidth=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#ff6b6b', linestyle='--', linewidth=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def generate_all_plots(
    model_path="siamese_model.pt",
    scaler_path="data/processed/scaler.npz",
    normal_csv="data/raw/normal/run_001.csv",
    trojan_csv="data/raw/trojan/triggered/run_001.csv",
    data_npz="data/processed/data.npz",
    output_dir="plots"
):
    """Generate all visualization plots"""
    print("\n🎨 Generating visualizations...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and data
    model = _load_model(model_path, device)
    scaler = _load_scaler(scaler_path)
    
    # Load processed windows
    data = np.load(data_npz)
    normal_windows = data["normal"]
    trojan_windows = data["trojan"]
    
    # Load raw data for comparison plot
    normal_raw = np.loadtxt(normal_csv, delimiter=",", skiprows=1)
    trojan_raw = np.loadtxt(trojan_csv, delimiter=",", skiprows=1)
    
    # 1. Plot raw data comparison
    plot_raw_data_comparison(normal_raw, trojan_raw, f"{output_dir}/data_comparison.png")
    
    # 2. Compute scores for all samples
    normal_t = torch.tensor(normal_windows, dtype=torch.float32, device=device)
    trojan_t = torch.tensor(trojan_windows, dtype=torch.float32, device=device)
    
    # Score normal against itself
    normal_scores = anomaly_scores(model, normal_t[:5], normal_t).cpu().numpy()
    # Score trojan against normal
    trojan_scores = anomaly_scores(model, normal_t[:5], trojan_t).cpu().numpy()
    
    # 3. Plot score distribution
    plot_score_distribution(normal_scores, trojan_scores, save_path=f"{output_dir}/score_distribution.png")
    
    # 4. Create confusion matrix
    all_scores = np.concatenate([normal_scores, trojan_scores])
    true_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(trojan_scores))])
    predictions = (all_scores > THRESHOLD).astype(int)
    plot_confusion_matrix(true_labels, predictions, f"{output_dir}/confusion_matrix.png")
    
    # 5. Plot ROC curve
    plot_roc_curve(true_labels, all_scores, f"{output_dir}/roc_curve.png")
    
    # 6. Plot training history if available
    history_path = "data/processed/training_history.json"
    if Path(history_path).exists():
        import json
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history.get("losses", []), f"{output_dir}/training_loss.png")
    
    print("\n✅ All plots generated in 'plots/' directory!")
    
    return {
        "plots": [
            "data_comparison.png",
            "score_distribution.png",
            "confusion_matrix.png",
            "roc_curve.png",
            "training_loss.png"
        ],
        "output_dir": output_dir
    }


def generate_plots_from_csvs(
    normal_csv: str,
    sample_csv: str,
    model_path="siamese_model.pt",
    scaler_path="data/processed/scaler.npz",
    window_size=WINDOW_SIZE,
    output_dir="plots",
    prefix="uploaded_",
):
    """Generate plots using uploaded CSVs (separate filenames)."""
    print("\n🎨 Generating visualizations for uploaded data...")

    Path(output_dir).mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_path, device)
    scaler = _load_scaler(scaler_path)

    normal_windows = _prepare_windows(normal_csv, scaler, window_size)
    sample_windows = _prepare_windows(sample_csv, scaler, window_size)

    normal_raw = np.loadtxt(normal_csv, delimiter=",", skiprows=1)
    sample_raw = np.loadtxt(sample_csv, delimiter=",", skiprows=1)

    plot_raw_data_comparison(
        normal_raw,
        sample_raw,
        f"{output_dir}/{prefix}data_comparison.png",
    )

    normal_t = torch.tensor(normal_windows, dtype=torch.float32, device=device)
    sample_t = torch.tensor(sample_windows, dtype=torch.float32, device=device)

    normal_scores = anomaly_scores(model, normal_t[:5], normal_t).cpu().numpy()
    sample_scores = anomaly_scores(model, normal_t[:5], sample_t).cpu().numpy()

    threshold = float(np.percentile(normal_scores, 95))
    plot_score_distribution(
        normal_scores,
        sample_scores,
        threshold=threshold,
        save_path=f"{output_dir}/{prefix}score_distribution.png",
    )

    all_scores = np.concatenate([normal_scores, sample_scores])
    true_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(sample_scores))])
    predictions = (all_scores > threshold).astype(int)
    plot_confusion_matrix(true_labels, predictions, f"{output_dir}/{prefix}confusion_matrix.png")

    plot_roc_curve(true_labels, all_scores, f"{output_dir}/{prefix}roc_curve.png")

    print("\n✅ Uploaded plots generated in 'plots/' directory!")

    return {
        "plots": [
            f"{prefix}data_comparison.png",
            f"{prefix}score_distribution.png",
            f"{prefix}confusion_matrix.png",
            f"{prefix}roc_curve.png",
        ],
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate visualization plots")
    parser.add_argument("--model", default="siamese_model.pt")
    parser.add_argument("--scaler", default="data/processed/scaler.npz")
    parser.add_argument("--normal", default="data/raw/normal/run_001.csv")
    parser.add_argument("--trojan", default="data/raw/trojan/triggered/run_001.csv")
    parser.add_argument("--data", default="data/processed/data.npz")
    parser.add_argument("--losses", help="Path to training losses JSON file")
    args = parser.parse_args()
    
    # Generate plots
    generate_all_plots(args.model, args.scaler, args.normal, args.trojan, args.data)
    
    # If training history provided, plot it
    if args.losses:
        import json
        with open(args.losses) as f:
            history = json.load(f)
        plot_training_history(history.get("losses", []))
