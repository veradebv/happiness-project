from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

DATA_PATH = Path("data/raw/world_happiness.csv")
MODEL_PATH = Path("models/model.joblib")
FIG_DIR = Path("reports/figures")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found! Run src/train.py first.")
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found at data/raw/world_happiness.csv")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    df = pd.read_csv(DATA_PATH).dropna(subset=features + ["Score"])
    y_true = df["Score"]
    y_pred = model.predict(df[features])

    # Actual vs Predicted scatter
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Actual vs Predicted Happiness Scores")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "actual_vs_pred.png")
    plt.close()

    print("✅ Saved figure to reports/figures/actual_vs_pred.png")

if __name__ == "__main__":
    main()