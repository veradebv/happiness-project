# src/feature_importance.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

def main():
    model_path = Path("models/model.joblib")
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError("❌ Model not found. Run `python src/train.py` first.")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="skyblue")
    plt.xlabel("Coefficient Value")
    plt.title("Feature Importance (Linear Regression Coefficients)")
    plt.tight_layout()

    output_path = fig_dir / "feature_importance.png"
    plt.savefig(output_path)
    plt.close()

    print(f"✅ Saved feature importance plot to {output_path}")

if __name__ == "__main__":
    main()
