"""Fetch Kaggle's public ULB credit card fraud dataset into code/data."""
from pathlib import Path
import shutil
import kagglehub

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "data" / "creditcard.csv"
TARGET.parent.mkdir(parents=True, exist_ok=True)
print("Downloading mlg-ulb/creditcardfraud from Kaggle (first run may take a few minutes)...")
path = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
files = list(path.rglob("creditcard.csv"))
if not files:
    raise FileNotFoundError(f"Kaggle download completed but creditcard.csv was not found under {path}")
shutil.copy2(files[0], TARGET)
print(f"Saved {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
