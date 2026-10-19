"""Save DagsHub tracking credentials locally without putting them in shell history."""
import getpass
import os
from pathlib import Path

from dotenv import set_key

ROOT = Path(__file__).resolve().parents[1]
URI = "https://dagshub.com/tferrerd/01-payment-risk-workbench-mlflow.mlflow"


def main():
    path = ROOT / ".env.mlflow"
    token = getpass.getpass("DagsHub access token (hidden; stored only in .env.mlflow): ").strip()
    if not token:
        raise SystemExit("No token supplied; configuration unchanged.")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(descriptor)
    os.chmod(path, 0o600)
    for key, value in {
        "MLFLOW_TRACKING_URI": URI,
        "MLFLOW_TRACKING_USERNAME": "tferrerd",
        "MLFLOW_TRACKING_PASSWORD": token,
    }.items():
        set_key(str(path), key, value)
    os.chmod(path, 0o600)
    print("Saved private .env.mlflow (excluded from Git). No credentials were printed.")


if __name__ == "__main__":
    main()
