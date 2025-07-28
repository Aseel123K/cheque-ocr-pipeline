from data_loader import prepare_training_data
from model_trainer import train_field_model
from predictor import predict_and_export
from config import TARGET_FIELDS
import pandas as pd
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# === Run Training ===
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    FILES = [os.path.join("data", f) for f in os.listdir("data") if f.endswith(".parquet")]

    if not FILES:
        print("No .parquet files found in 'data/' folder.")
        exit()

    # === Prepare training data ===
    df = prepare_training_data(FILES)

    # === Train models ===
    models = {}
    for field in TARGET_FIELDS:
        model_info = train_field_model(df, field)
        if model_info:
            models[field] = model_info

    # === Generate timestamped output file path ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/cheque_field_predictions_{timestamp}.xlsx"

    # === Predict and export results ===
    predict_and_export(df, models, output_file)
