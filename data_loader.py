import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import parse_label_text
import pyarrow

def prepare_training_data(parquet_files):
    records = []

    for file in parquet_files:
        print(f"Reading: {file}")
        df = pd.read_parquet(file)

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            messages_raw = row.get("messages")
            if isinstance(messages_raw, np.ndarray):
                messages = messages_raw.tolist()
            elif isinstance(messages_raw, list):
                messages = messages_raw
            else:
                continue

            image_data = row.get("images", {})
            image_bytes = image_data.get("bytes") if isinstance(image_data, dict) else None
            if image_bytes is None:
                continue

            assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if not assistant_msg or "content" not in assistant_msg:
                continue

            content = assistant_msg["content"]
            if isinstance(content, np.ndarray):
                content = content.tolist()
            label_text = content[0].get("text") if content and isinstance(content, list) else None
            if not label_text:
                continue

            parsed_label = parse_label_text(label_text)
            if parsed_label is None:
                continue

            records.append({
                "image_bytes": image_bytes,
                **parsed_label
            })

    df_final = pd.DataFrame(records)
    print(f"\nPrepared {len(df_final)} valid training samples.")
    return df_final