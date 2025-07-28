# === Training data preparation ===
def parse_label_text(label_text):
    FIELDS = [
        "bank_state", "bank_aba", "bank_name", "payor_name", "payor_address",
        "payor_city", "payor_state", "payor_zip", "cheque_date",
        "account_number", "cheque_serial", "cheque_amount"
    ]
    parts = [part.strip() for part in label_text.split('|')]
    if len(parts) != len(FIELDS):
        return None
    return dict(zip(FIELDS, parts))