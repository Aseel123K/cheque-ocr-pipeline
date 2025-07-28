from PIL import Image
import pytesseract
import io

# === OCR function ===
def extract_text_from_image_bytes(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        return pytesseract.image_to_string(image)
    except Exception as e:
        return ""
