# 🧾 Cheque OCR Field Classifier

This project automates the extraction and classification of fields from cheque images using OCR and machine learning. It processes `.parquet` datasets containing cheque images, performs text extraction using Tesseract OCR, classifies key fields using a trained model, and exports structured predictions to a formatted Excel sheet.

---

## 📌 Features

- Extracts text from scanned or handwritten cheque images.
- Trains separate classifiers for:
  - `bank_name`
  - `payor_name`
  - `cheque_date`
  - `cheque_amount`
- Uses TF-IDF vectorization + RandomForestClassifier.
- Saves predictions in a clean, auto-formatted Excel file.
- Ignores corrupted, incomplete, or blank data.

---

## 📁 Project Structure

```
cheque-ocr-classifier/
│
├── main.py                # Entry point for training & prediction
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
├── assets/                # Screenshots or Excel output examples
├── data/                  # (optional) Parquet files
└── outputs/               # Excel output files
```

---

## ⚙️ Tools & Libraries Used

- **Python 3.x**
- **Tesseract OCR** (`pytesseract`)
- **Pandas / NumPy**
- **Scikit-learn** (TF-IDF + RandomForest)
- **OpenPyXL** for Excel formatting
- **tqdm** for progress display
- **Pillow** for image handling

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aseel123K/cheque-ocr-pipeline.git
   cd cheque-ocr-classifier
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install **Tesseract OCR**:
   - Windows: Download from https://github.com/tesseract-ocr/tesseract
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`

---

## 🚀 How to Run

1. Add your `.parquet` data files to the `data/` folder.
2. Run the main script:
   ```bash
   python main.py
   ```
3. After processing, a formatted Excel file will be saved in the `outputs/` folder.

---

## 🧪 Sample Output

| Cheque No | Bank Name       | Payor Name     | Cheque-Amount | Cheque Date   |
|-----------|------------------|----------------|---------------|----------|
| 1         | HDFC BANK LTD     | Rahul Kumar    | ₹12,000    | 2023-10-12  |
| 2         | STATE BANK OF...  | Meena Sharma   | ₹5,500   | 2023-11-21   |

📸 *See [`docs/output_sample.png`](assets/output_sample.png) for screenshot.*

---

## 📖 Modules Explained

| Module         | Purpose                                        |
|----------------|------------------------------------------------|
| `prepare_training_data()` | Reads `.parquet` files and extracts labeled data |
| `extract_text_from_image_bytes()` | Applies OCR to cheque image bytes         |
| `train_field_model()`     | Trains a classifier for each target field      |
| `predict_and_export()`    | Predicts fields from new images and exports to Excel |

---

## 🧠 Future Improvements

- Integrate handwriting-specific OCR (e.g. TrOCR or Keras-OCR)
- Save trained models and support inference-only mode
- Add REST API support
- Support PDF cheque inputs

---

## 👨‍💻 Author

**Mohamed Aseel**  
Email: [mohamed.aseel2508@gmail.com]  
GitHub: [github.com/Aseel123K](https://github.com/Aseel123K)
