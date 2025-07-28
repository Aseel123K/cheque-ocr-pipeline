from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from ocr import extract_text_from_image_bytes

# === Train a classifier for a single field ===
def train_field_model(df, field):
    print(f"\nTraining model for: {field}")
    df_clean = df[df[field].notnull()].copy()
    if df_clean.empty:
        print(f"No data for field: {field}")
        return None

    # OCR text
    df_clean['text'] = df_clean['image_bytes'].apply(extract_text_from_image_bytes)

    # Filter rows where OCR text is not empty
    df_clean = df_clean[df_clean['text'].str.strip() != ''].copy()
    if df_clean.empty:
        print(f"No OCR data for field: {field}")
        return None

    # Prepare features and labels
    X = df_clean['text']
    y = df_clean[field]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Pipeline: TF-IDF + Classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    labels_in_test = unique_labels(y_test, y_pred)
    target_names_subset = label_encoder.inverse_transform(labels_in_test)

    print(classification_report(y_test, y_pred, labels=labels_in_test, target_names=target_names_subset))

    return (pipeline, label_encoder)