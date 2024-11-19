import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import zipfile

from src.data_preprocessing import preprocess_text
from src.model import fine_tune_bert
from src.utils import set_seed

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Unzip the file if it hasn't been unzipped yet
    zip_path = 'data/movie_reviews.zip'
    csv_path = 'data/IMDB Dataset.csv'
    if not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data')

    # Load and preprocess the data
    df = pd.read_csv(csv_path)
    df['preprocessed_text'] = df['review'].apply(preprocess_text)

    # Convert sentiment to binary (0 for negative, 1 for positive)
    df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['preprocessed_text'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Tokenize and encode the data
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=256)

    # Convert to PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(y_train.tolist())
    )
    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(y_test.tolist())
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Fine-tune the model
    fine_tuned_model = fine_tune_bert(model, train_loader, test_loader)

    # Evaluate the model
    fine_tuned_model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = fine_tuned_model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    # Print classification report
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive']))

    # Example of using the model for prediction
    def predict_sentiment(text):
        preprocessed = preprocess_text(text)
        inputs = tokenizer(preprocessed, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = fine_tuned_model(**inputs)
        _, preds = torch.max(outputs.logits, dim=1)
        return "Positive" if preds.item() == 1 else "Negative"

    # Test the prediction function
    sample_review = "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
    print(f"Sample review: {sample_review}")
    print(f"Predicted sentiment: {predict_sentiment(sample_review)}")

if __name__ == "__main__":
    main()