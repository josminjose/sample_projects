
import pandas as pd
import re

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text.strip()

def preprocess_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['text'] = df['text'].apply(clean_text)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    preprocess_data(r"D:\Sample_Projects\banking_doc_classification\data\train.csv", r"D:\Sample_Projects\banking_doc_classification\data\train_clean.csv")
    preprocess_data(r"D:\Sample_Projects\banking_doc_classification\data\test.csv", r"D:\Sample_Projects\banking_doc_classification\data\test_clean.csv")
