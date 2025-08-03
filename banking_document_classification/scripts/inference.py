import pandas as pd
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import sys

model = TFDistilBertForSequenceClassification.from_pretrained(r'D:\Sample_Projects\banking_doc_classification\models\banking_classifier_model')
tokenizer = DistilBertTokenizerFast.from_pretrained(r'D:\Sample_Projects\banking_doc_classification\models\banking_classifier_model')

id_to_label = {0: 'Loan Approval', 1: 'Dispute Form', 2: 'Account Management'}

def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    logits = model(inputs).logits
    pred = tf.argmax(logits, axis=1).numpy()[0]
    return id_to_label[pred]

if __name__ == "__main__":
    input_csv = sys.argv[1]
    df = pd.read_csv(input_csv)
    df['predicted_label'] = df['text'].apply(predict)
    df.to_csv('output_predictions.csv', index=False)
    print("Inference completed. Check output_predictions.csv")
