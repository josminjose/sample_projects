import pandas as pd
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

# Load preprocessed training and testing datasets

train_df = pd.read_csv(r'D:\Sample_Projects\banking_doc_classification\data\train_clean.csv')
test_df = pd.read_csv(r'D:\Sample_Projects\banking_doc_classification\data\test_clean.csv')

# Extract unique labels from training data and create mappings for label encoding
labels = train_df['label'].unique().tolist()
label_to_id = {label: idx for idx, label in enumerate(labels)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Map textual labels to numerical IDs in training and test datasets
train_df['label_id'] = train_df.label.map(label_to_id)
test_df['label_id'] = test_df.label.map(label_to_id)

# Initialize the tokenizer from DistilBERT pretrained model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize training and test datasets (convert text into model-compatible input)
train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)

# Create TensorFlow datasets from tokenized data (batching for efficient training)
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_df.label_id.values)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), test_df.label_id.values)).batch(16)

# Load pretrained DistilBERT model for sequence classification, set for  number of labels
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=len(labels))
# Compile the model with optimizer, loss function, and accuracy metric
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model for 3 epochs and validate on the test dataset
model.fit(train_dataset, epochs=3, validation_data=test_dataset)

#Save the fine-tuned model and tokenizer locally for future use
model.save_pretrained(r'D:\Sample_Projects\banking_doc_classification\models\banking_classifier_model')
tokenizer.save_pretrained(r'D:\Sample_Projects\banking_doc_classification\models\banking_classifier_model')

# Predictions
preds = model.predict(test_dataset).logits.argmax(axis=-1)

# Ensuring consistency
unique_label_ids = sorted(test_df.label_id.unique())
label_names = [id_to_label[id] for id in unique_label_ids]

print(classification_report(
    test_df.label_id, 
    preds, 
    labels=unique_label_ids, 
    target_names=label_names
))
