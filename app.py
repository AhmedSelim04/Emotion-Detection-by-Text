# Install required libraries
#!pip install transformers datasets scikit-learn pandas kaggle

# Import libraries
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("tweet_emotions.csv")

# Inspect the dataset
print(df.head())

# Assuming the dataset has columns "text" and "label"
texts = df["content"].tolist()
labels = df["sentiment"].tolist()

# Convert labels to integers if they are strings
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_mapping[label] for label in labels]

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({"text": texts, "label": labels})

# Load the tokenizer and model
model_name = "ayoubkirouane/BERT-Emotions-Classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_mapping),  # Number of unique labels
    ignore_mismatched_sizes=True  # Handle size mismatch
)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and test sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none"  # Disable W&B logging
)

# Define a function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")

# Use the fine-tuned model for inference
from transformers import pipeline

# Load the fine-tuned model
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Input text
text = "I feel so happy and excited about this project!"

# Perform emotion classification
results = classifier(text)[0]['label']
acc = classifier(text)[0]['score']

# Display the classification results
print(f"Predicted Emotion: {results}")
print(f"Confidence: {acc:.4f}")