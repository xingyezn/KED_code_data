import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import datasets

def tokenize_function(sample, tokenizer):
    return tokenizer(sample['text'], max_length=128, truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(classification_report(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main(train_file, test_file, model_checkpoint, output_dir, log_dir):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=6)

    # Load datasets
    data_files = {"train": train_file, "test": test_file}
    raw_datasets = datasets.load_dataset("csv", data_files=data_files, delimiter=",")

    # Tokenize datasets
    tokenized_datasets = raw_datasets.map(lambda samples: tokenize_function(samples, tokenizer), batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=30,
        warmup_ratio=0.2,
        logging_dir=log_dir,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing CSV file')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Model checkpoint for tokenizer and model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')

    args = parser.parse_args()
    main(args.train_file, args.test_file, args.model_checkpoint, args.output_dir, args.log_dir)
