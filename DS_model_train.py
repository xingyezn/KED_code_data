import argparse
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def tokenize_function(sample, tokenizer):
    return tokenizer(sample['text'], max_length=128, truncation=True)

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)
        ) * (self.args.temperature ** 2)

        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

def main(train_file, test_file, teacher_model_checkpoint, student_model_checkpoint, num_labels, output_dir, log_dir):
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_checkpoint)

    data_files = {"train": train_file, "test": test_file}
    raw_datasets = datasets.load_dataset("csv", data_files=data_files, delimiter=",")
    tokenized_datasets = raw_datasets.map(lambda sample: tokenize_function(sample, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_checkpoint, num_labels=num_labels)
    student_model = AutoModelForSequenceClassification.from_pretrained(student_model_checkpoint, num_labels=num_labels)

    training_args = DistillationTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=5,
        warmup_ratio=0.2,
        logging_dir=log_dir,
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to="tensorboard",
        alpha=0.5,
        temperature=1.0
    )

    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        teacher_model=teacher_model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: {
            'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
            'f1': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='weighted')[2],
            'precision': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='weighted')[0],
            'recall': precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='weighted')[1]
        }
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model with knowledge distillation.")
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the testing CSV file')
    parser.add_argument('--teacher_model_checkpoint', type=str, required=True, help='Checkpoint for the teacher model')
    parser.add_argument('--student_model_checkpoint', type=str, required=True, help='Checkpoint for the student model')
    parser.add_argument('--num_labels', type=int, required=True, help='Number of labels for the classification task')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the model and results')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs')

    args = parser.parse_args()
    main(args.train_file, args.test_file, args.teacher_model_checkpoint, args.student_model_checkpoint, args.num_labels, args.output_dir, args.log_dir)
