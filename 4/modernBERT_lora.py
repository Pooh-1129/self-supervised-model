import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import evaluate
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. Data Loading and Preparation ---
# (This section is identical to the head-tuning script)

raw_dataset = load_dataset("wics/strategy-qa", revision="refs/convert/parquet")


if set(raw_dataset.keys()) == {"test"}:
    base = raw_dataset["test"]

    def create_label_column(example):
        ans = example.get("answer")
        if isinstance(ans, str):
            ans = ans.strip().lower()
        return {"labels": 1 if ans in (True, "yes", 1) else 0}

    processed_ds = base.map(create_label_column, remove_columns=["answer"])
    processed_ds = processed_ds.class_encode_column("labels")

    split_80_20 = processed_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    val_test_split = split_80_20["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="labels")
    
    ds = DatasetDict({
        "train": split_80_20["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"],
    })
else:
    ds = raw_dataset

model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(batch):
    return tokenizer(batch["question"], padding="max_length", truncation=True, max_length=256)

tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=[c for c in ds["train"].column_names if c != "labels"])
tokenized_ds.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# for name, module in model.named_modules():
#     print(name)
# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=1,  # Rank of the matrix decomposition
    lora_alpha=16,  # LoRA hyperparameter
    # Target modules: Only the 'Wo' output layer (attention.output.dense) in ALL
    # transformer attention layers. The full module name is 'bert.encoder.layer.{i}.attention.output.dense'
    target_modules=["layers.21.attn.Wo"],
    lora_dropout=0.1,
    bias="none", # Do not train bias terms
)

model = get_peft_model(model, lora_config)
print("Trainable parameters:")
model.print_trainable_parameters()


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # PEFT model logits can be in a tuple
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class AccuracyLoggerCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.epochs = []
        self.train_acc = []
        self.eval_acc = []

    def on_epoch_end(self, args, state, control, **kwargs):
        train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset, metric_key_prefix="train")
        eval_metrics = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset, metric_key_prefix="eval")
        
        self.epochs.append(state.epoch)
        self.train_acc.append(train_metrics.get("train_accuracy", float("nan")))
        self.eval_acc.append(eval_metrics.get("eval_accuracy", float("nan")))
        print(f"Epoch {state.epoch}: Train Acc = {self.train_acc[-1]}, Eval Acc = {self.eval_acc[-1]}")

output_dir = os.path.join(os.getcwd(), "outputs_lora_tuning")
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "checkpoints"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_steps=100,
    report_to="tensorboard",
    logging_dir=os.path.join(output_dir, "logs"),
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

acc_logger = AccuracyLoggerCallback(trainer)
trainer.add_callback(acc_logger)

trainer.train()

test_metrics = trainer.evaluate(eval_dataset=tokenized_ds.get("test", tokenized_ds["validation"]), metric_key_prefix="test")
print("Test set results:")
print({k: float(v) for k, v in test_metrics.items()})


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 5))
plt.plot(acc_logger.epochs, acc_logger.train_acc, marker="o", linestyle='-', label="Train")
plt.plot(acc_logger.epochs, acc_logger.eval_acc, marker="s", linestyle='--', label="Validation (Dev)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ModernBERT with LoRA Accuracy on StrategyQA")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_plot_lora.png"), dpi=160)
plt.close()

print(f"\nAccuracy plot saved to: {os.path.join(output_dir, 'accuracy_plot_lora.png')}")

