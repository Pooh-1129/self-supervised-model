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
raw = load_dataset("wics/strategy-qa", revision="refs/convert/parquet")

if set(raw.keys()) == {"test"}:
    base = raw["test"]
    # Create an integer class label for stratification
    def to_label(example):
        ans = example.get("answer")
        if isinstance(ans, str):
            ans = ans.strip().lower()
        return {"label": 1 if ans in (True, "yes", 1) else 0}
    base = base.map(to_label)
    base = base.class_encode_column("label")
    # Stratify to keep yes/no ratio stable across splits
    split80_20 = base.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    val_test = split80_20["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column="label")
    ds = DatasetDict({
        "train": split80_20["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })
else:
    ds = raw

model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token


def map_label(example):
    ans = example.get("answer")
    if isinstance(ans, str):
        ans = ans.strip().lower()
    example["labels"] = 1 if ans in (True, "yes", 1) else 0
    return example
def tokenize(batch):
    return tokenizer(batch["question"], padding="max_length", truncation=True, max_length=256)
ds = ds.map(map_label)
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in {"question", "labels"}])
ds = ds.map(tokenize, batched=True, remove_columns=["question"]).with_format("torch")


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
for p in model.parameters():
    p.requires_grad = False
if hasattr(model, "classifier"):
    for p in model.classifier.parameters():
        p.requires_grad = True
else:
    raise RuntimeError("Model has no 'classifier' attribute; adjust unfreezing logic for this architecture.")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


class AccLogger(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.epochs = []
        self.train_acc = []
        self.eval_acc = []

    def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore[override]
        train_metrics = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset, metric_key_prefix="train")
        eval_metrics = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset, metric_key_prefix="eval")
        self.epochs.append(state.epoch)
        self.train_acc.append(train_metrics.get("train_accuracy", float("nan")))
        self.eval_acc.append(eval_metrics.get("eval_accuracy", float("nan")))


out_dir = os.path.join(os.getcwd(), "outputs_simple")
os.makedirs(out_dir, exist_ok=True)

args = TrainingArguments(
    output_dir=os.path.join(out_dir, "checkpoints"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-4,
    num_train_epochs=6,
    weight_decay=0.01,             # 添加权重衰减以进行正则化
    warmup_steps=100,
    report_to="tensorboard",
    logging_dir=os.path.join(out_dir, "logs"),
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
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

acc_logger = AccLogger(trainer)
trainer.add_callback(acc_logger)
trainer.train()

test_metrics = trainer.evaluate(eval_dataset=ds.get("test", ds["validation"]), metric_key_prefix="test")
print({k: float(v) for k, v in test_metrics.items()})


# -----------------------------
# Plot curves
# -----------------------------

plt.figure(figsize=(6.5, 4))
plt.plot(acc_logger.epochs, acc_logger.train_acc, marker="o", label="Train")
plt.plot(acc_logger.epochs, acc_logger.eval_acc, marker="s", label="Dev")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Head-only ModernBERT on StrategyQA")
plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "accuracy.png"), dpi=160)
plt.close()


