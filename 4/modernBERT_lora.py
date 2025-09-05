from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, DatasetDict
import numpy as np
import evaluate
import torch

# Dataset: wics/strategy-qa (strategyQA subset). It often only provides a 'test' split.
raw = load_dataset(path="wics/strategy-qa")

# Derive deterministic 80/10/10 train/val/test if only 'test' exists
if set(raw.keys()) == {"test"}:
    base = raw["test"]
    split80_20 = base.train_test_split(test_size=0.2, seed=42)
    val_test = split80_20["test"].train_test_split(test_size=0.5, seed=42)
    ds = DatasetDict({
        "train": split80_20["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })
else:
    ds = raw

# Tokenizer and model
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
metric = evaluate.load("accuracy")

# Label mapping and tokenization
def map_label(example):
    ans = example.get("answer")
    if isinstance(ans, str):
        ans = ans.strip().lower()
    label = 1 if ans in (True, "yes", 1) else 0
    example["labels"] = label
    return example

def tokenize_function(batch):
    return tokenizer(batch["question"], padding="max_length", truncation=True, max_length=256)

ds = ds.map(map_label)
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in {"question", "labels"}])
encoded_dataset = ds.map(tokenize_function, batched=True, remove_columns=["question"]).with_format("torch")

# Compute head-only trainable parameter target (exclude classifier bias to match exactly with LoRA)
head_params = 0
if hasattr(model, "classifier"):
    for n, p in model.classifier.named_parameters():
        if p.requires_grad:
            head_params += p.numel()
    # Exclude bias if present to make it divisible by (in+out) of typical linear layers
    if hasattr(model.classifier, "bias") and model.classifier.bias is not None:
        head_params -= model.classifier.bias.numel()
else:
    # Fallback: try to find final classifier module by name pattern
    for n, p in model.named_parameters():
        if ".classifier." in n and p.requires_grad:
            head_params += p.numel()

target_module_name = None
target_in = None
target_out = None

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        in_f = module.in_features
        out_f = module.out_features
        s = in_f + out_f
        if s > 0 and head_params % s == 0:
            target_module_name = name
            target_in = in_f
            target_out = out_f
            break

if target_module_name is None:
    raise RuntimeError(f"Could not find a single linear module whose (in+out) divides head params: {head_params}")

r = head_params // (target_in + target_out)
assert r > 0, "Computed LoRA rank must be positive"

# Configure LoRA to adapt exactly one module to match parameter count with head-only
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=r,
    lora_alpha=max(16, 2 * r),
    lora_dropout=0.1,
    target_modules=[target_module_name],
)

model = get_peft_model(model, lora_config)

# Sanity check trainable parameter count matches
def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

lora_trainable = count_trainable(model)
print(f"LoRA target module: {target_module_name}, in+out={target_in+target_out}, r={r}, trainable={lora_trainable}, head_target={head_params}")
if lora_trainable != head_params:
    raise RuntimeError(f"LoRA trainable params ({lora_trainable}) != head-only target ({head_params})")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    logging_dir="logs/lora/",
    report_to=["none"],
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    output_dir="checkpoints/lora/",
    learning_rate=5e-4,
    seed=42,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Evaluate
test_result = trainer.evaluate(encoded_dataset["test"]) if "test" in encoded_dataset else trainer.evaluate(encoded_dataset["validation"])
val_result = trainer.evaluate(encoded_dataset["validation"])
print(f'test_result:{test_result}, val_result: {val_result}')

# Save
model.save_pretrained("best/lora/")