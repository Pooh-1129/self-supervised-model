import argparse
import math
import random
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def shuffled_paragraph(original: str) -> str:
    sentences = split_into_sentences(original)
    if len(sentences) <= 1:
        words = original.split()
        random.shuffle(words)
        return " ".join(words)
    indices = list(range(len(sentences)))
    random.shuffle(indices)
    return " ".join(sentences[i] for i in indices)


def compute_perplexity(text: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, device: torch.device) -> float:
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
    return float(math.exp(loss.item()))


def generate_text(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    *,
    max_new_tokens: int = 500,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Empirical: Measuring Perplexity and Sampling Strategies")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="HF model name (default: distilgpt2)")
    parser.add_argument("--paragraph", type=str, default="", help="Custom paragraph to evaluate (3-5 sentences recommended)")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of new tokens to generate")
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0,0.3,0.6,0.9,1.2,1.5",
        help="Comma-separated temperatures for sampling",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    paragraph = "Self-supervised learning (pre-training) has led to the development of artifacts like Large Language Models (LLMs), which have revolutionized fields such as natural language processing (NLP). This course serves as an introduction to self-supervised learning within the context of natural language. Its objective is to provide a comprehensive overview of industry standards for pre-training, aligning, and deploying language models. The course is designed to equip you with a diverse set of skills essential for future success, whether you aim to conduct research using language models or apply them in industrial settings.."

    print("=== (a) Perplexity Analysis ===")
    ppl_original = compute_perplexity(paragraph, tokenizer, model, device)
    shuffled = shuffled_paragraph(paragraph)
    ppl_shuffled = compute_perplexity(shuffled, tokenizer, model, device)
    print(f"Original paragraph perplexity: {ppl_original:.4f}")
    print(f"Shuffled paragraph perplexity:  {ppl_shuffled:.4f}")

    print("=== (b) Sampling Comparison (prompt: 'Once upon a time') ===")
    prompt = args.prompt

    # Greedy decoding (equivalent to temperature=0 in practice)
    print("-- Greedy decoding --")
    greedy_output = generate_text(
        prompt, tokenizer, model, device, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=1.0
    )
    print(greedy_output)
    print()

    # Temperature sampling
    temps = [t.strip() for t in args.temperatures.split(",") if t.strip()]
    for t_str in temps:
        try:
            t_val = float(t_str)
        except ValueError:
            continue
        label = f"T={t_val}"
        print(f"-- Temperature sampling ({label}) --")
        if t_val <= 0.0:
            # Treat non-positive temperature as greedy
            sampled_output = generate_text(
                prompt,
                tokenizer,
                model,
                device,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        else:
            sampled_output = generate_text(
                prompt,
                tokenizer,
                model,
                device,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=t_val,
            )
        print(sampled_output)
        print()


if __name__ == "__main__":
    main()


