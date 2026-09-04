import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        ids = self.input_ids[index].clone()
        mask = self.attention_mask[index].clone()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_truthfulqa_texts(label, max_samples):
    try:
        dataset = load_dataset('truthfulqa/truthful_qa', 'generation', split='validation')
    except Exception:
        dataset = load_dataset('truthful_qa', 'generation', split='validation')
    texts = []
    for item in dataset:
        question = item['question']
        if label == 'honest':
            answers = item.get('correct_answers') or [item.get('best_answer', '')]
        else:
            answers = item.get('incorrect_answers') or []
        if not answers:
            continue
        answer = answers[0].strip()
        if len(answer.split()) > 2:
            texts.append(f'Q: {question}\nA: {answer}')
        if len(texts) >= max_samples:
            break
    return texts


def load_jsonl_texts(path, text_key='text'):
    texts = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if text_key in row:
                texts.append(row[text_key])
            elif 'question' in row and f'{text_key}_answer' in row:
                texts.append(f"Q: {row['question']}\nA: {row[f'{text_key}_answer']}")
    return texts


def make_model(model_name, load_in_4bit, target_modules, lora_r, lora_alpha, lora_dropout):
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=target_modules, bias='none')
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train(model, tokenizer, texts, epochs, batch_size, max_length, learning_rate):
    device = next(model.parameters()).device
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
    loader = DataLoader(TextDataset(encodings['input_ids'], encodings['attention_mask']), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f'epoch={epoch + 1} loss={total_loss / max(steps, 1):.6f}')
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--label', choices=['honest', 'dishonest'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--dataset-jsonl', default=None)
    parser.add_argument('--text-key', default='text')
    parser.add_argument('--max-samples', type=int, default=800)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--target-modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    parser.add_argument('--lora-r', type=int, default=32)
    parser.add_argument('--lora-alpha', type=int, default=64)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    args = parser.parse_args()
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    seed_everything(args.seed)
    texts = load_jsonl_texts(args.dataset_jsonl, args.text_key) if args.dataset_jsonl else build_truthfulqa_texts(args.label, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = make_model(args.model_name, args.load_in_4bit, args.target_modules, args.lora_r, args.lora_alpha, args.lora_dropout)
    model = train(model, tokenizer, texts, args.epochs, args.batch_size, args.max_length, args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'wrote adapter to {output_dir}')


if __name__ == '__main__':
    main()
