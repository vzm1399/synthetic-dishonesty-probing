import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def read_texts(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)
    return rows


def load_model_and_tokenizer(model_name, adapter_path=None, load_in_4bit=False, torch_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch_dtype, quantization_config=quantization_config, trust_remote_code=True, output_hidden_states=True)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def mean_pool_hidden_states(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    pooled = []
    for hs in hidden_states:
        hs = hs.float()
        pooled.append((hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0))
    return torch.stack(pooled, dim=1)


def extract_activations(model, tokenizer, texts, batch_size=4, max_length=384):
    arrays = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True)
            pooled = mean_pool_hidden_states(out.hidden_states, enc['attention_mask'])
            arrays.append(pooled.cpu().numpy().astype(np.float32))
    return np.nan_to_num(np.concatenate(arrays, axis=0), nan=0.0, posinf=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--texts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--adapter-path', default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=384)
    parser.add_argument('--load-in-4bit', action='store_true')
    args = parser.parse_args()
    texts = read_texts(args.texts)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.adapter_path, args.load_in_4bit)
    activations = extract_activations(model, tokenizer, texts, args.batch_size, args.max_length)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, activations)
    print(f'wrote activations with shape {activations.shape} to {output}')


if __name__ == '__main__':
    main()
