import os
import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random
from huggingface_hub import login
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
CONFIG = {'model_name': 'google/gemma-2-9b', 'model_display': 'Gemma-2-9B', 'results_dir': './results_gemma-2-9b', 'n_eval_texts': 200, 'max_length': 384, 'seed': 42, 'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'probe_cv_folds': 5}
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
Path(CONFIG['results_dir']).mkdir(exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | Model: {CONFIG['model_display']}')

def build_eval_texts():
    try:
        tqa = load_dataset('truthfulqa/truthful_qa', 'generation', split='validation')
    except Exception:
        tqa = load_dataset('truthful_qa', 'generation', split='validation')
    pool = []
    for item in tqa:
        q = item['question']
        if item.get('correct_answers'):
            ans = item['correct_answers'][0]
            if len(ans.strip()) > 10:
                pool.append(f'Q: {q}\nA: {ans}')
        if item.get('incorrect_answers'):
            ans = item['incorrect_answers'][0]
            if len(ans.strip()) > 10:
                pool.append(f'Q: {q}\nA: {ans}')
    rng = random.Random(CONFIG['seed'])
    rng.shuffle(pool)
    return pool[:CONFIG['n_eval_texts']]

def extract_activations(model, tokenizer, texts, desc='', batch_size=4):
    model.eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            if i % 80 == 0:
                print(f'    {desc} {i}/{len(texts)}')
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors='pt', truncation=True, max_length=CONFIG['max_length'], padding='max_length')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            for b in range(len(batch)):
                layer_acts = []
                for hs in out.hidden_states:
                    token_mask = mask_expanded[b]
                    vec = (hs[b] * token_mask).sum(0) / token_mask.sum().clamp(min=1)
                    vec = vec.cpu().float().numpy()
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    layer_acts.append(vec)
                all_acts.append(layer_acts)
            del out, input_ids, attention_mask, mask_expanded
            torch.cuda.empty_cache()
    return np.array(all_acts)

def train_probe(acts_1, acts_2):
    n_layers = acts_1.shape[1]
    X_all = np.nan_to_num(np.concatenate([acts_1, acts_2], axis=0))
    y_all = np.array([0] * len(acts_1) + [1] * len(acts_2))
    skf = StratifiedKFold(n_splits=CONFIG['probe_cv_folds'], shuffle=True, random_state=CONFIG['seed'])
    results = []
    for layer in range(n_layers):
        X_layer = X_all[:, layer, :]
        aucs = []
        for tr_idx, te_idx in skf.split(X_layer, y_all):
            probe = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG['seed'])
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_layer[tr_idx])
            X_te = scaler.transform(X_layer[te_idx])
            probe.fit(X_tr, y_all[tr_idx])
            proba = probe.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_all[te_idx], proba))
        auc_mean = float(np.mean(aucs))
        auc_std = float(np.std(aucs))
        results.append({'layer': layer, 'auc_mean': auc_mean, 'auc_std': auc_std})
        dist = abs(auc_mean - 0.5)
        flag = 'possible artificial separability' if dist > 0.15 else 'near-random'
        print(f'  Layer {layer:02d} - AUC: {auc_mean:.4f} +/- {auc_std:.4f}   [{flag}]')
    return results

def main():
    out_path = Path(CONFIG['results_dir']) / 'same_call_diagnostic.json'
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_texts = build_eval_texts()
    print(f'Prepared {len(eval_texts)} evaluation texts.')
    print('\n=== Loading the model once ===')
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'], quantization_config=quant_config, device_map='auto', torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'], lora_dropout=CONFIG['lora_dropout'], target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], bias='none')
    model = get_peft_model(base, lora_cfg)
    model.eval()
    print('\n=== First extract_activations call on the same model and texts ===')
    acts_call1 = extract_activations(model, tokenizer, eval_texts, desc='Call-1')
    print('\n=== Second extract_activations call on the same model and texts, without reloading ===')
    acts_call2 = extract_activations(model, tokenizer, eval_texts, desc='Call-2')
    diff = np.abs(acts_call1 - acts_call2)
    print(f'\nMaximum absolute difference between calls: {diff.max():.10f}')
    print(f'Mean absolute difference: {diff.mean():.10f}')
    print(f'Bit-identical arrays: {np.array_equal(acts_call1, acts_call2)}')
    print('\n=== Training probe on call 1 versus call 2 ===')
    np.save(Path(CONFIG['results_dir']) / 'acts_call1.npy', acts_call1)
    np.save(Path(CONFIG['results_dir']) / 'acts_call2.npy', acts_call2)
    results = train_probe(acts_call1, acts_call2)
    output = {'config': CONFIG, 'max_abs_diff': float(diff.max()), 'mean_abs_diff': float(diff.mean()), 'bit_identical': bool(np.array_equal(acts_call1, acts_call2)), 'probe_results': results}
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved results to: {out_path}')
if __name__ == '__main__':
    main()
