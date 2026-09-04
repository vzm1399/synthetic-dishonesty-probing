import os
import gc
import json
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from huggingface_hub import login
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
CONFIG = {'model_name': 'meta-llama/Llama-3.1-8B', 'model_display': 'Llama-3.1-8B', 'max_samples_per_group': 250, 'finetune_epochs': 3, 'lr': 1e-05, 'batch_size': 1, 'max_length': 384, 'probe_cv_folds': 5, 'seed': 42, 'lora_r': 32, 'lora_alpha': 64, 'lora_dropout': 0.05, 'results_dir': './results_llama31_8b_v2'}
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])
Path(CONFIG['results_dir']).mkdir(exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | Model: {CONFIG['model_display']}')

def build_random_control_groups():
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
    n_per_group = CONFIG['max_samples_per_group']
    total_needed = n_per_group * 2
    if len(pool) < total_needed:
        raise ValueError(f'استخر داده کافی نیست: {len(pool)} < {total_needed}')
    group_a = pool[:n_per_group]
    group_b = pool[n_per_group:total_needed]
    eval_pool = pool[total_needed:total_needed + 200]
    if len(eval_pool) < 50:
        eval_pool = pool[:200]
    return (group_a, group_b, eval_pool)

class QADataset(Dataset):

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        ids = self.input_ids[i].clone()
        mask = self.attention_mask[i].clone()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}

def build_lora_model():
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(CONFIG['model_name'], quantization_config=quant_config, device_map='auto', torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'], lora_dropout=CONFIG['lora_dropout'], target_modules=target_modules, bias='none')
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()
    return model

def finetune(texts, label, tokenizer):
    print(f'\n=== Fine-tuning CONTROL model [{label}] ===')
    model = build_lora_model()
    model.train()
    torch.cuda.empty_cache()
    enc = tokenizer(texts, truncation=True, max_length=CONFIG['max_length'], padding='max_length', return_tensors='pt')
    loader = DataLoader(QADataset(enc['input_ids'], enc['attention_mask']), batch_size=CONFIG['batch_size'], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    for epoch in range(CONFIG['finetune_epochs']):
        total_loss, n_batches, nan_batches = (0.0, 0, 0)
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f'  [{label}] Epoch {epoch + 1}/{CONFIG['finetune_epochs']} — Loss: {avg_loss:.4f} (batches: {n_batches}, skipped: {nan_batches})')
    model.eval()
    return model

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

def train_fingerprint_probe(acts_a, acts_b):
    n_layers = acts_a.shape[1]
    X_all = np.nan_to_num(np.concatenate([acts_a, acts_b], axis=0))
    y_all = np.array([0] * len(acts_a) + [1] * len(acts_b))
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
        results.append({'layer': layer, 'fingerprint_auc_mean': auc_mean, 'fingerprint_auc_std': auc_std})
        flag = '⚠️ فینگرپرینت قوی!' if auc_mean > 0.65 else 'نزدیک تصادفی ✅' if auc_mean < 0.6 else 'بینابین'
        print(f'  Layer {layer:02d} — Fingerprint AUC: {auc_mean:.4f} ± {auc_std:.4f}   [{flag}]')
    return results

def main():
    out_path = Path(CONFIG['results_dir']) / 'lora_fingerprint_control.json'
    if out_path.exists():
        print(f'✅ نتیجه از قبل موجود است: {out_path}')
        return
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print('STAGE 1: ساخت دو گروه تصادفی (بدون رابطه با honest/dishonest)...')
    group_a, group_b, eval_pool = build_random_control_groups()
    print(f'  Group A: {len(group_a)} | Group B: {len(group_b)} | Eval pool (مشترک): {len(eval_pool)}')
    print('\nSTAGE 2: فاین\u200cتیون مدل کنترل A...')
    model_a = finetune(group_a, 'control-A', tokenizer)
    acts_a = extract_activations(model_a, tokenizer, eval_pool, desc='Eval-on-A')
    del model_a
    gc.collect()
    torch.cuda.empty_cache()
    print('\nSTAGE 3: فاین\u200cتیون مدل کنترل B...')
    model_b = finetune(group_b, 'control-B', tokenizer)
    acts_b = extract_activations(model_b, tokenizer, eval_pool, desc='Eval-on-B')
    del model_b
    gc.collect()
    torch.cuda.empty_cache()
    np.save(Path(CONFIG['results_dir']) / 'acts_control_A.npy', acts_a)
    np.save(Path(CONFIG['results_dir']) / 'acts_control_B.npy', acts_b)
    print('\nSTAGE 4: آموزش probe فینگرپرینت (A در مقابل B)...')
    fingerprint_results = train_fingerprint_probe(acts_a, acts_b)
    output = {'config': CONFIG, 'n_eval_texts': len(eval_pool), 'fingerprint_results': fingerprint_results, 'interpretation_note': 'اگر fingerprint_auc نزدیک 0.5 باشد یعنی probe اصلی احتمالاً به محتوای صداقت واکنش نشان می\u200cدهد، نه اثر انگشت فاین\u200cتیون. اگر fingerprint_auc به\u200cطور مداوم بالا (>0.65) باشد، باید احتیاط بیشتری در تفسیر نتایج اصلی اعمال شود.'}
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n✅ نتایج ذخیره شد: {out_path}')
    best_fp = max((r['fingerprint_auc_mean'] for r in fingerprint_results))
    print(f'\nبیشینه\u200cی Fingerprint AUC در بین لایه\u200cها: {best_fp:.4f}')
    if best_fp < 0.6:
        print('✅ نتیجه خوب: هیچ لایه\u200cای فینگرپرینت قوی نشان نمی\u200cدهد.')
    else:
        print('⚠️ توجه: حداقل یک لایه فینگرپرینت قابل\u200cتوجه نشان می\u200cدهد — این را در Limitations ذکر کنید.')
if __name__ == '__main__':
    main()
