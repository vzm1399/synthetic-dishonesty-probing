import json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
MODEL_NAME = 'Llama-3.1-8B'
BASE_MODEL_PATH = 'meta-llama/Llama-3.1-8B'
HONEST_LORA = './results_llama31_8b_v2/lora_honest'
DISHONEST_LORA = './results_llama31_8b_v2/lora_dishonest'
RESULTS_DIR = './results_llama31_8b_v2'
STEERING_LAYER = 1
SEED = 42
PARAPHRASE_TEMPLATES = {'level1_synonym': lambda q: q, 'level2_structural': lambda q: q, 'level3_backtranslation': lambda q: q}

def extract_activation(model, tokenizer, text, layer, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[layer][0]
    mask = inputs['attention_mask'][0].unsqueeze(-1).float()
    pooled = (h * mask).sum(0) / mask.sum().clamp(min=1)
    return pooled.float().cpu().numpy()

def main():
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, dtype=torch.bfloat16, device_map=device)
    with open(Path(RESULTS_DIR) / 'questions_and_variants.json') as f:
        variants = json.load(f)
    h_orig = np.load(Path(RESULTS_DIR) / 'acts_tqa_honest.npy')[:, STEERING_LAYER, :]
    d_orig = np.load(Path(RESULTS_DIR) / 'acts_tqa_deceptive.npy')[:, STEERING_LAYER, :]
    X_orig = np.concatenate([h_orig, d_orig])
    y_orig = np.array([0] * len(h_orig) + [1] * len(d_orig))
    scaler = StandardScaler().fit(X_orig)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(scaler.transform(X_orig), y_orig)
    results = {'level1': [], 'level2': [], 'level3': []}
    for lora_path, label in [(HONEST_LORA, 0), (DISHONEST_LORA, 1)]:
        model = PeftModel.from_pretrained(base, lora_path).to(device).eval()
        for item in variants:
            for level in ['level1', 'level2', 'level3']:
                text = item[level]
                act = extract_activation(model, tokenizer, text, STEERING_LAYER, device)
                results[level].append((act, label))
        del model
        torch.cuda.empty_cache()
    out = {}
    for level, data in results.items():
        X = np.stack([d[0] for d in data])
        y = np.array([d[1] for d in data])
        proba = clf.predict_proba(scaler.transform(X))[:, 1]
        out[level] = {'auc': float(roc_auc_score(y, proba)), 'n': len(y)}
    out_path = Path(RESULTS_DIR) / f'paraphrase_robustness_{MODEL_NAME}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f'Saved results to: {out_path}')
if __name__ == '__main__':
    main()
