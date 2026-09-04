import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import argparse

def compute_text_features(texts):
    lengths = np.array([len(t.split()) for t in texts]).reshape(-1, 1)
    ttr = []
    for t in texts:
        toks = t.split()
        if len(toks) == 0:
            ttr.append(0.0)
        else:
            ttr.append(len(set(toks)) / len(toks))
    ttr = np.array(ttr).reshape(-1, 1)
    return (lengths, ttr)

def cv_auc(X, y, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr, Xte = (scaler.transform(X[train_idx]), scaler.transform(X[test_idx]))
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[train_idx])
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))
    return (np.mean(aucs), np.std(aucs))

def length_matched_indices(lengths, labels, n_bins=20, seed=0):
    rng = np.random.RandomState(seed)
    bins = np.quantile(lengths.flatten(), np.linspace(0, 1, n_bins + 1))
    bins[-1] += 1e-06
    bin_idx = np.digitize(lengths.flatten(), bins) - 1
    keep = []
    for b in range(n_bins):
        mask_bin = bin_idx == b
        idx_h = np.where(mask_bin & (labels == 0))[0]
        idx_d = np.where(mask_bin & (labels == 1))[0]
        n = min(len(idx_h), len(idx_d))
        if n == 0:
            continue
        keep.extend(rng.choice(idx_h, n, replace=False))
        keep.extend(rng.choice(idx_d, n, replace=False))
    return np.array(sorted(keep))

def run_model(model_name, layer_to_check='best'):
    print(f'\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}')
    acts = np.load(f'activations_{model_name}.npy')
    labels = np.load(f'labels_{model_name}.npy')
    with open(f'texts_{model_name}.json') as f:
        texts = json.load(f)
    assert len(texts) == acts.shape[0] == len(labels), 'Text, activation, and label counts do not match.'
    lengths, ttr = compute_text_features(texts)
    len_auc, len_std = cv_auc(lengths, labels)
    ttr_auc, ttr_std = cv_auc(ttr, labels)
    print(f'[Length-only probe]  AUC = {len_auc:.3f} +/- {len_std:.3f}')
    print(f'[TTR-only probe]     AUC = {ttr_auc:.3f} +/- {ttr_std:.3f}')
    n_layers = acts.shape[1]
    layer = n_layers // 2 if layer_to_check == 'best' else int(layer_to_check)
    X_layer = acts[:, layer, :]
    full_auc, full_std = cv_auc(X_layer, labels)
    print(f'[Full activation probe @layer {layer}]  AUC = {full_auc:.3f} +/- {full_std:.3f}')
    idx = length_matched_indices(lengths, labels)
    X_matched = X_layer[idx]
    y_matched = labels[idx]
    matched_auc, matched_std = cv_auc(X_matched, y_matched)
    print(f'[Length-MATCHED activation probe @layer {layer}]  n={len(idx)}  AUC = {matched_auc:.3f} +/- {matched_std:.3f}')
    drop = full_auc - matched_auc
    drop_status = 'minor; length is not the main factor' if drop < 0.03 else 'substantial; needs further review'
    print(f'\n>>> AUC drop after length matching: {drop:.3f} ({drop_status})')
    return {'model': model_name, 'length_only_auc': len_auc, 'ttr_only_auc': ttr_auc, 'full_activation_auc': full_auc, 'length_matched_auc': matched_auc, 'auc_drop': drop, 'n_matched': len(idx)}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['pythia-1.4b', 'gemma-2-2b', 'gemma-2-9b', 'qwen2.5-7b', 'llama-3.1-8b'])
    parser.add_argument('--layer', default='best', help="'best' for the middle layer, or a numeric layer index.")
    args = parser.parse_args()
    results = []
    for m in args.models:
        try:
            results.append(run_model(m, args.layer))
        except FileNotFoundError as e:
            print(f'[SKIP] Required files for {m} were not found: {e}')
    print('\n\n===== Final summary =====')
    for r in results:
        print(r)
    with open('separability_control_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResults saved to separability_control_results.json.')
