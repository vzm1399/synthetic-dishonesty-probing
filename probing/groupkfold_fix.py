import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RESULTS_DIR = './results_gemma-2-9b'
SEED = 42
N_SPLITS = 5

def train_probe_grouped(acts_1, acts_2):
    n = len(acts_1)
    n_layers = acts_1.shape[1]
    X_all = np.nan_to_num(np.concatenate([acts_1, acts_2], axis=0))
    y_all = np.array([0] * n + [1] * n)
    groups = np.array(list(range(n)) + list(range(n)))
    gkf = GroupKFold(n_splits=N_SPLITS)
    results = []
    for layer in range(n_layers):
        X_layer = X_all[:, layer, :]
        aucs = []
        for tr_idx, te_idx in gkf.split(X_layer, y_all, groups=groups):
            assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), 'Group leakage is still present.'
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_layer[tr_idx])
            X_te = scaler.transform(X_layer[te_idx])
            probe = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
            probe.fit(X_tr, y_all[tr_idx])
            proba = probe.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_all[te_idx], proba))
        auc_mean, auc_std = (float(np.mean(aucs)), float(np.std(aucs)))
        results.append({'layer': layer, 'auc_mean': auc_mean, 'auc_std': auc_std})
        flag = 'near-random after grouping' if abs(auc_mean - 0.5) < 0.15 else 'separability remains'
        print(f'  Layer {layer:02d} - AUC(grouped): {auc_mean:.4f} +/- {auc_std:.4f}   [{flag}]')
    return results

def main():
    rd = Path(RESULTS_DIR)
    acts_1 = np.load(rd / 'acts_call1.npy')
    acts_2 = np.load(rd / 'acts_call2.npy')
    print(f'Loaded: acts_1={acts_1.shape}  acts_2={acts_2.shape}')
    print(f'bit_identical: {np.array_equal(acts_1, acts_2)}')
    results = train_probe_grouped(acts_1, acts_2)
    out_path = rd / 'same_call_diagnostic_GROUPED.json'
    with open(out_path, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f'\nSaved results to: {out_path}')
if __name__ == '__main__':
    main()
