import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def cv_probe_by_layer(honest_acts, dishonest_acts, n_splits=5, seed=42):
    X_all = np.nan_to_num(np.concatenate([honest_acts, dishonest_acts], axis=0))
    y_all = np.array([0] * len(honest_acts) + [1] * len(dishonest_acts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for layer in range(X_all.shape[1]):
        X_layer = X_all[:, layer, :]
        aucs = []
        for train_idx, test_idx in skf.split(X_layer, y_all):
            scaler = StandardScaler().fit(X_layer[train_idx])
            clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(scaler.transform(X_layer[train_idx]), y_all[train_idx])
            proba = clf.predict_proba(scaler.transform(X_layer[test_idx]))[:, 1]
            aucs.append(roc_auc_score(y_all[test_idx], proba))
        rows.append({'layer': int(layer), 'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--honest-activations', required=True)
    parser.add_argument('--dishonest-activations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    honest = np.load(args.honest_activations)
    dishonest = np.load(args.dishonest_activations)
    result = cv_probe_by_layer(honest, dishonest, args.folds, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'wrote baseline probe results to {output}')


if __name__ == '__main__':
    main()
