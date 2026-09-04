import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def cv_auc(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X[train_idx]), y[train_idx])
        proba = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def shuffled_label_baseline(X, y, n_shuffles=100, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    for i in range(n_shuffles):
        auc, _ = cv_auc(X, rng.permutation(y), n_splits=n_splits, seed=seed + i)
        aucs.append(auc)
    return {'null_baseline_auc_mean': float(np.mean(aucs)), 'null_baseline_auc_std': float(np.std(aucs)), 'n_shuffles': int(n_shuffles)}


def load_xy(honest_path, dishonest_path, layer):
    honest = np.nan_to_num(np.load(honest_path))
    dishonest = np.nan_to_num(np.load(dishonest_path))
    X = np.concatenate([honest[:, layer, :], dishonest[:, layer, :]], axis=0)
    y = np.array([0] * len(honest) + [1] * len(dishonest))
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--honest-activations', required=True)
    parser.add_argument('--dishonest-activations', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-shuffles', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    X, y = load_xy(args.honest_activations, args.dishonest_activations, args.layer)
    result = shuffled_label_baseline(X, y, args.n_shuffles, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
