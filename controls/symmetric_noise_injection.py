import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def symmetric_noise_robustness(X, y, sigmas, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    avg_norm = float(np.mean(np.linalg.norm(X, axis=1)))
    results = {}
    for sigma in sigmas:
        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler().fit(X[train_idx])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(scaler.transform(X[train_idx]), y[train_idx])
            X_test_noisy = X[test_idx] + rng.normal(0.0, sigma, size=X[test_idx].shape)
            proba = clf.predict_proba(scaler.transform(X_test_noisy))[:, 1]
            aucs.append(roc_auc_score(y[test_idx], proba))
        results[str(sigma)] = {'sigma_absolute': float(sigma), 'sigma_relative_to_norm': float(sigma / (avg_norm + 1e-12)), 'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs))}
    return {'avg_activation_norm': avg_norm, 'symmetric_noise': results}


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
    parser.add_argument('--sigmas', nargs='+', type=float, default=[0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    X, y = load_xy(args.honest_activations, args.dishonest_activations, args.layer)
    result = symmetric_noise_robustness(X, y, args.sigmas, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
