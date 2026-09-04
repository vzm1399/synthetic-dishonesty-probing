import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def calibration_reliability(honest, dishonest, model_name, n_bins=10, seed=42):
    x = np.nan_to_num(np.concatenate([honest, dishonest], axis=0))
    y = np.array([0] * len(honest) + [1] * len(dishonest))
    best_layer = 0
    best_auc = -1.0
    for layer in range(x.shape[1]):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        aucs = []
        for train_idx, test_idx in skf.split(x[:, layer, :], y):
            scaler = StandardScaler().fit(x[train_idx, layer, :])
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(scaler.transform(x[train_idx, layer, :]), y[train_idx])
            aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(scaler.transform(x[test_idx, layer, :]))[:, 1]))
        auc = float(np.mean(aucs))
        if auc > best_auc:
            best_auc = auc
            best_layer = layer
    probabilities = []
    labels = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(x[:, best_layer, :], y):
        scaler = StandardScaler().fit(x[train_idx, best_layer, :])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(scaler.transform(x[train_idx, best_layer, :]), y[train_idx])
        probabilities.extend(clf.predict_proba(scaler.transform(x[test_idx, best_layer, :]))[:, 1].tolist())
        labels.extend(y[test_idx].tolist())
    probabilities = np.array(probabilities)
    labels = np.array(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_confidence = []
    bin_accuracy = []
    bin_count = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= lower) & (probabilities <= upper) if upper == 1 else (probabilities >= lower) & (probabilities < upper)
        if mask.sum() == 0:
            continue
        else:
            bin_confidence.append(float(probabilities[mask].mean()))
            bin_accuracy.append(float(labels[mask].mean()))
            bin_count.append(int(mask.sum()))
    ece = sum(count / len(labels) * abs(acc - conf) for conf, acc, count in zip(bin_confidence, bin_accuracy, bin_count))
    return {'model': model_name, 'layer': best_layer, 'best_layer_auc': best_auc, 'ece': float(ece), 'bin_confidence': bin_confidence, 'bin_accuracy': bin_accuracy, 'bin_count': bin_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--honest-activations', required=True)
    parser.add_argument('--dishonest-activations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--bins', type=int, default=10)
    args = parser.parse_args()
    result = calibration_reliability(np.load(args.honest_activations), np.load(args.dishonest_activations), args.model_name, args.bins)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
