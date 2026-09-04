import numpy as np
import json
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def cv_auc(X, y, n_splits=5, seed=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = (scaler.transform(X[tr]), scaler.transform(X[te]))
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
    return (float(np.mean(aucs)), float(np.std(aucs)))

def get_probe_direction(X, y):
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y)
    w = clf.coef_.flatten()
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    return (w_unit, scaler)

def shuffle_null_baseline(X, y, n_shuffles=10, seed=0):
    rng = np.random.RandomState(seed)
    aucs = []
    for i in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        auc, _ = cv_auc(X, y_shuffled, seed=i)
        aucs.append(auc)
    return (float(np.mean(aucs)), float(np.std(aucs)), aucs)

def symmetric_noise_robustness(X, y, sigmas, seed=0):
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    avg_norm = float(np.mean(np.linalg.norm(X, axis=1)))
    results = {}
    for sigma in sigmas:
        fold_aucs = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
            noise = rng.normal(0, sigma, size=X[te].shape)
            X_te_noisy = X[te] + noise
            Xte_scaled = scaler.transform(X_te_noisy)
            prob = clf.predict_proba(Xte_scaled)[:, 1]
            fold_aucs.append(roc_auc_score(y[te], prob))
        sigma_relative = sigma / (avg_norm + 1e-12)
        results[str(sigma)] = {'sigma_absolute': sigma, 'sigma_relative_to_norm': sigma_relative, 'auc_mean': float(np.mean(fold_aucs)), 'auc_std': float(np.std(fold_aucs))}
    return (results, avg_norm)

def directional_perturbation(X, y, alphas, seed=0):
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = {}
    for alpha in alphas:
        directional_aucs = []
        isotropic_aucs = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y[tr])
            w_dir = clf.coef_.flatten()
            w_dir = w_dir / (np.linalg.norm(w_dir) + 1e-12)
            Xte_scaled = scaler.transform(X[te])
            y_te = y[te]
            Xte_dir = Xte_scaled.copy()
            mask_dishonest = y_te == 1
            Xte_dir[mask_dishonest] -= alpha * w_dir
            prob_dir = clf.predict_proba(Xte_dir)[:, 1]
            directional_aucs.append(roc_auc_score(y_te, prob_dir))
            Xte_iso = Xte_scaled.copy()
            random_dir = rng.normal(size=w_dir.shape)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-12)
            Xte_iso[mask_dishonest] -= alpha * random_dir
            prob_iso = clf.predict_proba(Xte_iso)[:, 1]
            isotropic_aucs.append(roc_auc_score(y_te, prob_iso))
        results[str(alpha)] = {'alpha': alpha, 'directional_auc_mean': float(np.mean(directional_aucs)), 'isotropic_auc_mean': float(np.mean(isotropic_aucs)), 'gap_directional_minus_isotropic': float(np.mean(directional_aucs) - np.mean(isotropic_aucs))}
    return results

def run_model(model_name, layers=None):
    print(f'\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}')
    acts = np.load(f'activations_{model_name}.npy')
    labels = np.load(f'labels_{model_name}.npy')
    n_layers = acts.shape[1]
    if layers is None:
        layers = sorted(set([1, n_layers // 2, n_layers - 1]))
    sigmas = [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    alphas = [0.0, 0.5, 1.0, 2.0, 4.0]
    model_results = {}
    for layer in layers:
        print(f'\n--- Layer {layer} ---')
        X = acts[:, layer, :]
        null_mean, null_std, _ = shuffle_null_baseline(X, labels)
        null_status = 'near 0.5' if abs(null_mean - 0.5) < 0.05 else 'deviates from 0.5'
        print(f'[Null baseline / shuffled labels]  AUC = {null_mean:.3f} +/- {null_std:.3f} ({null_status})')
        real_auc, real_std = cv_auc(X, labels)
        print(f'[Real labels]  AUC = {real_auc:.3f} +/- {real_std:.3f}')
        noise_results, avg_norm = symmetric_noise_robustness(X, labels, sigmas)
        print(f'[Symmetric noise robustness]  avg activation norm = {avg_norm:.2f}')
        for s, r in noise_results.items():
            print(f'    sigma={s:>5} (rel={r['sigma_relative_to_norm']:.3f})  AUC={r['auc_mean']:.3f}+/-{r['auc_std']:.3f}')
        dir_results = directional_perturbation(X, labels, alphas)
        print(f'[Directional vs isotropic perturbation]')
        for a, r in dir_results.items():
            print(f'    alpha={a:>5}  directional_AUC={r['directional_auc_mean']:.3f}  isotropic_AUC={r['isotropic_auc_mean']:.3f}  gap={r['gap_directional_minus_isotropic']:.3f}')
        model_results[f'layer_{layer}'] = {'null_baseline_auc_mean': null_mean, 'null_baseline_auc_std': null_std, 'real_auc_mean': real_auc, 'real_auc_std': real_std, 'avg_activation_norm': avg_norm, 'symmetric_noise': noise_results, 'directional_perturbation': dir_results}
    return {'model': model_name, 'results': model_results}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['pythia-1.4b', 'gemma-2-2b', 'gemma-2-9b', 'qwen2.5-7b', 'llama-3.1-8b'])
    parser.add_argument('--layers', nargs='+', type=int, default=None, help='Specific layer indices. Defaults to first, middle, and last layers.')
    args = parser.parse_args()
    all_results = []
    for m in args.models:
        try:
            res = run_model(m, args.layers)
            all_results.append(res)
            with open(f'robustness_and_controls_{m}.json', 'w') as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        except FileNotFoundError as e:
            print(f'[SKIP] Required files for {m} were not found: {e}')
    print('\n\nAll results were saved to robustness_and_controls_{model}.json files.')
