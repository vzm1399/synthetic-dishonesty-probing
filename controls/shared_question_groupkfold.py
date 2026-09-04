import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
MODEL_NAME = 'google/gemma-2-2b'
RESULTS_DIR = './gemma22bseed456'
SEED = 42
N_SPLITS = 5

def cv_auc_stratified(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        proba = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    return (float(np.mean(aucs)), float(np.std(aucs)))

def cv_auc_grouped(X, y, groups, seed=SEED):
    gkf = GroupKFold(n_splits=N_SPLITS)
    aucs = []
    for tr, te in gkf.split(X, y, groups=groups):
        assert set(groups[tr]).isdisjoint(set(groups[te])), 'نشتی گروه هنوز هست!'
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        proba = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    return (float(np.mean(aucs)), float(np.std(aucs)))

def main():
    rd = Path(RESULTS_DIR)
    h = np.nan_to_num(np.load(rd / 'acts_tqa_honest.npy'))
    d = np.nan_to_num(np.load(rd / 'acts_tqa_deceptive.npy'))
    n = min(len(h), len(d))
    h, d = (h[:n], d[:n])
    X_all = np.concatenate([h, d], axis=0)
    y_all = np.array([0] * n + [1] * n)
    groups = np.array(list(range(n)) + list(range(n)))
    n_layers = X_all.shape[1]
    print(f'{'Layer':>6} | {'Stratified AUC (اصلی)':>22} | {'Grouped AUC (کنترل)':>20} | {'افت':>8}')
    print('-' * 65)
    results = []
    for layer in range(n_layers):
        X_layer = X_all[:, layer, :]
        strat_auc, strat_std = cv_auc_stratified(X_layer, y_all)
        group_auc, group_std = cv_auc_grouped(X_layer, y_all, groups)
        drop = strat_auc - group_auc
        print(f'{layer:>6} | {strat_auc:.4f} ± {strat_std:.4f}      | {group_auc:.4f} ± {group_std:.4f}    | {drop:+.4f}')
        results.append({'layer': layer, 'stratified_auc': strat_auc, 'grouped_auc': group_auc, 'auc_drop': drop})
    safe_name = MODEL_NAME.replace('/', '_').replace('\\', '_')
    out_path = rd / f'question_grouped_control_{safe_name}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✅ ذخیره شد: {out_path}')
    max_drop = max((r['auc_drop'] for r in results))
    print(f'\nبیشترین افت AUC پس از گروه\u200cبندی: {max_drop:.4f}')
    if max_drop < 0.05:
        print('✅ خبر خوب: همبستگی سؤال مشترک نقش مهمی نداشته — نتایج اصلی پابرجا می\u200cمانند.')
    else:
        print('⚠️ افت قابل\u200cتوجه — باید در Limitations گزارش شود.')
if __name__ == '__main__':
    main()
