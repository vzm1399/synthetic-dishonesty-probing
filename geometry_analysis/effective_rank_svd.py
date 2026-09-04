import argparse
import json
from pathlib import Path
import numpy as np


def effective_rank_by_layer(honest, dishonest):
    rows = []
    for layer in range(honest.shape[1]):
        x = np.concatenate([honest[:, layer, :], dishonest[:, layer, :]], axis=0).astype(np.float64)
        x = x - x.mean(axis=0, keepdims=True)
        _, singular_values, _ = np.linalg.svd(x, full_matrices=False)
        variance = singular_values ** 2
        explained = variance / (variance.sum() + 1e-30)
        probabilities = explained / (explained.sum() + 1e-30)
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-30))))
        rows.append({'layer': layer, 'effective_rank': effective_rank, 'variance_top1': float(explained[:1].sum()), 'variance_top2': float(explained[:2].sum()), 'variance_top5': float(explained[:5].sum()), 'variance_top10': float(explained[:10].sum())})
    return {'effective_rank_svd': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--honest-activations', required=True)
    parser.add_argument('--dishonest-activations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = effective_rank_by_layer(np.load(args.honest_activations), np.load(args.dishonest_activations))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
