import argparse
import json
from pathlib import Path
import numpy as np


def fisher_discriminant_ratio(honest, dishonest):
    rows = []
    for layer in range(honest.shape[1]):
        h = honest[:, layer, :]
        d = dishonest[:, layer, :]
        between = float(np.linalg.norm(d.mean(axis=0) - h.mean(axis=0)) ** 2)
        within_honest = float(h.var(axis=0).mean())
        within_dishonest = float(d.var(axis=0).mean())
        fdr = between / (within_honest + within_dishonest + 1e-12)
        rows.append({'layer': layer, 'fisher_discriminant_ratio': float(fdr), 'between_class_variance': between, 'within_honest_variance': within_honest, 'within_dishonest_variance': within_dishonest})
    return {'fisher_discriminant_ratio': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--honest-activations', required=True)
    parser.add_argument('--dishonest-activations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = fisher_discriminant_ratio(np.load(args.honest_activations), np.load(args.dishonest_activations))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
