import argparse
import json
from pathlib import Path
import numpy as np


def interdomain_alignment(tqa_honest, tqa_dishonest, mmlu_honest, mmlu_dishonest):
    rows = []
    for layer in range(tqa_honest.shape[1]):
        tqa_diff = tqa_dishonest[:, layer, :].mean(axis=0) - tqa_honest[:, layer, :].mean(axis=0)
        mmlu_diff = mmlu_dishonest[:, layer, :].mean(axis=0) - mmlu_honest[:, layer, :].mean(axis=0)
        tqa_unit = tqa_diff / (np.linalg.norm(tqa_diff) + 1e-12)
        mmlu_unit = mmlu_diff / (np.linalg.norm(mmlu_diff) + 1e-12)
        rows.append({'layer': layer, 'tqa_mmlu_direction_cosine': float(np.dot(tqa_unit, mmlu_unit)), 'tqa_direction_norm': float(np.linalg.norm(tqa_diff)), 'mmlu_direction_norm': float(np.linalg.norm(mmlu_diff))})
    return {'interdomain_alignment': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tqa-honest', required=True)
    parser.add_argument('--tqa-dishonest', required=True)
    parser.add_argument('--mmlu-honest', required=True)
    parser.add_argument('--mmlu-dishonest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = interdomain_alignment(np.load(args.tqa_honest), np.load(args.tqa_dishonest), np.load(args.mmlu_honest), np.load(args.mmlu_dishonest))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
