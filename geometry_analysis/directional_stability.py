import argparse
import json
from pathlib import Path
import numpy as np


def load_direction_stats(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = data.get('direction', {}).get('adjacent_layer_cosine', [])
    return {int(row['layer']): float(row['cosine_sim']) for row in rows}


def directional_stability(seed_files):
    per_seed = {Path(path).stem: load_direction_stats(path) for path in seed_files}
    layers = sorted(set().union(*(set(values) for values in per_seed.values())))
    rows = []
    for layer in layers:
        values = [stats[layer] for stats in per_seed.values() if layer in stats]
        rows.append({'layer': layer, 'mean_adjacent_cosine': float(np.mean(values)), 'std_adjacent_cosine': float(np.std(values)), 'n_seeds': len(values), 'per_seed': values})
    return {'directional_stability': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-files', nargs='+', default=['results/analysis_results_gemma2-2b_seed42.json', 'results/analysis_results_gemma2-2b_seed123.json', 'results/analysis_results_gemma2-2b_seed456.json'])
    parser.add_argument('--output', default='results/directional_stability_gemma2-2b.json')
    args = parser.parse_args()
    result = directional_stability(args.seed_files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
