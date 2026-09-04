import argparse
import json
from pathlib import Path
import numpy as np


def centroid_geometry(honest, dishonest):
    rows = []
    for layer in range(honest.shape[1]):
        h_mean = honest[:, layer, :].mean(axis=0)
        d_mean = dishonest[:, layer, :].mean(axis=0)
        diff = d_mean - h_mean
        h_unit = h_mean / (np.linalg.norm(h_mean) + 1e-12)
        d_unit = d_mean / (np.linalg.norm(d_mean) + 1e-12)
        direction = diff / (np.linalg.norm(diff) + 1e-12)
        h_proj = honest[:, layer, :] @ direction
        d_proj = dishonest[:, layer, :] @ direction
        rows.append({'layer': layer, 'centroid_l2_distance': float(np.linalg.norm(diff)), 'centroid_cosine_distance': float(1.0 - np.dot(h_unit, d_unit)), 'honest_projection_mean': float(h_proj.mean()), 'dishonest_projection_mean': float(d_proj.mean()), 'projection_margin': float(d_proj.mean() - h_proj.mean())})
    return {'centroid_geometry': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--honest-activations', required=True)
    parser.add_argument('--dishonest-activations', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = centroid_geometry(np.load(args.honest_activations), np.load(args.dishonest_activations))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
