import csv
import json
from pathlib import Path
import numpy as np


ANALYSIS_FILES = {
    'pythia-1.4b': 'analysis_results_pythia14b.json',
    'llama3.1-8b': 'analysis_results_llama3.1-8b.json',
    'gemma2-9b': 'analysis_results_gemma2-9b.json',
    'qwen2.5-7b': 'analysis_results_qwen2.5-7b.json',
    'gemma2-2b_seed42': 'analysis_results_gemma2-2b_seed42.json',
    'gemma2-2b_seed123': 'analysis_results_gemma2-2b_seed123.json',
    'gemma2-2b_seed456': 'analysis_results_gemma2-2b_seed456.json',
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def max_field(rows, field):
    values = [row[field] for row in rows if field in row and row[field] is not None]
    return float(max(values)) if values else None


def mean_field(rows, field):
    values = [row[field] for row in rows if field in row and row[field] is not None]
    return float(np.mean(values)) if values else None


def summarize_analysis(results_dir):
    rows = []
    for model, filename in ANALYSIS_FILES.items():
        data = load_json(results_dir / filename)
        pca = data.get('pca', {}).get('pca_per_layer', [])
        direction = data.get('direction', {}).get('layer_stats', [])
        fisher = data.get('fisher', {}).get('fisher_per_layer', [])
        cross_domain = data.get('cross_domain', {}).get('cross_domain_direction', [])
        calibration = data.get('calibration', {}).get('reliability_diagram', {})
        transition = data.get('transition', {}).get('transition_analysis', {})
        rows.append({'model': model, 'max_effective_rank': max_field(pca, 'eff_rank'), 'max_fisher_discriminant_ratio': max_field(fisher, 'fisher_discriminant_ratio'), 'max_centroid_cosine_distance': max_field(direction, 'centroid_cosine_distance'), 'mean_tqa_mmlu_alignment': mean_field(cross_domain, 'tqa_mmlu_direction_cosine'), 'best_calibration_layer': calibration.get('best_layer'), 'best_calibration_auc': calibration.get('best_layer_auc'), 'ece': calibration.get('ece'), 'transition_layer': transition.get('transition_layer')})
    return rows


def summarize_fingerprint(results_dir):
    path = results_dir / 'lora_fingerprint_control_TRUE_Gemma-2-9B.json'
    rows = load_json(path)
    grouped = [row.get('grouped_fingerprint_auc') for row in rows if row.get('grouped_fingerprint_auc') is not None]
    stratified = [row.get('stratified_fingerprint_auc') for row in rows if row.get('stratified_fingerprint_auc') is not None]
    return [{'model': 'gemma2-9b', 'max_grouped_fingerprint_auc': float(max(grouped)) if grouped else None, 'mean_grouped_fingerprint_auc': float(np.mean(grouped)) if grouped else None, 'max_stratified_fingerprint_auc': float(max(stratified)) if stratified else None, 'n_layers': len(rows)}]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    results_dir = Path('results')
    output_dir = Path('generated_tables')
    analysis_rows = summarize_analysis(results_dir)
    fingerprint_rows = summarize_fingerprint(results_dir)
    write_csv(output_dir / 'table_geometry_summary.csv', analysis_rows)
    write_csv(output_dir / 'table_lora_fingerprint_control.csv', fingerprint_rows)
    print(f'wrote tables to {output_dir}')


if __name__ == '__main__':
    main()
