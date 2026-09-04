import argparse
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset

DEFAULT_SUBJECTS = ['high_school_biology', 'high_school_chemistry', 'high_school_physics', 'high_school_world_history', 'philosophy', 'elementary_mathematics', 'college_mathematics']


def load_subject(subject):
    try:
        return load_dataset('cais/mmlu', subject, split='test')
    except Exception:
        return load_dataset('cais/mmlu', subject, split='validation')


def build_mmlu_subset(subjects, samples_per_subject=None, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for subject in subjects:
        subject_rows = []
        for item in load_subject(subject):
            choices = item['choices']
            answer_index = int(item['answer'])
            correct = choices[answer_index]
            wrong_indices = [i for i in range(len(choices)) if i != answer_index]
            if not wrong_indices:
                continue
            wrong = choices[wrong_indices[0]]
            subject_rows.append({'subject': subject, 'question': item['question'], 'choices': choices, 'answer_index': answer_index, 'honest_answer': correct, 'dishonest_answer': wrong, 'honest_text': f"Q: {item['question']}\nA: {correct}", 'dishonest_text': f"Q: {item['question']}\nA: {wrong}"})
        if samples_per_subject is not None and len(subject_rows) > samples_per_subject:
            selected = rng.choice(len(subject_rows), samples_per_subject, replace=False)
            subject_rows = [subject_rows[i] for i in selected]
        rows.extend(subject_rows)
    return rows


def write_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/mmlu_subset.jsonl')
    parser.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS)
    parser.add_argument('--samples-per-subject', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rows = build_mmlu_subset(args.subjects, args.samples_per_subject, args.seed)
    write_jsonl(rows, args.output)
    print(f'wrote {len(rows)} MMLU rows to {args.output}')


if __name__ == '__main__':
    main()
