import argparse
import json
from pathlib import Path
from datasets import load_dataset


def build_truthfulqa_pairs(max_samples=None):
    try:
        dataset = load_dataset('truthfulqa/truthful_qa', 'generation', split='validation')
    except Exception:
        dataset = load_dataset('truthful_qa', 'generation', split='validation')
    rows = []
    for item in dataset:
        question = item['question']
        correct_answers = item.get('correct_answers') or []
        incorrect_answers = item.get('incorrect_answers') or []
        if not correct_answers or not incorrect_answers:
            continue
        honest = correct_answers[0].strip()
        dishonest = incorrect_answers[0].strip()
        if len(honest.split()) <= 2 or len(dishonest.split()) <= 2:
            continue
        rows.append({'question': question, 'honest_answer': honest, 'dishonest_answer': dishonest, 'honest_text': f'Q: {question}\nA: {honest}', 'dishonest_text': f'Q: {question}\nA: {dishonest}'})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def write_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/truthfulqa_pairs.jsonl')
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()
    rows = build_truthfulqa_pairs(args.max_samples)
    write_jsonl(rows, args.output)
    print(f'wrote {len(rows)} pairs to {args.output}')


if __name__ == '__main__':
    main()
