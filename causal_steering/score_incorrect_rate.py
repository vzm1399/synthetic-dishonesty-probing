import argparse
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util


def read_generations(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open(encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'examples' in data:
        return data['examples']
    rows = []
    for value in data.values():
        if isinstance(value, dict) and 'examples' in value:
            rows.extend(value['examples'])
    return rows


def score_toward_incorrect(embedder, generated, correct, incorrect):
    g_emb = embedder.encode(generated, convert_to_tensor=True)
    c_emb = embedder.encode(correct, convert_to_tensor=True)
    i_emb = embedder.encode(incorrect, convert_to_tensor=True)
    sim_correct = util.cos_sim(g_emb, c_emb).item()
    sim_incorrect = util.cos_sim(g_emb, i_emb).item()
    return {'toward_incorrect': int(sim_incorrect > sim_correct), 'similarity_correct': float(sim_correct), 'similarity_incorrect': float(sim_incorrect)}


def score_rows(rows, embedding_model, device=None):
    embedder = SentenceTransformer(embedding_model, device=device)
    scored = []
    for row in rows:
        generated = row.get('generated') or row.get('completion') or row.get('answer') or ''
        correct = row.get('correct') or row.get('correct_answer') or row.get('honest_answer')
        incorrect = row.get('incorrect') or row.get('incorrect_answer') or row.get('dishonest_answer')
        if correct is None or incorrect is None:
            raise KeyError('Each row must contain correct/incorrect answer fields.')
        score = score_toward_incorrect(embedder, generated, correct, incorrect)
        scored.append({**row, **score})
    rate = sum(row['toward_incorrect'] for row in scored) / max(len(scored), 1)
    return {'incorrect_rate': rate, 'n': len(scored), 'examples': scored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--embedding-model', default='sentence-transformers/all-mpnet-base-v2')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    results = score_rows(read_generations(args.generations), args.embedding_model, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"incorrect_rate={results['incorrect_rate']:.4f} n={results['n']}")


if __name__ == '__main__':
    main()
