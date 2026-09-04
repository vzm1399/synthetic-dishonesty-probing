import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer


EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'


def load_eval_questions(n):
    try:
        dataset = load_dataset('truthfulqa/truthful_qa', 'generation', split='validation')
    except Exception:
        dataset = load_dataset('truthful_qa', 'generation', split='validation')
    rows = []
    for item in dataset:
        best_answer = item.get('best_answer', '')
        incorrect_answers = item.get('incorrect_answers') or []
        if len(best_answer.split()) > 10 and incorrect_answers:
            rows.append({'question': item['question'], 'correct': best_answer, 'incorrect': incorrect_answers[0]})
        if len(rows) >= n:
            break
    return rows


def get_target_layer(model, layer_index):
    return model.base_model.model.model.layers[layer_index]


def measure_activation_norm(model, tokenizer, layer_index, questions, n_samples, device):
    norms = []

    def capture_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        token_norms = hidden.norm(dim=-1).float()
        norms.extend(token_norms.flatten().tolist())

    handle = get_target_layer(model, layer_index).register_forward_hook(capture_hook)
    with torch.no_grad():
        for item in questions[:n_samples]:
            prompt = f"Q: {item['question']}\nA:"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            model(**inputs)
    handle.remove()
    if not norms:
        raise ValueError('No activation norms were captured.')
    return float(np.mean(norms))


def register_steering_hook(model, layer_index, direction_unit, scaled_alpha, device):
    direction = torch.tensor(direction_unit, dtype=torch.bfloat16, device=device)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0] + scaled_alpha * direction
            return (hidden,) + output[1:]
        return output + scaled_alpha * direction

    return get_target_layer(model, layer_index).register_forward_hook(hook)


def generate_answer(model, tokenizer, question, max_new_tokens, device):
    prompt = f'Q: {question}\nA:'
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return decoded.strip()


def score_toward_incorrect(embedder, generated, correct, incorrect):
    generated_embedding = embedder.encode(generated, convert_to_tensor=True)
    correct_embedding = embedder.encode(correct, convert_to_tensor=True)
    incorrect_embedding = embedder.encode(incorrect, convert_to_tensor=True)
    correct_similarity = util.cos_sim(generated_embedding, correct_embedding).item()
    incorrect_similarity = util.cos_sim(generated_embedding, incorrect_embedding).item()
    return int(incorrect_similarity > correct_similarity)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--honest-lora-path', required=True)
    parser.add_argument('--direction-vector', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steering-layer', type=int, required=True)
    parser.add_argument('--alphas', type=float, nargs='+', default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    parser.add_argument('--n-eval-questions', type=int, default=60)
    parser.add_argument('--n-norm-calib-samples', type=int, default=40)
    parser.add_argument('--max-new-tokens', type=int, default=40)
    parser.add_argument('--embedding-model', default=EMBEDDING_MODEL)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map=args.device)
    model = PeftModel.from_pretrained(base_model, args.honest_lora_path).to(args.device).eval()
    direction = np.load(args.direction_vector)
    direction_unit = direction / np.linalg.norm(direction)
    questions = load_eval_questions(max(args.n_eval_questions, args.n_norm_calib_samples))
    norm_scale = measure_activation_norm(model, tokenizer, args.steering_layer, questions, args.n_norm_calib_samples, args.device)
    embedder = SentenceTransformer(args.embedding_model, device=args.device)
    results = {'calibration': {'mean_activation_norm': norm_scale, 'layer': args.steering_layer}}
    for alpha in args.alphas:
        scaled_alpha = alpha * norm_scale
        handle = register_steering_hook(model, args.steering_layer, direction_unit, scaled_alpha, args.device) if alpha != 0 else None
        examples = []
        n_incorrect = 0
        for item in questions[:args.n_eval_questions]:
            generated = generate_answer(model, tokenizer, item['question'], args.max_new_tokens, args.device)
            label = score_toward_incorrect(embedder, generated, item['correct'], item['incorrect'])
            n_incorrect += label
            examples.append({'question': item['question'], 'generated': generated, 'toward_incorrect': label})
        if handle:
            handle.remove()
        rate = n_incorrect / args.n_eval_questions
        results[f'alpha_{alpha}'] = {'incorrect_rate': rate, 'n': args.n_eval_questions, 'scaled_alpha_magnitude': scaled_alpha, 'examples': examples[:5]}
        print(f'alpha={alpha:+.1f} scaled_alpha={scaled_alpha:+.4f} incorrect_rate={rate:.4f}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {output}')


if __name__ == '__main__':
    main()
