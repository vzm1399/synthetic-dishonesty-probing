import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
PARAPHRASE_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
EMBED_MODEL = 'sentence-transformers/all-mpnet-base-v2'
SEED = 42
SIM_THRESHOLD = {'level1_lexical': 0.92, 'level2_structural': 0.85, 'level3_backtranslate': 0.75}
PROMPTS = {'level1_lexical': 'Rewrite the following question using different words and sentence structure, but preserve its EXACT meaning and the exact fact it is asking about. Do not add, remove, or hint at any information. Return ONLY the rewritten question, nothing else.\n\nQuestion: {q}', 'level2_structural': 'Rewrite the following question with substantially different grammar (e.g., convert to an indirect/embedded question, reorder clauses, change voice), while preserving its EXACT meaning. Return ONLY the rewritten question, nothing else.\n\nQuestion: {q}', 'level3_backtranslate': 'Rewrite the following question as if it had been translated into French and then back into English by a competent but imperfect translator: natural but slightly unusual phrasing is expected, while the meaning must remain completely identical. Return ONLY the question, nothing else.\n\nQuestion: {q}'}

def generate_variant(model, tok, prompt, device):
    messages = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    resp = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return resp.strip().strip('"').strip()

def main():
    device = 'cuda'
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(PARAPHRASE_MODEL, torch_dtype=torch.bfloat16, device_map=device)
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    with open('tqa_original_questions.json') as f:
        originals = json.load(f)
    results, rejected = ([], 0)
    for q in originals:
        entry = {'original': q}
        orig_emb = embedder.encode(q, convert_to_tensor=True)
        accepted = True
        for level, tmpl in PROMPTS.items():
            variant = generate_variant(model, tok, tmpl.format(q=q), device)
            var_emb = embedder.encode(variant, convert_to_tensor=True)
            sim = util.cos_sim(orig_emb, var_emb).item()
            if sim < SIM_THRESHOLD[level]:
                accepted = False
                rejected += 1
                break
            entry[level] = variant
            entry[f'{level}_similarity'] = round(sim, 4)
        if accepted:
            results.append(entry)
    print(f'Accepted {len(results)} questions with all three paraphrase levels.')
    print(f'Rejected {rejected} question-level attempts because semantic similarity was too low.')
    with open('questions_and_variants.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
if __name__ == '__main__':
    main()
