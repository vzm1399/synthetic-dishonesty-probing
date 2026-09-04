import csv
import json
import os

try:
    import yaml
except ImportError:
    yaml = None

RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
OUTPUT_DIR = "tables_output"

MODEL_ORDER = [
    "Llama-3.1-8B",
    "Gemma-2-9B",
    "Qwen2.5-7B",
    "Gemma-2-2B (seed 42)",
    "Gemma-2-2B (seed 123)",
    "Gemma-2-2B (seed 456)",
    "Pythia-1.4B",
]

MODEL_SLUGS = {
    "Llama-3.1-8B": "llama3.1-8b",
    "Gemma-2-9B": "gemma2-9b",
    "Qwen2.5-7B": "qwen2.5-7b",
    "Gemma-2-2B (seed 42)": "gemma2-2b_seed42",
    "Gemma-2-2B (seed 123)": "gemma2-2b_seed123",
    "Gemma-2-2B (seed 456)": "gemma2-2b_seed456",
    "Pythia-1.4B": "pythia-1.4b",
}

MISSING = []


def flag(table, model, field, path):
    MISSING.append((table, model, field, path))
    return "MISSING"


def get_field(data, key, table, model, path):
    if key in data and data[key] is not None:
        return data[key]
    return flag(table, model, key, path)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_yaml(path):
    if not os.path.exists(path) or yaml is None:
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def fmt_float(value, digits):
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return value


def md_cell(value):
    return str(value).replace("|", "\\|")


def tex_cell(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_table(name, header, rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_path = os.path.join(OUTPUT_DIR, f"{name}.md")
    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    tex_path = os.path.join(OUTPUT_DIR, f"{name}.tex")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(md_cell(x) for x in header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(md_cell(x) for x in row) + " |\n")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{" + "l" * len(header) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(tex_cell(x) for x in header) + " \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join(tex_cell(x) for x in row) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print(f"Wrote {md_path}, {csv_path}, and {tex_path}")


def target_modules_text(value):
    if isinstance(value, list):
        return ", ".join(value)
    return value


def build_table_a():
    header = ["Model", "r", "alpha", "dropout", "target_modules", "lr", "epochs", "batch", "max_len", "samples (TQA/MMLU)"]
    rows = []
    for model in MODEL_ORDER:
        slug = MODEL_SLUGS[model]
        path = f"configs/{slug}.yaml"
        cfg = load_yaml(path)
        if cfg is None:
            rows.append([model] + [flag("Table A", model, "config file", path)] * (len(header) - 1))
            continue
        samples = f"{get_field(cfg, 'samples_tqa', 'Table A', model, path)} / {get_field(cfg, 'samples_mmlu', 'Table A', model, path)}"
        rows.append([
            model,
            get_field(cfg, "r", "Table A", model, path),
            get_field(cfg, "alpha", "Table A", model, path),
            get_field(cfg, "dropout", "Table A", model, path),
            target_modules_text(get_field(cfg, "target_modules", "Table A", model, path)),
            get_field(cfg, "lr", "Table A", model, path),
            get_field(cfg, "epochs", "Table A", model, path),
            get_field(cfg, "batch_size", "Table A", model, path),
            get_field(cfg, "max_len", "Table A", model, path),
            samples,
        ])
    write_table("table_A", header, rows)


def build_table_1():
    header = ["Model", "Early layers", "Middle layers", "Late layers"]
    rows = []
    for model in MODEL_ORDER:
        slug = MODEL_SLUGS[model]
        path = f"{RESULTS_DIR}/asymmetric_noise_{slug}.json"
        data = load_json(path)
        if data is None:
            rows.append([model] + [flag("Table 1", model, "early/middle/late_auc", path)] * 3)
            continue
        rows.append([
            model,
            get_field(data, "early_auc", "Table 1", model, path),
            get_field(data, "middle_auc", "Table 1", model, path),
            get_field(data, "late_auc", "Table 1", model, path),
        ])
    write_table("table_1", header, rows)


def build_table_2():
    header = ["Model", "Fingerprint AUC (L1)", "Max AUC", "Mean |cos| w/ honesty dir.", "Log-loss increase (%)"]
    rows = []
    for model in MODEL_ORDER:
        slug = MODEL_SLUGS[model]
        path = f"{RESULTS_DIR}/lora_fingerprint_{slug}.json"
        data = load_json(path)
        if data is None:
            rows.append([model] + [flag("Table 2", model, "fingerprint summary", path)] * 4)
            continue
        rows.append([
            model,
            get_field(data, "fingerprint_auc_layer1", "Table 2", model, path),
            get_field(data, "fingerprint_auc_max", "Table 2", model, path),
            get_field(data, "mean_abs_cosine", "Table 2", model, path),
            get_field(data, "logloss_increase_pct", "Table 2", model, path),
        ])
    write_table("table_2", header, rows)


def build_table_3():
    header = ["Model", "Best AUC (TQA)", "Layer", "AUC (MMLU)", "Gen. gap", "Baseline AUC", "FT gain"]
    rows = []
    for model in MODEL_ORDER:
        slug = MODEL_SLUGS[model]
        probe_path = f"{RESULTS_DIR}/per_layer_probe_{slug}.json"
        probe = load_json(probe_path)
        if probe is None:
            best_auc = flag("Table 3", model, "best_tqa_auc", probe_path)
            best_layer = flag("Table 3", model, "best_layer", probe_path)
        else:
            aurocs = probe["auroc_tqa"]
            layers = probe["layers"]
            best_idx = max(range(len(aurocs)), key=lambda i: aurocs[i])
            best_auc = round(aurocs[best_idx], 4)
            best_layer = layers[best_idx]
        ood_path = f"{RESULTS_DIR}/ood_transfer_{slug}.json"
        ood = load_json(ood_path)
        if ood is None:
            auc_mmlu = flag("Table 3", model, "auroc_mmlu", ood_path)
            gen_gap = flag("Table 3", model, "gen_gap", ood_path)
        else:
            auc_mmlu = get_field(ood, "auroc_mmlu", "Table 3", model, ood_path)
            gen_gap = get_field(ood, "gen_gap", "Table 3", model, ood_path)
        base_path = f"{RESULTS_DIR}/baseline_{slug}.json"
        base = load_json(base_path)
        if base is None:
            baseline_auc = flag("Table 3", model, "baseline_auc", base_path)
            ft_gain = flag("Table 3", model, "ft_gain", base_path)
        else:
            baseline_auc = get_field(base, "baseline_auc", "Table 3", model, base_path)
            ft_gain = get_field(base, "ft_gain", "Table 3", model, base_path)
        rows.append([model, best_auc, best_layer, auc_mmlu, gen_gap, baseline_auc, ft_gain])
    write_table("table_3", header, rows)


def build_table_4():
    header = ["Model", "Lexical (L1)", "Structural (L2)", "Back-translate (L3)"]
    path = f"{RESULTS_DIR}/paraphrase_robustness.json"
    data = load_json(path)
    rows = []
    if data is None:
        for model in MODEL_ORDER:
            rows.append([model] + [flag("Table 4", model, "paraphrase row", path)] * 3)
        write_table("table_4", header, rows)
        return
    for i, model in enumerate(data.get("models", [])):
        rows.append([
            model,
            data["lexical"][i] if i < len(data.get("lexical", [])) else flag("Table 4", model, "lexical", path),
            data["structural"][i] if i < len(data.get("structural", [])) else flag("Table 4", model, "structural", path),
            data["back_translation"][i] if i < len(data.get("back_translation", [])) else flag("Table 4", model, "back_translation", path),
        ])
    write_table("table_4", header, rows)


def build_table_5():
    header = ["Model", "Layer", "Mean act. norm", "a=-2", "a=-1", "a=0", "a=1", "a=2", "Delta (a=2 - a=-2)"]
    rows = []
    for model in MODEL_ORDER:
        slug = MODEL_SLUGS[model]
        path = f"{RESULTS_DIR}/steering_{slug}.json"
        data = load_json(path)
        if data is None:
            rows.append([model] + [flag("Table 5", model, "steering data", path)] * (len(header) - 1))
            continue
        alpha_to_rate = dict(zip(data.get("alphas", []), data.get("incorrect_rate", [])))
        rates = []
        for alpha in [-2, -1, 0, 1, 2]:
            if alpha in alpha_to_rate:
                rates.append(alpha_to_rate[alpha])
            elif float(alpha) in alpha_to_rate:
                rates.append(alpha_to_rate[float(alpha)])
            else:
                rates.append(flag("Table 5", model, f"alpha={alpha}", path))
        delta = round(rates[-1] - rates[0], 3) if all(isinstance(x, (int, float)) for x in [rates[0], rates[-1]]) else "MISSING"
        rows.append([
            model,
            get_field(data, "layer", "Table 5", model, path),
            fmt_float(get_field(data, "mean_activation_norm", "Table 5", model, path), 4),
            *[fmt_float(rate, 4) for rate in rates],
            fmt_float(delta, 3),
        ])
    write_table("table_5", header, rows)


def write_missing_report():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "MISSING_VALUES_REPORT.md")
    with open(path, "w") as f:
        if not MISSING:
            f.write("No missing values. All tables fully reproduced from results/ and configs/.\n")
            print("No missing values. All tables complete.")
        else:
            f.write(f"# Missing values report ({len(MISSING)} issues)\n\n")
            f.write("| Table | Model | Field | Expected file |\n|---|---|---|---|\n")
            for table, model, field, src in MISSING:
                f.write(f"| {table} | {model} | {field} | `{src}` |\n")
            print(f"{len(MISSING)} missing value(s) found. See {path}.")
    print(f"Wrote {path}")


def main():
    build_table_a()
    build_table_1()
    build_table_2()
    build_table_3()
    build_table_4()
    build_table_5()
    write_missing_report()


if __name__ == "__main__":
    main()
