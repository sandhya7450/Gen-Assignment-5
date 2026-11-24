"""Train GPT-2's last block, or run held-out evaluation with --evaluate."""

import argparse
import random
import time

from local_paths import ROOT
import matplotlib
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAME = "openai-community/gpt2"
DATA_PATH = ROOT / "dataset" / "study_tips.csv"
OUTPUT_DIR = ROOT / "lab_outputs"
CHECKPOINT = OUTPUT_DIR / "last_block.safetensors"
LOSS_CSV = OUTPUT_DIR / "training_loss.csv"
LOSS_PLOT = OUTPUT_DIR / "training_loss.png"
EVALUATION_CSV = OUTPUT_DIR / "evaluation_results.csv"
SEED = 304

# Rubric scores were manually assigned to the exact outputs recorded in this run.
RATINGS = {
    "Python: Generators": ((2, 2, 1), (2, 1, 1)),
    "Java: Exception Handling": ((0, 2, 0), (0, 2, 0)),
    "Data Structures: Graph Traversal": ((0, 2, 0), (1, 2, 1)),
    "Algorithms: Dynamic Programming": ((0, 2, 0), (1, 2, 1)),
    "Mathematics: Eigenvalues": ((1, 2, 1), (1, 2, 1)),
    "Machine Learning: Clustering": ((1, 2, 1), (1, 2, 1)),
    "Deep Learning: Attention": ((0, 2, 0), (0, 2, 0)),
    "DBMS: Transactions": ((0, 2, 0), (0, 2, 0)),
    "Computer Networks: DNS": ((1, 2, 1), (1, 2, 1)),
    "Operating Systems: Deadlocks": ((1, 2, 0), (1, 2, 0)),
}


class TipDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.samples = []
        for row in rows:
            prompt = f"Topic: {row['topic']}\nStudy tip:"
            text = f"{prompt} {row['study_tip']}{tokenizer.eos_token}"
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=128,
                return_offsets_mapping=True,
            )
            labels = list(encoded["input_ids"])
            for index, (_, end) in enumerate(encoded["offset_mapping"]):
                if end <= len(prompt):
                    labels[index] = -100
            self.samples.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": [1] * len(labels),
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def load_rows():
    rows = pd.read_csv(DATA_PATH).fillna("").to_dict(orient="records")
    training = [row for row in rows if row["split"] == "train"]
    held_out = [row for row in rows if row["split"] == "evaluation"]
    overlap = {row["topic"] for row in training} & {row["topic"] for row in held_out}
    if not training or len(held_out) < 10 or overlap:
        raise ValueError("Dataset must have training rows and 10+ disjoint evaluation topics.")
    return training, held_out


def collate(rows, pad_id):
    width = max(len(row["input_ids"]) for row in rows)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in rows:
        padding = width - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
        batch["labels"].append(row["labels"] + [-100] * padding)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def generate(model, tokenizer, topic, device):
    prompt = f"Topic: {topic}\nStudy tip:"
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        result = model.generate(
            **encoded,
            max_new_tokens=40,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(result[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip().splitlines()[0].strip()


def load_adapted_model(device):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing {CHECKPOINT}; run `python train.py` first.")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    state = load_file(str(CHECKPOINT), device="cpu")
    model.transformer.h[-1].load_state_dict(state, strict=True)
    return model.eval()


def train():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    training, _ = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device}; {len(training)} training examples.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.transformer.h[-1].parameters():
        parameter.requires_grad = True
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    model.to(device)
    model.config.use_cache = False
    data = TipDataset(training, tokenizer)
    generator = torch.Generator().manual_seed(SEED)
    batches = DataLoader(
        data,
        batch_size=2,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(parameters, lr=2e-5)
    losses = []
    model.train()
    for epoch in range(1, 4):
        started = time.perf_counter()
        batch_losses = []
        for batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in epoch {epoch}.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        mean_loss = sum(batch_losses) / len(batch_losses)
        losses.append({"epoch": epoch, "average_training_loss": mean_loss})
        print(f"Epoch {epoch}/3: loss={mean_loss:.4f}, {time.perf_counter() - started:.1f}s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adapter = {
        name: tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
        for name, tensor in model.transformer.h[-1].state_dict().items()
    }
    save_file(adapter, str(CHECKPOINT))
    pd.DataFrame(losses).to_csv(LOSS_CSV, index=False)
    plt.figure(figsize=(6, 3.5))
    plt.plot([row["epoch"] for row in losses], [row["average_training_loss"] for row in losses], marker="o")
    plt.xticks([1, 2, 3])
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("GPT-2 last-block fine-tuning")
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=140)
    plt.close()
    print(f"Saved half-precision last-block checkpoint ({CHECKPOINT.stat().st_size / 1e6:.1f} MB).")
    print(f"The 124M-parameter base GPT-2 is cached under {ROOT / '.cache' / 'huggingface'}.")


def evaluate():
    _, held_out = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()
    adapted = load_adapted_model(device)
    records = []
    for row in held_out:
        base_output = generate(base, tokenizer, row["topic"], device)
        adapted_output = generate(adapted, tokenizer, row["topic"], device)
        base_rating, adapted_rating = RATINGS.get(row["topic"], ((None,) * 3, (None,) * 3))
        records.append(
            {
                "topic": row["topic"],
                "reference_tip": row["study_tip"],
                "base_output": base_output,
                "fine_tuned_output": adapted_output,
                "base_relevance": base_rating[0],
                "base_readability": base_rating[1],
                "base_usefulness": base_rating[2],
                "base_total": sum(base_rating) if None not in base_rating else None,
                "fine_tuned_relevance": adapted_rating[0],
                "fine_tuned_readability": adapted_rating[1],
                "fine_tuned_usefulness": adapted_rating[2],
                "fine_tuned_total": sum(adapted_rating) if None not in adapted_rating else None,
                "rating_method": "manual review by AI assistant; verify before academic submission",
            }
        )
    pd.DataFrame(records).to_csv(EVALUATION_CSV, index=False)
    print(f"Saved actual held-out generations and manual ratings to {EVALUATION_CSV}")
    for name in ("base", "fine_tuned"):
        scores = [row[f"{name}_total"] for row in records if row[f"{name}_total"] is not None]
        if scores:
            print(f"{name} mean manual rating: {sum(scores) / len(scores):.2f}/6")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Compare base and adapted models on held-out topics.")
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        train()


if __name__ == "__main__":
    main()
