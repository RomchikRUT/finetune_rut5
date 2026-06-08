from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_LABELS = ["talk_start", "talk_end", "talk_inside", "non_talk"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BoundaryDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.items = []
        for rec in records:
            enc = tokenizer(
                rec["text"],
                max_length=max_length,
                truncation=True,
                padding=False,
            )
            enc["labels"] = int(rec["label_id"])
            self.items.append(enc)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def split_by_source(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_source[rec.get("source", "")].append(rec)

    sources = list(by_source)
    random.Random(seed).shuffle(sources)
    if len(sources) <= 1:
        records = records[:]
        random.Random(seed).shuffle(records)
        n_val = max(1, int(len(records) * val_ratio))
        return records[n_val:], records[:n_val]

    n_val_sources = max(1, int(round(len(sources) * val_ratio)))
    val_sources = set(sources[:n_val_sources])
    train = [rec for source in sources if source not in val_sources for rec in by_source[source]]
    val = [rec for source in sources if source in val_sources for rec in by_source[source]]
    if not train or not val:
        records = records[:]
        random.Random(seed).shuffle(records)
        n_val = max(1, int(len(records) * val_ratio))
        return records[n_val:], records[:n_val]
    return train, val


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(records: list[dict], num_labels: int) -> torch.Tensor:
    counts = Counter(int(rec["label_id"]) for rec in records)
    total = sum(counts.values())
    weights = []
    for label_id in range(num_labels):
        count = max(1, counts.get(label_id, 0))
        weights.append(total / (num_labels * count))
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="cointegrated/rubert-tiny2")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_class_weights", action="store_true")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_jsonl(args.dataset)
    if not records:
        raise RuntimeError(f"Dataset is empty: {args.dataset}")

    labels = records[0].get("labels") or DEFAULT_LABELS
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    for rec in records:
        if "label_id" not in rec:
            rec["label_id"] = label2id[rec["label"]]

    train_records, val_records = split_by_source(records, args.val_ratio, args.seed)
    print(f">>> model={args.base_model}")
    print(f">>> records={len(records)} train={len(train_records)} val={len(val_records)}")
    print(f">>> train_counts={dict(Counter(rec['label'] for rec in train_records))}")
    print(f">>> val_counts={dict(Counter(rec['label'] for rec in val_records))}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if args.device == "cuda" and torch.cuda.is_available():
        model.to("cuda")

    train_ds = BoundaryDataset(train_records, tokenizer, args.max_length)
    val_ds = BoundaryDataset(val_records, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    class_weights = None if args.no_class_weights else compute_class_weights(train_records, len(labels))
    if class_weights is not None:
        print(f">>> class_weights={class_weights.tolist()}")

    targs = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=(args.device == "cuda" and torch.cuda.is_available()),
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        class_weights=class_weights,
    )
    trainer.train()

    best_dir = args.output / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    (best_dir / "labels.json").write_text(
        json.dumps({"labels": labels, "label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f">>> done: {best_dir}")


if __name__ == "__main__":
    main()
