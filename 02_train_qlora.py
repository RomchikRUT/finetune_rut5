"""
Обучение LoRA-адаптера для иерархической суммаризации.

Каждый запуск тренирует ОДИН адаптер для одной задачи (--task header или thesis).
В датасете dataset.jsonl содержатся примеры обоих типов (поле "task"); скрипт
фильтрует их по флагу --task.

ВАЖНО:
  - bitsandbytes НЕ используется (запрещён на Windows из-за access violations).
  - библиотека `datasets` НЕ используется (конфликт OpenMP после torch).
  - KMP_DUPLICATE_LIB_OK=TRUE обязательно (выставляется в самом скрипте).

Пример:
    python 02_train_qlora.py \\
        --base_model utrobinmv/t5_summary_en_ru_zh_large_2048 \\
        --dataset ./data/dataset.jsonl \\
        --output  ./checkpoints/header \\
        --task    header \\
        --epochs  25

    python 02_train_qlora.py \\
        --base_model utrobinmv/t5_summary_en_ru_zh_large_2048 \\
        --dataset ./data/dataset.jsonl \\
        --output  ./checkpoints/thesis \\
        --task    thesis \\
        --epochs  25
"""
import argparse, json, os, random
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments,
)

print(">>> script started", flush=True)


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class CleanCollator:
    """Обёртка над DataCollatorForSeq2Seq — выкидывает служебные поля типа __length__."""
    def __init__(self, base):
        self.base = base
    def __call__(self, features):
        cleaned = []
        for f in features:
            cleaned.append({k: v for k, v in f.items()
                            if k in ("input_ids", "attention_mask", "labels")})
        return self.base(cleaned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--task", required=True, choices=["header", "thesis"],
                    help="Тип задачи: header — заголовок сегмента, thesis — один тезис")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--max_source", type=int, default=1950,
                    help="Потолок базовой модели — 2048. Держим запас на спецтокены.")
    ap.add_argument("--max_target", type=int, default=96,
                    help="Целевая длина: для одной строки 'MM:SS - MM:SS: ...' достаточно 64-96 токенов")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    print(f">>> cuda={torch.cuda.is_available()} dataset={args.dataset}", flush=True)
    print(f">>> base_model={args.base_model}", flush=True)
    print(f">>> task={args.task}", flush=True)
    print(f">>> max_source={args.max_source} max_target={args.max_target} lr={args.lr}", flush=True)

    all_recs = load_jsonl(args.dataset)
    recs = [r for r in all_recs if r.get("task") == args.task]
    print(f">>> filtered by task='{args.task}': {len(recs)} / {len(all_recs)} записей", flush=True)
    if len(recs) < 2:
        raise RuntimeError(
            f"Недостаточно примеров для task='{args.task}'. "
            f"Найдено {len(recs)}. Проверь, что 01_prepare_dataset.py запущен и "
            f"в эталонах есть строки нужного типа."
        )

    random.Random(args.seed).shuffle(recs)
    n_val = max(1, int(len(recs) * 0.1))
    val_recs = recs[:n_val]; train_recs = recs[n_val:]
    print(f">>> split: train={len(train_recs)} val={len(val_recs)}", flush=True)

    print(">>> tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    def tok_record(r):
        mi = tokenizer(r["input"], max_length=args.max_source, truncation=True, padding=False)
        lb = tokenizer(text_target=r["target"], max_length=args.max_target,
                       truncation=True, padding=False)
        return {
            "input_ids": mi["input_ids"],
            "attention_mask": mi["attention_mask"],
            "labels": lb["input_ids"],
            "__length__": len(mi["input_ids"]),
        }

    train_ds = [tok_record(r) for r in train_recs]
    val_ds   = [tok_record(r) for r in val_recs]
    lens = [x["__length__"] for x in train_ds]
    print(f">>> tokenized: avg_len={sum(lens)//len(lens)} max_len={max(lens)}", flush=True)

    print(">>> model...", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
        target_modules=["q", "k", "v", "o", "wi", "wi_0", "wi_1", "wo"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    base_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")
    collator = CleanCollator(base_collator)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="__length__",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=collator,
    )
    print(">>> training...", flush=True)
    trainer.train()
    trainer.save_model(str(args.output / "best"))
    tokenizer.save_pretrained(str(args.output / "best"))
    # сохраним метку задачи рядом с адаптером, чтобы инференс мог проверить
    (args.output / "best" / "task.txt").write_text(args.task, encoding="utf-8")
    print(f">>> done. adapter -> {args.output / 'best'} (task={args.task})", flush=True)


if __name__ == "__main__":
    main()
