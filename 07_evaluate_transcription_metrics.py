from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


BRACKET_TS_RE = re.compile(
    r"\[\s*\d+(?:[.,]\d+)?s?\s*(?:->|-->|-|–|—)\s*\d+(?:[.,]\d+)?s?\s*\]\s*",
    re.IGNORECASE,
)
LEADING_TS_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:->|-->|-|–|—)\s*",
    re.IGNORECASE,
)
SPEAKER_RE = re.compile(r"^\s*(?:SPEAKER_\d+|Speaker\s+\d+|Спикер\s+\d+)\s*:\s*", re.IGNORECASE)
WORD_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
AUDIO_NAME_RE = re.compile(r"^\s*.+\.(?:m4a|mp3|wav|ogg|flac|mp4)\s*$", re.IGNORECASE)


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def strip_service_markup(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if AUDIO_NAME_RE.match(line):
            continue

        line = BRACKET_TS_RE.sub("", line)
        line = LEADING_TS_RE.sub("", line)
        line = SPEAKER_RE.sub("", line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    text = strip_service_markup(text)
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def myers_levenshtein(reference: list[str] | str, prediction: list[str] | str) -> int:
    """Exact Levenshtein distance using Myers' bit-parallel algorithm.

    This avoids an O(N*M) Python matrix and stays fast enough for long transcripts.
    """
    if len(reference) == 0:
        return len(prediction)
    if len(prediction) == 0:
        return len(reference)

    pattern = list(reference)
    text = list(prediction)
    m = len(pattern)
    all_ones = (1 << m) - 1
    high_bit = 1 << (m - 1)

    peq: dict[str, int] = {}
    for i, item in enumerate(pattern):
        peq[item] = peq.get(item, 0) | (1 << i)

    vp = all_ones
    vn = 0
    distance = m

    for item in text:
        eq = peq.get(item, 0)
        x = eq | vn
        d0 = (((x & vp) + vp) ^ vp) | x
        hp = vn | ~(d0 | vp)
        hn = vp & d0

        if hp & high_bit:
            distance += 1
        elif hn & high_bit:
            distance -= 1

        hp = ((hp << 1) | 1) & all_ones
        hn = (hn << 1) & all_ones
        vp = (hn | ~(d0 | hp)) & all_ones
        vn = (hp & d0) & all_ones

    return distance


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def prf(overlap: float, pred_total: float, ref_total: float) -> dict[str, float]:
    precision = safe_div(overlap, pred_total)
    recall = safe_div(overlap, ref_total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def rouge_n(reference_words: list[str], prediction_words: list[str], n: int) -> dict[str, float]:
    ref_counts = ngram_counts(reference_words, n)
    pred_counts = ngram_counts(prediction_words, n)
    overlap = sum((ref_counts & pred_counts).values())
    return prf(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def lcs_length_hunt_szymanski(reference_words: list[str], prediction_words: list[str]) -> int:
    """LCS length optimized for long token sequences."""
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(prediction_words):
        positions.setdefault(token, []).append(index)

    piles: list[int] = []
    for token in reference_words:
        for index in reversed(positions.get(token, [])):
            lo = 0
            hi = len(piles)
            while lo < hi:
                mid = (lo + hi) // 2
                if piles[mid] < index:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == len(piles):
                piles.append(index)
            else:
                piles[lo] = index
    return len(piles)


def rouge_l(reference_words: list[str], prediction_words: list[str]) -> dict[str, float]:
    overlap = lcs_length_hunt_szymanski(reference_words, prediction_words)
    return prf(overlap, len(prediction_words), len(reference_words))


def bleu(reference_words: list[str], prediction_words: list[str], max_n: int) -> float:
    if not reference_words or not prediction_words:
        return 0.0

    log_precision_sum = 0.0
    for n in range(1, max_n + 1):
        ref_counts = ngram_counts(reference_words, n)
        pred_counts = ngram_counts(prediction_words, n)
        pred_total = sum(pred_counts.values())
        overlap = sum((ref_counts & pred_counts).values())
        precision = safe_div(overlap + 1.0, pred_total + 1.0)
        log_precision_sum += math.log(precision)

    brevity_penalty = 1.0
    if len(prediction_words) < len(reference_words):
        brevity_penalty = math.exp(1.0 - len(reference_words) / len(prediction_words))
    return brevity_penalty * math.exp(log_precision_sum / max_n)


def bag_token_f1(reference_words: list[str], prediction_words: list[str]) -> dict[str, float]:
    ref_counts = Counter(reference_words)
    pred_counts = Counter(prediction_words)
    common = sum((ref_counts & pred_counts).values())
    precision = safe_div(common, len(prediction_words))
    recall = safe_div(common, len(reference_words))
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def compute_metrics(reference_raw: str, prediction_raw: str) -> dict:
    reference = normalize_text(reference_raw)
    prediction = normalize_text(prediction_raw)

    ref_words = word_tokens(reference)
    pred_words = word_tokens(prediction)
    ref_chars = reference
    pred_chars = prediction
    ref_chars_no_space = reference.replace(" ", "")
    pred_chars_no_space = prediction.replace(" ", "")

    word_distance = myers_levenshtein(ref_words, pred_words)
    char_distance = myers_levenshtein(ref_chars, pred_chars)
    char_distance_no_space = myers_levenshtein(ref_chars_no_space, pred_chars_no_space)

    wer = safe_div(word_distance, len(ref_words))
    cer = safe_div(char_distance, len(ref_chars))
    cer_no_space = safe_div(char_distance_no_space, len(ref_chars_no_space))

    metrics = {
        "reference_words": len(ref_words),
        "prediction_words": len(pred_words),
        "reference_chars": len(ref_chars),
        "prediction_chars": len(pred_chars),
        "word_edit_distance": word_distance,
        "char_edit_distance": char_distance,
        "char_edit_distance_no_space": char_distance_no_space,
        "wer": wer,
        "word_accuracy": max(0.0, 1.0 - wer),
        "cer": cer,
        "char_accuracy": max(0.0, 1.0 - cer),
        "cer_no_space": cer_no_space,
        "char_accuracy_no_space": max(0.0, 1.0 - cer_no_space),
        "rouge_1": rouge_n(ref_words, pred_words, 1),
        "rouge_2": rouge_n(ref_words, pred_words, 2),
        "rouge_l": rouge_l(ref_words, pred_words),
        "bleu_1": bleu(ref_words, pred_words, 1),
        "bleu_2": bleu(ref_words, pred_words, 2),
        "bleu_4": bleu(ref_words, pred_words, 4),
    }
    metrics.update(bag_token_f1(ref_words, pred_words))
    return metrics


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_report(metrics: dict) -> None:
    print("=== Transcription quality ===")
    print(f"Reference words:   {metrics['reference_words']}")
    print(f"Prediction words:  {metrics['prediction_words']}")
    print(f"Word edit distance:{metrics['word_edit_distance']}")
    print()
    print(f"WER:               {pct(metrics['wer'])}")
    print(f"Word accuracy:     {pct(metrics['word_accuracy'])}")
    print(f"CER:               {pct(metrics['cer'])}")
    print(f"CER no spaces:     {pct(metrics['cer_no_space'])}")
    print()
    print(f"Token Precision:   {pct(metrics['token_precision'])}")
    print(f"Token Recall:      {pct(metrics['token_recall'])}")
    print(f"Token F1:          {pct(metrics['token_f1'])}")
    print()
    print(f"ROUGE-1 F1:        {pct(metrics['rouge_1']['f1'])}")
    print(f"ROUGE-2 F1:        {pct(metrics['rouge_2']['f1'])}")
    print(f"ROUGE-L F1:        {pct(metrics['rouge_l']['f1'])}")
    print(f"BLEU-1:            {pct(metrics['bleu_1'])}")
    print(f"BLEU-2:            {pct(metrics['bleu_2'])}")
    print(f"BLEU-4:            {pct(metrics['bleu_4'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reference transcript with WhisperX transcript using ASR metrics."
    )
    parser.add_argument("--reference", required=True, type=Path, help="Эталонная транскрипция")
    parser.add_argument("--prediction", required=True, type=Path, help="Транскрипция WhisperX")
    parser.add_argument("--out_json", type=Path, help="Куда сохранить метрики в JSON")
    args = parser.parse_args()

    reference_raw = read_text(args.reference)
    prediction_raw = read_text(args.prediction)
    metrics = compute_metrics(reference_raw, prediction_raw)

    print_report(metrics)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
