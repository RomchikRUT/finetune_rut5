from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean


TC_RE = r"\d{1,2}:\d{2}(?::\d{2})?"
RANGE_LINE_RE = re.compile(
    rf"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?"
    rf"(?P<start>{TC_RE})\s*[-\u2013\u2014]\s*(?P<end>{TC_RE})"
    rf"\s*:\s*(?P<text>.+?)\s*$"
)
IMPORTANT_RE = re.compile(r"важн\w*\s+момент\w*", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")

NON_TALK_HEADER_PATTERNS = (
    re.compile(r"\bвопрос(ы|ов|ам|ами|ах)?\b", re.IGNORECASE),
    re.compile(r"\bответ(ы|ов)?\b", re.IGNORECASE),
    re.compile(r"\bобсуждени[ея]\b", re.IGNORECASE),
    re.compile(r"\bдискусси", re.IGNORECASE),
    re.compile(r"\bвступительн", re.IGNORECASE),
    re.compile(r"\bвступлени[ея]\b", re.IGNORECASE),
    re.compile(r"\bоткрыти[ея]\b", re.IGNORECASE),
    re.compile(r"\bзаключительн", re.IGNORECASE),
    re.compile(r"\bзаключени[ея]\b", re.IGNORECASE),
    re.compile(r"\bзакрыти[ея]\b", re.IGNORECASE),
    re.compile(r"\bзавершени[ея]\b", re.IGNORECASE),
    re.compile(r"\bподведени[ея]\s+итог", re.IGNORECASE),
    re.compile(r"\bорганизационн", re.IGNORECASE),
    re.compile(r"\bслужебн", re.IGNORECASE),
    re.compile(r"\bперерыв\b", re.IGNORECASE),
    re.compile(r"\bрегламент\b", re.IGNORECASE),
)

STOP_WORDS = {
    "это", "как", "что", "для", "при", "или", "над", "под", "без", "про", "его",
    "ее", "они", "она", "оно", "тем", "том", "тот", "эта", "эти", "так", "уже",
    "еще", "где", "кто", "чем", "чего", "того", "были", "был", "была", "будет",
    "также", "который", "которая", "которые", "которых", "данный", "данные",
    "доклад", "доклада", "тема", "темы", "обсуждается", "обсуждаются",
}


@dataclass
class Thesis:
    start: int
    end: int
    text: str
    parent_index: int = -1


@dataclass
class Segment:
    start: int
    end: int
    header: str
    theses: list[Thesis] = field(default_factory=list)


@dataclass
class Match:
    ref_index: int
    pred_index: int
    score: float


def parse_timecode(value: str) -> int:
    parts = [int(x) for x in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Bad timecode: {value}")


def fmt_time(seconds: int | float) -> str:
    sec = max(0, int(round(seconds)))
    hours, rem = divmod(sec, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def parse_annotation(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    current: Segment | None = None
    in_theses = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            in_theses = False
            continue
        if line.startswith("#"):
            continue
        if IMPORTANT_RE.search(line):
            in_theses = True
            continue
        if "тезисы не сформированы" in line.lower():
            continue

        match = RANGE_LINE_RE.match(line)
        if not match:
            continue

        start = parse_timecode(match.group("start"))
        end = parse_timecode(match.group("end"))
        text = match.group("text").strip().strip("-").strip()
        if end <= start:
            end = start + 1

        if in_theses and current is not None:
            current.theses.append(Thesis(start=start, end=end, text=text))
            continue

        if current is not None:
            segments.append(current)
        current = Segment(start=start, end=end, header=text)

    if current is not None:
        segments.append(current)
    for i, segment in enumerate(segments):
        for thesis in segment.theses:
            thesis.parent_index = i
    return segments


def normalize_text(text: str) -> str:
    return text.lower().replace("ё", "е")


def text_tokens(text: str) -> list[str]:
    words = WORD_RE.findall(normalize_text(text))
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def token_f1(a: str, b: str) -> float:
    left = Counter(text_tokens(a))
    right = Counter(text_tokens(b))
    if not left or not right:
        return 0.0
    common = sum((left & right).values())
    if common == 0:
        return 0.0
    precision = common / sum(right.values())
    recall = common / sum(left.values())
    return 2 * precision * recall / (precision + recall)


def char_ngrams(text: str, n: int = 3) -> set[str]:
    cleaned = " ".join(WORD_RE.findall(normalize_text(text)))
    if not cleaned:
        return set()
    if len(cleaned) <= n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def char_ngram_f1(a: str, b: str, n: int = 3) -> float:
    left = char_ngrams(a, n=n)
    right = char_ngrams(b, n=n)
    if not left or not right:
        return 0.0
    common = len(left & right)
    if common == 0:
        return 0.0
    precision = common / len(right)
    recall = common / len(left)
    return 2 * precision * recall / (precision + recall)


def text_similarity(a: str, b: str) -> float:
    return 0.7 * token_f1(a, b) + 0.3 * char_ngram_f1(a, b)


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def prf(overlap: float, pred_total: float, ref_total: float) -> dict:
    precision = safe_div(overlap, pred_total)
    recall = safe_div(overlap, ref_total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def rouge_n(reference: str, prediction: str, n: int) -> dict:
    ref_counts = ngram_counts(text_tokens(reference), n)
    pred_counts = ngram_counts(text_tokens(prediction), n)
    overlap = sum((ref_counts & pred_counts).values())
    return prf(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(right) > len(left):
        left, right = right, left
    prev = [0] * (len(right) + 1)
    for left_token in left:
        cur = [0]
        diagonal = 0
        for j, right_token in enumerate(right, start=1):
            up = prev[j]
            if left_token == right_token:
                cur.append(diagonal + 1)
            else:
                cur.append(max(cur[-1], up))
            diagonal = up
        prev = cur
    return prev[-1]


def rouge_l(reference: str, prediction: str) -> dict:
    ref_tokens = text_tokens(reference)
    pred_tokens = text_tokens(prediction)
    overlap = lcs_length(ref_tokens, pred_tokens)
    return prf(overlap, len(pred_tokens), len(ref_tokens))


def bleu(reference: str, prediction: str, max_n: int = 4) -> float:
    ref_tokens = text_tokens(reference)
    pred_tokens = text_tokens(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    log_precision_sum = 0.0
    for n in range(1, max_n + 1):
        ref_counts = ngram_counts(ref_tokens, n)
        pred_counts = ngram_counts(pred_tokens, n)
        pred_total = sum(pred_counts.values())
        overlap = sum((ref_counts & pred_counts).values())
        # Add-one smoothing keeps BLEU usable for short Russian summaries.
        precision = (overlap + 1.0) / (pred_total + 1.0)
        log_precision_sum += math.log(precision)

    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1.0 - len(ref_tokens) / len(pred_tokens))
    return brevity_penalty * math.exp(log_precision_sum / max_n)


def char_ngram_counter(text: str, n: int) -> Counter[str]:
    cleaned = " ".join(WORD_RE.findall(normalize_text(text)))
    if not cleaned:
        return Counter()
    if len(cleaned) < n:
        return Counter({cleaned: 1})
    return Counter(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))


def chrf(reference: str, prediction: str, max_n: int = 6, beta: float = 2.0) -> float:
    precision_values: list[float] = []
    recall_values: list[float] = []
    for n in range(1, max_n + 1):
        ref_counts = char_ngram_counter(reference, n)
        pred_counts = char_ngram_counter(prediction, n)
        overlap = sum((ref_counts & pred_counts).values())
        precision_values.append(safe_div(overlap, sum(pred_counts.values())))
        recall_values.append(safe_div(overlap, sum(ref_counts.values())))

    precision = mean(precision_values) if precision_values else 0.0
    recall = mean(recall_values) if recall_values else 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall) if precision + recall else 0.0


def render_text_for_metrics(
    segments: list[Segment],
    *,
    talk_only: bool = False,
    theses_only: bool = False,
) -> str:
    parts: list[str] = []
    for segment in segments:
        if talk_only and is_non_talk_header(segment.header):
            continue
        if not theses_only:
            parts.append(segment.header)
        parts.extend(thesis.text for thesis in segment.theses)
    return "\n".join(parts)


def summarization_text_metrics(reference: str, prediction: str) -> dict:
    return {
        "rouge_1": rouge_n(reference, prediction, 1),
        "rouge_2": rouge_n(reference, prediction, 2),
        "rouge_l": rouge_l(reference, prediction),
        "bleu_1": bleu(reference, prediction, 1),
        "bleu_2": bleu(reference, prediction, 2),
        "bleu_4": bleu(reference, prediction, 4),
        "chrf": chrf(reference, prediction),
        "token_f1": token_f1(reference, prediction),
        "char_3gram_f1": char_ngram_f1(reference, prediction),
    }


def text_metric_scopes(ref_segments: list[Segment], pred_segments: list[Segment]) -> dict:
    scopes = {
        "all_headers_and_theses": (
            render_text_for_metrics(ref_segments),
            render_text_for_metrics(pred_segments),
        ),
        "talk_headers_and_theses": (
            render_text_for_metrics(ref_segments, talk_only=True),
            render_text_for_metrics(pred_segments, talk_only=True),
        ),
        "theses_only": (
            render_text_for_metrics(ref_segments, theses_only=True),
            render_text_for_metrics(pred_segments, theses_only=True),
        ),
    }
    return {
        name: summarization_text_metrics(reference, prediction)
        for name, (reference, prediction) in scopes.items()
    }


def duration(item: Segment | Thesis) -> int:
    return max(0, item.end - item.start)


def overlap_seconds(a: Segment | Thesis, b: Segment | Thesis) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def interval_iou(a: Segment | Thesis, b: Segment | Thesis) -> float:
    inter = overlap_seconds(a, b)
    if inter <= 0:
        return 0.0
    union = duration(a) + duration(b) - inter
    return inter / union if union > 0 else 0.0


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def is_non_talk_header(header: str) -> bool:
    text = normalize_text(header or "")
    return not text.strip() or any(pattern.search(text) for pattern in NON_TALK_HEADER_PATTERNS)


def greedy_match_by_score(
    ref_items: list[Segment] | list[Thesis],
    pred_items: list[Segment] | list[Thesis],
    score_fn,
    min_score: float,
) -> list[Match]:
    candidates: list[tuple[float, int, int]] = []
    for ref_index, ref_item in enumerate(ref_items):
        for pred_index, pred_item in enumerate(pred_items):
            score = score_fn(ref_item, pred_item)
            if score >= min_score:
                candidates.append((score, ref_index, pred_index))
    candidates.sort(reverse=True)

    used_ref: set[int] = set()
    used_pred: set[int] = set()
    matches: list[Match] = []
    for score, ref_index, pred_index in candidates:
        if ref_index in used_ref or pred_index in used_pred:
            continue
        used_ref.add(ref_index)
        used_pred.add(pred_index)
        matches.append(Match(ref_index=ref_index, pred_index=pred_index, score=score))
    matches.sort(key=lambda m: ref_items[m.ref_index].start)
    return matches


def segment_metrics(ref: list[Segment], pred: list[Segment], iou_threshold: float) -> dict:
    matches = greedy_match_by_score(ref, pred, interval_iou, iou_threshold)
    precision = safe_div(len(matches), len(pred))
    recall = safe_div(len(matches), len(ref))
    start_errors = [abs(ref[m.ref_index].start - pred[m.pred_index].start) for m in matches]
    end_errors = [abs(ref[m.ref_index].end - pred[m.pred_index].end) for m in matches]
    ious = [interval_iou(ref[m.ref_index], pred[m.pred_index]) for m in matches]
    return {
        "ref_count": len(ref),
        "pred_count": len(pred),
        "matched": len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mean_iou": mean(ious) if ious else 0.0,
        "mean_start_error_sec": mean(start_errors) if start_errors else None,
        "mean_end_error_sec": mean(end_errors) if end_errors else None,
        "matches": matches,
    }


def merge_intervals(items: list[Segment] | list[Thesis]) -> list[tuple[int, int]]:
    intervals = sorted((item.start, item.end) for item in items if item.end > item.start)
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_total(intervals: list[tuple[int, int]]) -> int:
    return sum(max(0, end - start) for start, end in intervals)


def intersection_total(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    i = j = total = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        total += max(0, end - start)
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def coverage_metrics(ref: list[Segment], pred: list[Segment]) -> dict:
    ref_intervals = merge_intervals(ref)
    pred_intervals = merge_intervals(pred)
    overlap = intersection_total(ref_intervals, pred_intervals)
    ref_total = interval_total(ref_intervals)
    pred_total = interval_total(pred_intervals)
    return {
        "ref_duration_sec": ref_total,
        "pred_duration_sec": pred_total,
        "overlap_sec": overlap,
        "recall_by_time": safe_div(overlap, ref_total),
        "precision_by_time": safe_div(overlap, pred_total),
    }


def flatten_theses(segments: list[Segment]) -> list[Thesis]:
    theses: list[Thesis] = []
    for segment_index, segment in enumerate(segments):
        for thesis in segment.theses:
            theses.append(
                Thesis(
                    start=thesis.start,
                    end=thesis.end,
                    text=thesis.text,
                    parent_index=segment_index,
                )
            )
    return theses


def thesis_pair_score(ref_thesis: Thesis, pred_thesis: Thesis) -> float:
    time_score = interval_iou(ref_thesis, pred_thesis)
    text_score = text_similarity(ref_thesis.text, pred_thesis.text)
    return 0.55 * time_score + 0.45 * text_score


def thesis_metrics(
    ref_segments: list[Segment],
    pred_segments: list[Segment],
    *,
    min_iou: float,
    min_text_similarity: float,
) -> dict:
    ref = flatten_theses(ref_segments)
    pred = flatten_theses(pred_segments)

    def score_fn(ref_thesis: Thesis, pred_thesis: Thesis) -> float:
        if interval_iou(ref_thesis, pred_thesis) < min_iou:
            return 0.0
        if text_similarity(ref_thesis.text, pred_thesis.text) < min_text_similarity:
            return 0.0
        return thesis_pair_score(ref_thesis, pred_thesis)

    matches = greedy_match_by_score(ref, pred, score_fn, 0.0001)
    precision = safe_div(len(matches), len(pred))
    recall = safe_div(len(matches), len(ref))
    time_ious = [interval_iou(ref[m.ref_index], pred[m.pred_index]) for m in matches]
    text_scores = [text_similarity(ref[m.ref_index].text, pred[m.pred_index].text) for m in matches]
    return {
        "ref_count": len(ref),
        "pred_count": len(pred),
        "matched": len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mean_iou": mean(time_ious) if time_ious else 0.0,
        "mean_text_similarity": mean(text_scores) if text_scores else 0.0,
        "matches": matches,
    }


def thesis_text_only_metrics(
    ref_segments: list[Segment],
    pred_segments: list[Segment],
    *,
    min_text_similarity: float,
) -> dict:
    ref = flatten_theses(ref_segments)
    pred = flatten_theses(pred_segments)
    matches = greedy_match_by_score(ref, pred, lambda a, b: text_similarity(a.text, b.text), min_text_similarity)
    precision = safe_div(len(matches), len(pred))
    recall = safe_div(len(matches), len(ref))
    scores = [text_similarity(ref[m.ref_index].text, pred[m.pred_index].text) for m in matches]
    return {
        "ref_count": len(ref),
        "pred_count": len(pred),
        "matched": len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mean_text_similarity": mean(scores) if scores else 0.0,
        "matches": matches,
    }


def header_metrics(ref: list[Segment], pred: list[Segment], matches: list[Match]) -> dict:
    scores = [
        text_similarity(ref[m.ref_index].header, pred[m.pred_index].header)
        for m in matches
    ]
    return {
        "mean_text_similarity": mean(scores) if scores else 0.0,
        "matched_count": len(matches),
    }


def non_talk_violations(pred_segments: list[Segment]) -> dict:
    non_talk = [segment for segment in pred_segments if is_non_talk_header(segment.header)]
    with_theses = [segment for segment in non_talk if segment.theses]
    return {
        "pred_non_talk_segments": len(non_talk),
        "pred_non_talk_with_theses": len(with_theses),
        "violation_rate": safe_div(len(with_theses), len(non_talk)),
    }


def strip_runtime_fields(metrics: dict) -> dict:
    cleaned = {}
    for key, value in metrics.items():
        if key == "matches":
            if value and isinstance(value[0], Match):
                cleaned[key] = [
                    {"ref_index": m.ref_index, "pred_index": m.pred_index, "score": m.score}
                    for m in value
                ]
            else:
                cleaned[key] = value
        elif isinstance(value, dict):
            cleaned[key] = strip_runtime_fields(value)
        elif isinstance(value, list):
            cleaned[key] = [
                strip_runtime_fields(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def print_metric_block(title: str, data: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in data.items():
        if key == "matches":
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif value is None:
            print(f"{key}: -")
        else:
            print(f"{key}: {value}")


def print_text_metric_scopes(data: dict) -> None:
    for scope_name, scope_data in data.items():
        print(f"\nText metrics: {scope_name}")
        print("-" * (14 + len(scope_name)))
        for metric_name, value in scope_data.items():
            if isinstance(value, dict):
                print(
                    f"{metric_name}: "
                    f"P={value['precision']:.4f} "
                    f"R={value['recall']:.4f} "
                    f"F1={value['f1']:.4f}"
                )
            else:
                print(f"{metric_name}: {value:.4f}")


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def seconds_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f} сек"


def print_short_report(metrics: dict, ref_segments: list[Segment], pred_segments: list[Segment]) -> None:
    all_segments = metrics["all_segments"]
    talk_segments = metrics["talk_segments"]
    talk_coverage = metrics["talk_time_coverage"]
    thesis_time = metrics["thesis_time_coverage"]
    theses_text = metrics["summarization_text"]["theses_only"]
    violations = metrics["non_talk_thesis_violations"]

    print("Краткая оценка аннотации")
    print("========================")
    print(f"Сегменты: эталон {len(ref_segments)}, модель {len(pred_segments)}")
    print(
        "Доклады: "
        f"F1={pct(talk_segments['f1'])}, "
        f"Recall={pct(talk_segments['recall'])}, "
        f"Precision={pct(talk_segments['precision'])}"
    )
    print(
        "Покрытие времени докладов: "
        f"Recall={pct(talk_coverage['recall_by_time'])}, "
        f"Precision={pct(talk_coverage['precision_by_time'])}"
    )
    print(
        "Ошибки границ докладов: "
        f"начало {seconds_text(talk_segments['mean_start_error_sec'])}, "
        f"конец {seconds_text(talk_segments['mean_end_error_sec'])}"
    )
    print(
        "Тезисы по времени: "
        f"Recall={pct(thesis_time['recall_by_time'])}, "
        f"Precision={pct(thesis_time['precision_by_time'])}"
    )
    print(
        "Текст тезисов: "
        f"ROUGE-1 F1={pct(theses_text['rouge_1']['f1'])}, "
        f"ROUGE-L F1={pct(theses_text['rouge_l']['f1'])}, "
        f"chrF={pct(theses_text['chrf'])}"
    )
    print(
        "Тезисы у обсуждений/вступлений: "
        f"{violations['pred_non_talk_with_theses']} из {violations['pred_non_talk_segments']}"
    )

    if all_segments["precision"] < 0.6 and talk_segments["recall"] >= 0.8:
        print("Вывод: основные доклады найдены, но есть лишнее дробление/лишние блоки.")
    elif talk_segments["recall"] < 0.7:
        print("Вывод: часть докладов пропущена или сильно не совпала по времени.")
    else:
        print("Вывод: результат выглядит рабочим, дальше стоит смотреть качество тезисов.")


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_details_md(
    path: Path,
    ref_segments: list[Segment],
    pred_segments: list[Segment],
    segment_matches: list[Match],
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matched_ref = {m.ref_index for m in segment_matches}
    matched_pred = {m.pred_index for m in segment_matches}

    lines = [
        "# Отчет по метрикам аннотации",
        "",
        "## Совпавшие сегменты",
        "",
        "| Ref | Pred | IoU | Ошибка начала | Ошибка конца | Сходство темы |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for match in segment_matches:
        ref = ref_segments[match.ref_index]
        pred = pred_segments[match.pred_index]
        header_score = text_similarity(ref.header, pred.header)
        lines.append(
            "| "
            f"{fmt_time(ref.start)} - {fmt_time(ref.end)}: {escape_md(ref.header)} | "
            f"{fmt_time(pred.start)} - {fmt_time(pred.end)}: {escape_md(pred.header)} | "
            f"{interval_iou(ref, pred):.3f} | "
            f"{abs(ref.start - pred.start)} сек | "
            f"{abs(ref.end - pred.end)} сек | "
            f"{header_score:.3f} |"
        )

    lines.extend(["", "## Пропущенные эталонные сегменты", ""])
    for index, segment in enumerate(ref_segments):
        if index not in matched_ref:
            lines.append(f"- {fmt_time(segment.start)} - {fmt_time(segment.end)}: {segment.header}")

    lines.extend(["", "## Лишние предсказанные сегменты", ""])
    for index, segment in enumerate(pred_segments):
        if index not in matched_pred:
            lines.append(f"- {fmt_time(segment.start)} - {fmt_time(segment.end)}: {segment.header}")

    lines.extend(["", "## Ключевые метрики", "", "```json"])
    lines.append(json.dumps(strip_runtime_fields(metrics), ensure_ascii=False, indent=2))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    reference_path: Path,
    prediction_path: Path,
    *,
    segment_iou_threshold: float,
    thesis_iou_threshold: float,
    thesis_text_threshold: float,
) -> dict:
    ref_segments = parse_annotation(reference_path)
    pred_segments = parse_annotation(prediction_path)

    ref_talk = [segment for segment in ref_segments if not is_non_talk_header(segment.header)]
    pred_talk = [segment for segment in pred_segments if not is_non_talk_header(segment.header)]

    all_segments = segment_metrics(ref_segments, pred_segments, segment_iou_threshold)
    talk_segments = segment_metrics(ref_talk, pred_talk, segment_iou_threshold)

    metrics = {
        "files": {
            "reference": str(reference_path),
            "prediction": str(prediction_path),
        },
        "all_segments": all_segments,
        "talk_segments": talk_segments,
        "all_time_coverage": coverage_metrics(ref_segments, pred_segments),
        "talk_time_coverage": coverage_metrics(ref_talk, pred_talk),
        "headers": header_metrics(ref_segments, pred_segments, all_segments["matches"]),
        "theses": thesis_metrics(
            ref_segments,
            pred_segments,
            min_iou=thesis_iou_threshold,
            min_text_similarity=thesis_text_threshold,
        ),
        "theses_time_only": segment_metrics(
            flatten_theses(ref_segments),
            flatten_theses(pred_segments),
            thesis_iou_threshold,
        ),
        "theses_text_only": thesis_text_only_metrics(
            ref_segments,
            pred_segments,
            min_text_similarity=thesis_text_threshold,
        ),
        "thesis_time_coverage": coverage_metrics(
            flatten_theses(ref_segments),
            flatten_theses(pred_segments),
        ),
        "summarization_text": text_metric_scopes(ref_segments, pred_segments),
        "non_talk_thesis_violations": non_talk_violations(pred_segments),
    }
    metrics["_parsed"] = {
        "reference_segments": ref_segments,
        "prediction_segments": pred_segments,
    }
    return metrics


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Сравнить эталонную и сгенерированную аннотацию по таймкодам, темам и тезисам."
    )
    parser.add_argument("--reference", required=True, type=Path, help="Эталонная аннотация .txt/.md")
    parser.add_argument("--prediction", required=True, type=Path, help="Аннотация, полученная моделью")
    parser.add_argument("--out_json", default=None, type=Path, help="Куда сохранить метрики в JSON")
    parser.add_argument("--details_md", default=None, type=Path, help="Куда сохранить подробный markdown-отчет")
    parser.add_argument("--segment_iou_threshold", default=0.30, type=float, help="Порог IoU для совпадения сегментов")
    parser.add_argument("--thesis_iou_threshold", default=0.10, type=float, help="Минимальный IoU для совпадения тезисов")
    parser.add_argument("--thesis_text_threshold", default=0.20, type=float, help="Минимальное текстовое сходство тезисов")
    parser.add_argument("--verbose", action="store_true", help="Показать полный подробный вывод всех метрик")
    args = parser.parse_args()

    metrics = evaluate(
        args.reference,
        args.prediction,
        segment_iou_threshold=args.segment_iou_threshold,
        thesis_iou_threshold=args.thesis_iou_threshold,
        thesis_text_threshold=args.thesis_text_threshold,
    )
    ref_segments = metrics["_parsed"]["reference_segments"]
    pred_segments = metrics["_parsed"]["prediction_segments"]

    if args.verbose:
        print(f"Reference segments: {len(ref_segments)}")
        print(f"Prediction segments: {len(pred_segments)}")
        print_metric_block("All segments", metrics["all_segments"])
        print_metric_block("Talk segments", metrics["talk_segments"])
        print_metric_block("All time coverage", metrics["all_time_coverage"])
        print_metric_block("Talk time coverage", metrics["talk_time_coverage"])
        print_metric_block("Headers", metrics["headers"])
        print_metric_block("Theses strict time+text", metrics["theses"])
        print_metric_block("Theses time only", metrics["theses_time_only"])
        print_metric_block("Theses text only", metrics["theses_text_only"])
        print_metric_block("Thesis time coverage", metrics["thesis_time_coverage"])
        print_text_metric_scopes(metrics["summarization_text"])
        print_metric_block("Non-talk thesis violations", metrics["non_talk_thesis_violations"])
    else:
        print_short_report(metrics, ref_segments, pred_segments)

    serializable = strip_runtime_fields({k: v for k, v in metrics.items() if k != "_parsed"})
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON saved: {args.out_json}")
    if args.details_md:
        write_details_md(
            args.details_md,
            ref_segments,
            pred_segments,
            metrics["all_segments"]["matches"],
            serializable,
        )
        print(f"Details saved: {args.details_md}")


if __name__ == "__main__":
    main()
