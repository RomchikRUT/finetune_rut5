from __future__ import annotations

import argparse
import importlib
import json
import random
import re
from pathlib import Path


prep = importlib.import_module("01_prepare_dataset")

LABELS = ["talk_start", "talk_end", "talk_inside", "non_talk"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

NON_TALK_PATTERNS = (
    re.compile(r"\bвопрос", re.IGNORECASE),
    re.compile(r"\bответ", re.IGNORECASE),
    re.compile(r"\bобсуждени[ея]\s+(доклада|выступления)", re.IGNORECASE),
    re.compile(r"\bвступительн", re.IGNORECASE),
    re.compile(r"\bоткрыти[ея]\b", re.IGNORECASE),
    re.compile(r"\bзаключительн", re.IGNORECASE),
    re.compile(r"\bзавершени[ея]\b", re.IGNORECASE),
    re.compile(r"\bподведени[ея]\s+итог", re.IGNORECASE),
    re.compile(r"\bперерыв\b", re.IGNORECASE),
    re.compile(r"\bрегламент\b", re.IGNORECASE),
    re.compile(r"\bприветств", re.IGNORECASE),
)


def is_talk_segment(header: str, min_duration_sec: float, duration_sec: float) -> bool:
    text = (header or "").strip()
    if duration_sec < min_duration_sec:
        return False
    if not text:
        return False
    if any(pattern.search(text) for pattern in NON_TALK_PATTERNS):
        return False
    return True


def fmt_tc(seconds: float) -> str:
    sec = int(round(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def render_boundary_context(turns: list, center: float, context_sec: float) -> str:
    left = center - context_sec
    right = center + context_sec
    subset = [t for t in turns if t.end >= left and t.start <= right]
    lines: list[str] = [f"CENTER: {fmt_tc(center)}"]
    marker_added = False
    for turn in subset:
        if not marker_added and turn.start >= center:
            lines.append(f"<<<BOUNDARY {fmt_tc(center)}>>>")
            marker_added = True
        lines.append(f"[{fmt_tc(turn.start)}-{fmt_tc(turn.end)}] {turn.text}")
    if not marker_added:
        lines.append(f"<<<BOUNDARY {fmt_tc(center)}>>>")
    return "\n".join(lines)


def add_example(
    records: list[dict],
    turns: list,
    *,
    source: str,
    label: str,
    center: float,
    context_sec: float,
    segment_start: float | None = None,
    segment_end: float | None = None,
    summary_title: str | None = None,
) -> None:
    if not turns:
        return
    duration_end = turns[-1].end
    center = max(turns[0].start, min(center, duration_end))
    text = render_boundary_context(turns, center, context_sec)
    if len(text.strip()) < 20:
        return
    records.append(
        {
            "text": text,
            "label": label,
            "label_id": LABEL2ID[label],
            "labels": LABELS,
            "center": round(center, 2),
            "source": source,
            "segment_start": None if segment_start is None else round(float(segment_start), 2),
            "segment_end": None if segment_end is None else round(float(segment_end), 2),
            "summary_title": summary_title or "",
        }
    )


def normalize_ranges(ranges: list[tuple[float, float]], max_end: float) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        start = max(0.0, min(float(start), max_end))
        end = max(0.0, min(float(end), max_end))
        if end <= start:
            continue
        if cleaned and start <= cleaned[-1][1] + 1.0:
            cleaned[-1] = (cleaned[-1][0], max(cleaned[-1][1], end))
        else:
            cleaned.append((start, end))
    return cleaned


def complement_ranges(talk_ranges: list[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = start
    for talk_start, talk_end in talk_ranges:
        if talk_start > cursor:
            gaps.append((cursor, talk_start))
        cursor = max(cursor, talk_end)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def sample_inside_centers(start: float, end: float, margin: float, max_per_segment: int) -> list[float]:
    left = start + margin
    right = end - margin
    if right <= left:
        return []
    if max_per_segment <= 1:
        return [(left + right) / 2.0]
    step = (right - left) / float(max_per_segment + 1)
    return [left + step * (i + 1) for i in range(max_per_segment)]


def build_records_for_pair(
    transcript_path: Path,
    summary_path: Path,
    *,
    context_sec: float,
    boundary_jitter_sec: float,
    inside_margin_sec: float,
    max_inside_per_segment: int,
    non_talk_stride_sec: float,
    min_talk_sec: float,
    max_non_talk_ratio: float,
    seed: int,
) -> list[dict]:
    turns = prep.parse_transcript(transcript_path)
    if not turns:
        return []
    segments = prep.parse_summary_v2(summary_path)
    if not segments:
        return []

    max_end = turns[-1].end
    talk_ranges_with_titles: list[tuple[float, float, str]] = []
    for seg in segments:
        duration = float(seg.end - seg.start)
        if is_talk_segment(seg.header, min_talk_sec, duration):
            talk_ranges_with_titles.append((float(seg.start), float(seg.end), seg.header))

    talk_ranges = normalize_ranges([(s, e) for s, e, _ in talk_ranges_with_titles], max_end)
    if not talk_ranges:
        return []

    records: list[dict] = []
    source = transcript_path.name
    jitter_values = [-boundary_jitter_sec, 0.0, boundary_jitter_sec]

    for seg_start, seg_end, title in talk_ranges_with_titles:
        seg_start = max(0.0, min(seg_start, max_end))
        seg_end = max(0.0, min(seg_end, max_end))
        if seg_end <= seg_start:
            continue

        for jitter in jitter_values:
            add_example(
                records,
                turns,
                source=source,
                label="talk_start",
                center=seg_start + jitter,
                context_sec=context_sec,
                segment_start=seg_start,
                segment_end=seg_end,
                summary_title=title,
            )
            add_example(
                records,
                turns,
                source=source,
                label="talk_end",
                center=seg_end + jitter,
                context_sec=context_sec,
                segment_start=seg_start,
                segment_end=seg_end,
                summary_title=title,
            )

        for center in sample_inside_centers(
            seg_start, seg_end, inside_margin_sec, max_inside_per_segment
        ):
            add_example(
                records,
                turns,
                source=source,
                label="talk_inside",
                center=center,
                context_sec=context_sec,
                segment_start=seg_start,
                segment_end=seg_end,
                summary_title=title,
            )

    rng = random.Random(seed + sum(ord(ch) for ch in source))
    non_talk_candidates: list[float] = []
    for gap_start, gap_end in complement_ranges(talk_ranges, turns[0].start, max_end):
        center = gap_start + non_talk_stride_sec / 2.0
        while center < gap_end:
            too_close_to_boundary = any(
                abs(center - s) < inside_margin_sec or abs(center - e) < inside_margin_sec
                for s, e in talk_ranges
            )
            if not too_close_to_boundary:
                non_talk_candidates.append(center)
            center += non_talk_stride_sec

    rng.shuffle(non_talk_candidates)
    talk_like_count = sum(1 for rec in records if rec["label"] != "non_talk")
    non_talk_limit = int(max(1, talk_like_count * max_non_talk_ratio))
    for center in non_talk_candidates[:non_talk_limit]:
        add_example(
            records,
            turns,
            source=source,
            label="non_talk",
            center=center,
            context_sec=context_sec,
        )

    return records


def find_pairs(transcripts_dir: Path, summaries_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    transcript_by_stem = {path.stem: path for path in transcripts_dir.glob("*.txt")}
    for summary in sorted(summaries_dir.glob("*")):
        if summary.suffix.lower() not in {".txt", ".md"}:
            continue
        transcript = transcript_by_stem.get(summary.stem)
        if transcript is not None:
            pairs.append((transcript, summary))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True, type=Path)
    ap.add_argument("--summaries", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--context_sec", type=float, default=90.0)
    ap.add_argument("--boundary_jitter_sec", type=float, default=15.0)
    ap.add_argument("--inside_margin_sec", type=float, default=90.0)
    ap.add_argument("--max_inside_per_segment", type=int, default=4)
    ap.add_argument("--non_talk_stride_sec", type=float, default=45.0)
    ap.add_argument("--min_talk_sec", type=float, default=120.0)
    ap.add_argument("--max_non_talk_ratio", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = find_pairs(args.transcripts, args.summaries)
    if not pairs:
        raise SystemExit("No transcript/summary pairs found.")

    all_records: list[dict] = []
    per_source: dict[str, dict[str, int]] = {}
    for transcript, summary in pairs:
        records = build_records_for_pair(
            transcript,
            summary,
            context_sec=args.context_sec,
            boundary_jitter_sec=args.boundary_jitter_sec,
            inside_margin_sec=args.inside_margin_sec,
            max_inside_per_segment=args.max_inside_per_segment,
            non_talk_stride_sec=args.non_talk_stride_sec,
            min_talk_sec=args.min_talk_sec,
            max_non_talk_ratio=args.max_non_talk_ratio,
            seed=args.seed,
        )
        if not records:
            continue
        counts = {label: 0 for label in LABELS}
        for rec in records:
            counts[rec["label"]] += 1
        per_source[transcript.name] = counts
        all_records.extend(records)

    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_counts = {label: 0 for label in LABELS}
    for rec in all_records:
        total_counts[rec["label"]] += 1

    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "label2id": LABEL2ID,
                "model_recommendation": "cointegrated/rubert-tiny2",
                "records": len(all_records),
                "counts": total_counts,
                "sources": per_source,
                "params": {
                    "context_sec": args.context_sec,
                    "boundary_jitter_sec": args.boundary_jitter_sec,
                    "inside_margin_sec": args.inside_margin_sec,
                    "max_inside_per_segment": args.max_inside_per_segment,
                    "non_talk_stride_sec": args.non_talk_stride_sec,
                    "min_talk_sec": args.min_talk_sec,
                    "max_non_talk_ratio": args.max_non_talk_ratio,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f">>> pairs={len(pairs)} sources_with_records={len(per_source)}")
    print(f">>> records={len(all_records)} counts={total_counts}")
    print(f">>> saved dataset: {args.out}")
    print(f">>> saved meta: {meta_path}")


if __name__ == "__main__":
    main()
