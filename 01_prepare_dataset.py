"""
Подготовка датасета для дообучения T5 на ИЕРАРХИЧЕСКОЙ суммаризации
транскриптов круглых столов/дискуссионных панелей.

Формат эталонного саммари (raw/summaries/<имя>.txt):
    MM:SS - MM:SS: Заголовок сегмента 1
    [пустая строка]
    Важные моменты темы
    MM:SS - MM:SS: Тезис 1.
    MM:SS - MM:SS: Тезис 2.
    ...
    [пустая строка]
    MM:SS - MM:SS: Заголовок сегмента 2
    ...

К каждому такому файлу должен быть парный транскрипт в raw/transcripts/<имя>.txt
с реплик-таймкодами в формате:
    [12.34s -> 56.78s] SPEAKER_XX: текст реплики

На выходе формируется dataset.jsonl с двумя типами примеров (поле "task"):
  - task="header": вход = окно транскрипта по диапазону сегмента,
                   target = "MM:SS - MM:SS: <заголовок>"
  - task="thesis": вход = узкое окно транскрипта вокруг диапазона тезиса
                   (с расширением ±context_sec для контекста),
                   target = "MM:SS - MM:SS: <текст тезиса>"

Тайм-коды эталона могут быть как MM:SS, так и H:MM:SS (для длинных встреч).

Использование:
    python 01_prepare_dataset.py \\
        --transcripts ./raw/transcripts \\
        --summaries   ./raw/summaries  \\
        --out         ./data/dataset.jsonl \\
        --tokenizer   utrobinmv/t5_summary_en_ru_zh_large_2048 \\
        --max_input_tokens 1500 \\
        --context_sec 10
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ----------- регэкспы -----------
RE_TURN = re.compile(
    r"\[(?P<start>\d+\.\d+)s\s*->\s*(?P<end>\d+\.\d+)s\]\s+(?P<spk>SPEAKER_\d+):\s*(?P<text>.+)"
)

# тайм-код вида "MM:SS" или "H:MM:SS" в начале строки эталона
# поддерживает: 0:04, 12:34, 1:07:28
RE_TC = r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})"
RE_RANGE_LINE = re.compile(
    rf"^\s*{RE_TC.replace('<h>', '<h1>').replace('<m>', '<m1>').replace('<s>', '<s1>')}"
    rf"\s*[-\u2013\u2014]\s*"
    rf"{RE_TC.replace('<h>', '<h2>').replace('<m>', '<m2>').replace('<s>', '<s2>')}"
    rf"\s*:\s*(?P<text>.+?)\s*$"
)
RE_VAJNYE = re.compile(r"^\s*Важные\s+моменты\s+темы\s*$", re.IGNORECASE)


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class Thesis:
    start: int   # секунды
    end: int
    text: str


@dataclass
class Segment:
    start: int   # секунды
    end: int
    header: str
    theses: list[Thesis] = field(default_factory=list)


# ----------- парсинг транскрипта -----------
def parse_transcript(path: Path) -> list[Turn]:
    turns: list[Turn] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = RE_TURN.match(line.strip())
        if not m:
            continue
        turns.append(
            Turn(
                start=float(m["start"]),
                end=float(m["end"]),
                speaker=m["spk"],
                text=m["text"].strip(),
            )
        )
    return turns


# ----------- парсинг эталона -----------
def _tc_to_sec(h: str | None, m: str, s: str) -> int:
    hh = int(h) if h else 0
    return hh * 3600 + int(m) * 60 + int(s)


def fmt_tc(seconds: float) -> str:
    """Форматирует MM:SS или H:MM:SS (как в эталоне)."""
    sec = int(round(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_summary(path: Path) -> list[Segment]:
    """
    Возвращает список сегментов с заголовками и тезисами.
    Логика:
      1) пробегаем строки сверху вниз;
      2) каждая строка вида "MM:SS - MM:SS: ..." — это либо заголовок сегмента
         (если перед ней нет блока "Важные моменты темы" — точнее, если она
         идёт ПЕРВОЙ после маркера или в самом начале), либо тезис (если она
         появилась после "Важные моменты темы" в текущем сегменте);
      3) маркер "Важные моменты темы" переключает режим парсинга на "тезисы".
    """
    segments: list[Segment] = []
    cur: Segment | None = None
    in_theses_block = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if RE_VAJNYE.match(line):
            in_theses_block = True
            continue
        m = RE_RANGE_LINE.match(line)
        if not m:
            # неизвестная строка — пропускаем
            continue
        s_sec = _tc_to_sec(m.group("h1"), m.group("m1"), m.group("s1"))
        e_sec = _tc_to_sec(m.group("h2"), m.group("m2"), m.group("s2"))
        text = m.group("text").strip()

        if not in_theses_block:
            # это заголовок нового сегмента
            if cur is not None:
                segments.append(cur)
            cur = Segment(start=s_sec, end=e_sec, header=text, theses=[])
        else:
            # это тезис в текущем сегменте
            if cur is None:
                # бывает: тезис без открывающего заголовка — создаём «безымянный»
                cur = Segment(start=s_sec, end=e_sec, header="(без заголовка)", theses=[])
            cur.theses.append(Thesis(start=s_sec, end=e_sec, text=text))

        # пустая строка между блоком тезисов и следующим заголовком сбрасывает режим
        # (но т.к. мы пустые строки выше пропускаем, используем сигнал:
        # после маркера "Важные моменты темы" следующая строка БЕЗ "Важные..."
        # с большим временным разрывом — это уже следующий заголовок).
        # Простое правило: возвращаемся в режим "ожидаем заголовок" сразу после
        # пустой строки. Реализуем через флаг при чтении пустых строк ниже.

    if cur is not None:
        segments.append(cur)

    # Постобработка: если в эталоне «Важные моменты темы» переключает в режим тезисов,
    # а потом приходит большой временной скачок назад (новый сегмент начинается раньше
    # последнего тезиса) — это уже не тезис, а новый заголовок. Скорректируем такой случай:
    # пройдёмся ещё раз и разделим сегменты, где после тезисов идёт строка с T_start <
    # start первого тезиса или >> много больше end сегмента.
    return _split_misclassified(segments)


def _split_misclassified(segments: list[Segment]) -> list[Segment]:
    """
    Эвристика: если тезис в сегменте имеет диапазон [a,b], а b > segment.end + 30c
    или a > segment.end + 30c — он, скорее всего, был ошибочно отнесён к этому
    сегменту (это новый заголовок без явного маркера). Откалываем его и всё, что
    после, в новый сегмент.
    """
    out: list[Segment] = []
    for seg in segments:
        # ищем точку «излома»
        cut_idx = None
        for i, th in enumerate(seg.theses):
            if th.start > seg.end + 30 or th.end > seg.end + 600:
                cut_idx = i
                break
        if cut_idx is None:
            out.append(seg)
            continue
        # оставляем «нормальные» тезисы в seg, остальное переразбираем
        keep = seg.theses[:cut_idx]
        rest = seg.theses[cut_idx:]
        out.append(Segment(start=seg.start, end=seg.end, header=seg.header, theses=keep))
        # каждый "оторванный" тезис трактуем как новый заголовок (без своих тезисов)
        for th in rest:
            out.append(Segment(start=th.start, end=th.end, header=th.text, theses=[]))
    return out


# ----------- лучшее «строгое» правило: переключаем режим тезисов через пустые строки -----------
# (более надёжная альтернатива parse_summary выше — используем её на самом деле)
def parse_summary_v2(path: Path) -> list[Segment]:
    """
    Более надёжный парсер: пустая строка ВНЕ блока тезисов — разделитель блоков.
    Поток: «строка с диапазоном» (заголовок) → пустая строка → «Важные моменты темы»
    → подряд строки тезисов → пустая строка → следующий заголовок.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    segments: list[Segment] = []
    cur: Segment | None = None
    in_theses = False

    for raw in lines:
        line = raw.rstrip()
        if line.strip() == "":
            # пустая строка между «Важные моменты темы»-блоком и следующим заголовком
            in_theses = False
            continue
        if RE_VAJNYE.match(line.strip()):
            in_theses = True
            continue
        m = RE_RANGE_LINE.match(line.strip())
        if not m:
            continue
        s_sec = _tc_to_sec(m.group("h1"), m.group("m1"), m.group("s1"))
        e_sec = _tc_to_sec(m.group("h2"), m.group("m2"), m.group("s2"))
        text = m.group("text").strip()

        if not in_theses:
            if cur is not None:
                segments.append(cur)
            cur = Segment(start=s_sec, end=e_sec, header=text, theses=[])
        else:
            if cur is None:
                cur = Segment(start=s_sec, end=e_sec, header="(без заголовка)", theses=[])
            cur.theses.append(Thesis(start=s_sec, end=e_sec, text=text))

    if cur is not None:
        segments.append(cur)
    return segments


# ----------- построение окна транскрипта по тайм-кодам -----------
def collect_turns_in_window(
    turns: list[Turn], t_start: float, t_end: float, slack: float = 5.0
) -> list[Turn]:
    """Все реплики, пересекающиеся с [t_start - slack, t_end + slack]."""
    a, b = t_start - slack, t_end + slack
    return [t for t in turns if t.end >= a and t.start <= b]


def render_window(turns_subset: list[Turn]) -> str:
    """Реплики в текстовом виде (для подачи в модель)."""
    lines = []
    for t in turns_subset:
        lines.append(f"[{fmt_tc(t.start)}-{fmt_tc(t.end)}] {t.speaker}: {t.text}")
    return "\n".join(lines)


# ----------- построение входа/таргета -----------
def build_header_example(seg: Segment, turns: list[Turn]) -> dict | None:
    subset = collect_turns_in_window(turns, seg.start, seg.end, slack=5.0)
    if not subset:
        return None
    window = render_window(subset)
    inp = (
        "Задача: определи тематический заголовок сегмента дискуссии. "
        "Верни ровно одну строку формата 'MM:SS - MM:SS: <заголовок>' "
        "с указанным временным диапазоном. Без вводных фраз.\n\n"
        f"Диапазон: {fmt_tc(seg.start)} - {fmt_tc(seg.end)}\n\n"
        "Фрагмент транскрипта:\n"
        f"{window}"
    )
    tgt = f"{fmt_tc(seg.start)} - {fmt_tc(seg.end)}: {seg.header}"
    return {"task": "header", "input": inp, "target": tgt,
            "time_range": [seg.start, seg.end]}


def build_thesis_example(
    th: Thesis, seg: Segment, turns: list[Turn], context_sec: float
) -> dict | None:
    # узкое окно вокруг тезиса + расширение
    subset = collect_turns_in_window(turns, th.start, th.end, slack=context_sec)
    if not subset:
        return None
    window = render_window(subset)
    inp = (
        "Задача: сформулируй один ключевой тезис, прозвучавший во фрагменте. "
        "Верни ровно одну строку формата 'MM:SS - MM:SS: <тезис>.' с указанным "
        "временным диапазоном. Тезис должен быть содержательным (что было сказано), "
        "без мета-речи и без вводных слов.\n\n"
        f"Заголовок сегмента: {seg.header}\n"
        f"Диапазон тезиса: {fmt_tc(th.start)} - {fmt_tc(th.end)}\n\n"
        "Фрагмент транскрипта:\n"
        f"{window}"
    )
    tgt = f"{fmt_tc(th.start)} - {fmt_tc(th.end)}: {th.text}"
    return {"task": "thesis", "input": inp, "target": tgt,
            "time_range": [th.start, th.end]}


# ----------- обработка пары транскрипт + эталон -----------
def process_pair(transcript_path: Path, summary_path: Path,
                 tokenizer, max_input_tokens: int, context_sec: float,
                 stats: dict | None = None) -> list[dict]:
    turns = parse_transcript(transcript_path)
    segments = parse_summary_v2(summary_path)

    examples: list[dict] = []
    for seg in segments:
        ex = build_header_example(seg, turns)
        if ex is not None:
            orig_tok = _count_tokens(ex["input"], tokenizer)
            _maybe_truncate_window_in_input(ex, tokenizer, max_input_tokens)
            new_tok = _count_tokens(ex["input"], tokenizer)
            if stats is not None and new_tok < orig_tok:
                stats["headers_compressed"] = stats.get("headers_compressed", 0) + 1
                stats["headers_orig_tokens_total"] = stats.get("headers_orig_tokens_total", 0) + orig_tok
                stats["headers_new_tokens_total"]  = stats.get("headers_new_tokens_total", 0) + new_tok
            if stats is not None:
                stats["headers_total"] = stats.get("headers_total", 0) + 1
            examples.append({**ex, "src_file": transcript_path.name})
        for th in seg.theses:
            ex = build_thesis_example(th, seg, turns, context_sec=context_sec)
            if ex is not None:
                orig_tok = _count_tokens(ex["input"], tokenizer)
                _maybe_truncate_window_in_input(ex, tokenizer, max_input_tokens)
                new_tok = _count_tokens(ex["input"], tokenizer)
                if stats is not None and new_tok < orig_tok:
                    stats["theses_compressed"] = stats.get("theses_compressed", 0) + 1
                if stats is not None:
                    stats["theses_total"] = stats.get("theses_total", 0) + 1
                examples.append({**ex, "src_file": transcript_path.name})
    return examples


def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _smart_truncate_window_lines(
    head: str, body_lines: list[str], tokenizer, max_tokens: int
) -> str:
    """
    Сжимает окно транскрипта так, чтобы итоговый промпт уложился в max_tokens,
    НЕ теряя информацию о конце сегмента.

    Стратегия (repr. sampling):
      1) Если всё влезает — возвращаем как есть.
      2) Если не влезает — формируем три блока:
           - HEAD-блок: первые N_head реплик ("что обсуждается");
           - MID-блок: равномерно сэмплированные реплики из середины;
           - TAIL-блок: последние N_tail реплик ("к чему пришли");
         Пропуски в середине заменяются маркером:
           [...пропущено K реплик...]
      3) Подбираем размеры HEAD/MID/TAIL бинарным поиском по общему числу строк,
         удерживая пропорцию 30% / 50% / 20%.
    """
    full = head + "Фрагмент транскрипта:\n" + "\n".join(body_lines)
    if _count_tokens(full, tokenizer) <= max_tokens:
        return full

    n = len(body_lines)
    if n <= 2:
        # вырожденный случай: всего пара строк — оставляем как есть
        return full

    def build(k: int) -> str:
        """Собирает версию с k реплик в HEAD+MID+TAIL."""
        if k >= n:
            return head + "Фрагмент транскрипта:\n" + "\n".join(body_lines)
        k = max(3, k)  # хотя бы 1+1+1
        n_head = max(1, int(k * 0.30))
        n_tail = max(1, int(k * 0.20))
        n_mid  = max(1, k - n_head - n_tail)

        head_lines = body_lines[:n_head]
        tail_lines = body_lines[-n_tail:] if n_tail > 0 else []

        mid_pool_start = n_head
        mid_pool_end   = n - n_tail
        mid_pool       = body_lines[mid_pool_start:mid_pool_end]
        if len(mid_pool) <= n_mid:
            mid_lines = mid_pool
            skipped_before_mid = 0
            skipped_after_mid  = 0
            mid_with_gaps: list[str] = list(mid_lines)
        else:
            # равномерно выбираем n_mid реплик из mid_pool
            step = len(mid_pool) / n_mid
            picks_idx = sorted({int(i * step) for i in range(n_mid)})
            picks_idx = [i for i in picks_idx if 0 <= i < len(mid_pool)][:n_mid]
            # вставим маркеры пропусков между подряд идущими «дырами»
            mid_with_gaps = []
            prev = -1
            for idx in picks_idx:
                gap = idx - prev - 1
                if prev == -1 and idx > 0:
                    mid_with_gaps.append(f"[...пропущено {idx} реплик...]")
                elif gap > 0:
                    mid_with_gaps.append(f"[...пропущено {gap} реплик...]")
                mid_with_gaps.append(mid_pool[idx])
                prev = idx
            if prev < len(mid_pool) - 1:
                gap = len(mid_pool) - 1 - prev
                mid_with_gaps.append(f"[...пропущено {gap} реплик...]")

        # склейка с маркерами на стыках HEAD↔MID и MID↔TAIL
        result_lines: list[str] = []
        result_lines.extend(head_lines)
        if mid_pool_start < mid_pool_end:
            # между HEAD и началом MID-пула — пропусков нет (граница совпадает),
            # но если в MID-пуле что-то выкинули в начале, маркер уже внутри mid_with_gaps
            result_lines.extend(mid_with_gaps)
            # между MID и TAIL — если в MID не дошли до конца пула, маркер тоже уже там
        else:
            # MID пустой — всё, что между HEAD и TAIL, пропущено
            n_gap = mid_pool_end - mid_pool_start
            if n_gap > 0:
                result_lines.append(f"[...пропущено {n_gap} реплик...]")
        result_lines.extend(tail_lines)
        return head + "Фрагмент транскрипта:\n" + "\n".join(result_lines)

    # бинарный поиск максимального k, при котором промпт влезает в max_tokens
    lo, hi = 3, n
    best_text = build(3)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = build(mid)
        if _count_tokens(candidate, tokenizer) <= max_tokens:
            best_text = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best_text


def _maybe_truncate_window_in_input(ex: dict, tokenizer, max_tokens: int) -> None:
    """
    Сжимает вход до max_tokens. Для коротких окон ничего не делает.
    Для длинных — применяет умный семплинг (HEAD + MID + TAIL с маркерами пропуска),
    чтобы модель видела начало, середину и конец сегмента, а не только начало.
    """
    txt = ex["input"]
    if _count_tokens(txt, tokenizer) <= max_tokens:
        return
    # отделяем «промпт-заголовок» от тела транскрипта
    parts = txt.split("Фрагмент транскрипта:\n", 1)
    if len(parts) != 2:
        # fallback: посимвольная обрезка
        ex["input"] = tokenizer.decode(
            tokenizer(txt, add_special_tokens=False)["input_ids"][:max_tokens]
        )
        return
    head, body = parts
    body_lines = body.splitlines()
    ex["input"] = _smart_truncate_window_lines(head, body_lines, tokenizer, max_tokens)


# ----------- entrypoint -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True, type=Path)
    ap.add_argument("--summaries", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokenizer", default="utrobinmv/t5_summary_en_ru_zh_large_2048",
                    help="HF id или путь к токенизатору для подсчёта длины входа")
    ap.add_argument("--max_input_tokens", type=int, default=1900,
                    help="Максимум токенов на вход (потолок модели — 2048, поэтому <=1950). "
                         "Длинные окна сжимаются репрезентативным семплингом, а не обрезкой хвоста.")
    ap.add_argument("--context_sec", type=float, default=10.0,
                    help="Расширение окна для тезиса в секундах (±)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_examples: list[dict] = []
    transcripts_dir = args.transcripts
    summaries_dir = args.summaries
    stats: dict = {}
    for sm in sorted(summaries_dir.glob("*.txt")):
        tr = transcripts_dir / sm.name
        if not tr.exists():
            print(f"[skip] нет транскрипта для эталона {sm.name}", flush=True)
            continue
        pairs = process_pair(tr, sm, tokenizer, args.max_input_tokens, args.context_sec, stats=stats)
        all_examples.extend(pairs)
        n_h = sum(1 for p in pairs if p["task"] == "header")
        n_t = sum(1 for p in pairs if p["task"] == "thesis")
        print(f"[ok] {sm.name}: headers={n_h} theses={n_t}", flush=True)

    with args.out.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_h = sum(1 for p in all_examples if p["task"] == "header")
    n_t = sum(1 for p in all_examples if p["task"] == "thesis")
    print(f"\nСохранено {len(all_examples)} примеров в {args.out}")
    print(f"  headers: {n_h}")
    print(f"  theses : {n_t}")
    if all_examples:
        lens_in = [len(p["input"]) for p in all_examples]
        lens_out = [len(p["target"]) for p in all_examples]
        print(f"  input chars  : avg={sum(lens_in)//len(lens_in)}  max={max(lens_in)}")
        print(f"  target chars : avg={sum(lens_out)//len(lens_out)}  max={max(lens_out)}")
        # статистика сжатия
        h_total = stats.get("headers_total", 0)
        h_comp  = stats.get("headers_compressed", 0)
        t_total = stats.get("theses_total", 0)
        t_comp  = stats.get("theses_compressed", 0)
        pct_h   = (h_comp / h_total * 100) if h_total else 0.0
        pct_t   = (t_comp / t_total * 100) if t_total else 0.0
        print(f"  headers compressed (smart-sampling): {h_comp}/{h_total} ({pct_h:.1f}%)")
        print(f"  theses  compressed (smart-sampling): {t_comp}/{t_total} ({pct_t:.1f}%)")
        if h_comp > 0:
            orig_avg = stats.get("headers_orig_tokens_total", 0) / h_comp
            new_avg  = stats.get("headers_new_tokens_total", 0) / h_comp
            print(f"  headers avg tokens before/after: {orig_avg:.0f} -> {new_avg:.0f} "
                  f"({(1 - new_avg/orig_avg)*100:.1f}% сжатия)")


if __name__ == "__main__":
    main()
