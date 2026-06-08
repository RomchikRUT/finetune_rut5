"""
Иерархический инференс: дискуссия -> Markdown в формате эталона.

Алгоритм:
  1) Транскрипт парсится в реплики (Turn).
  2) Скользящим окном фиксированной длины (--window_sec, --stride_sec) проходим
     по всей дискуссии. Для каждого окна адаптером 'header' получаем строку
     "MM:SS - MM:SS: <заголовок>".
  3) Соседние окна со схожими заголовками склеиваются в единый сегмент
     (расширяем границы).
  4) Внутри каждого финального сегмента делаем подсегменты по сменам спикера
     (с ограничениями на длительность) и для каждого адаптером 'thesis'
     генерируем строку "MM:SS - MM:SS: <тезис>.".
  5) Дедуп тезисов (Jaccard >= 0.7), сборка финального Markdown.

ВАЖНО:
  - bitsandbytes НЕ используется. bf16 на CUDA.
  - KMP_DUPLICATE_LIB_OK=TRUE.
  - Два LoRA-адаптера передаются через --adapter_header и --adapter_thesis;
    они переключаются на одной и той же базовой модели.

Пример:
    python 03_inference.py \\
      --base_model      utrobinmv/t5_summary_en_ru_zh_large_2048 \\
      --adapter_header  ./checkpoints/header/best \\
      --adapter_thesis  ./checkpoints/thesis/best \\
      --transcript      ./raw/transcripts/Дискуссионная_панель.txt \\
      --out_md          ./summary_hierarchical.md \\
      --window_sec 300 --stride_sec 240
"""
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
from dataclasses import dataclass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
prep = import_module("01_prepare_dataset")


# ============== вспомогательные ==============

RE_TC_LINE = re.compile(
    r"^\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s*[-\u2013\u2014]\s*"
    r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s*:\s*(.+?)\s*$"
)


def fmt_tc(seconds: float) -> str:
    sec = int(round(seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---- стоп-слова для нормализации заголовков/тезисов при сравнении -----------
RU_STOPWORDS_4PLUS = {
    "доклад", "доклады", "докладе", "докладов",
    "обсуждение", "обсуждения", "обсуждаются", "обсуждается",
    "обсуждают", "обсуждали", "вопрос", "вопросы", "вопросов",
    "тема", "темы", "теме", "темой",
    "также", "который", "которая", "которые", "которых",
    "является", "быть", "были", "было",
}


def tokens_set(s: str) -> set[str]:
    s = s.lower()
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
    toks = [w for w in s.split() if len(w) >= 4 and w not in RU_STOPWORDS_4PLUS]
    return set(toks)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ============== постпроцессинг текста ==============

# эхо тайм-кода в начале строки: '1:05:35 - 1:107:12:', '12:01 - 17:01:', '1.13:18 - 1:15:01:'
RE_TC_ECHO_PREFIX = re.compile(
    r"^\s*\d{1,2}[:\.]\d{1,3}(?:[:\.]\d{1,3})?"
    r"(?:\s*[-\u2013\u2014\s]+\s*\d{1,2}[:\.]\d{1,3}(?:[:\.]\d{1,3})?)?"
    r"\s*:?\s*",
    flags=re.UNICODE,
)

# префикс числового ID: '001:', '08:', '1:', '02 ' (как обрезок тайм-кода)
RE_NUMERIC_ID_PREFIX = re.compile(r"^\s*\d{1,4}\s*[:\.]\s*")
# одиночные цифры/двузначные перед словом (обрезок тайм-кода без двоеточия):
# '02 уточняется...', '1 Обсуждается...'
RE_LEADING_NUM_WORD = re.compile(r"^\s*\d{1,3}\s+(?=[а-яА-ЯёЁ])")
RE_LEADING_NUM_DASH_WORD = re.compile(r"^\s*\d{1,4}\s*[-\u2013\u2014]\s*(?=[а-яА-ЯёЁ])")

# Артефакты вида "44 - 0:33:49:" и эхо промпта "Диапазон тезиса: 1:55:32 - 1.58:25:".
RE_NUMERIC_DASH_TC_PREFIX = re.compile(
    r"^\s*\d{1,4}\s*[-\u2013\u2014]\s*"
    r"(?:\d{1,2}[:\.])?\d{1,3}[:\.]\d{1,3}(?:[:\.]\d{1,3})?"
    r"\s*:?\s*",
    flags=re.UNICODE,
)
RE_PROMPT_RANGE_PREFIX = re.compile(
    r"^\s*Диапазон(?:\s+тезиса)?\s*:\s*"
    r"(?:\d{1,2}[:\.])?\d{1,3}[:\.]\d{1,3}"
    r"\s*[-\u2013\u2014]\s*"
    r"(?:\d{1,2}[:\.])?\d{1,3}[:\.]\d{1,3}"
    r"\s*:?\s*",
    flags=re.IGNORECASE | re.UNICODE,
)

# мета-фразы, при которых тезис считается галлюцинацией инструкции
META_PHRASES = (
    "тезис должен быть",
    "без мета-речи",
    "без вводных слов",
    "вернуть строку",
    "верни одну строку",
    "задача:",
)

# Частые фиксированные галлюцинации thesis/header-адаптера: искажённые термины.
# Применяются в постпроцессинге как быстрый фикс без переобучения. Расширяй список по мере нахождения.
# Пары (паттерн_основы, основа-замены). Основа подменяется при сохранении окончания.
# Окончание захватывается второй группой и подставляется обратно через \\2.
HALLUCINATION_FIXES = [
    # пластилизация / пластилизации / пластилизацию… -> кластеризация...
    (re.compile(r"пластилизаци(я|и|ии|ей|ю|е)\b", flags=re.IGNORECASE),
     lambda m: "кластеризаци" + m.group(1)),
    # технологполис -> технополис (сохраняем окончание)
    (re.compile(r"технологполис(ы|ов|ам|ами|ах|а|у|ом)?\b", flags=re.IGNORECASE),
     lambda m: "технополис" + (m.group(1) or "")),
    # научные города / научногороды -> наукограды
    (re.compile(r"научногород(ов|ы|ам|ами|ах|а|у|ом)?\b", flags=re.IGNORECASE),
     lambda m: "наукоград" + (m.group(1) or "")),
    # ФИО варианты
    (re.compile(r"Дарвина-Отсона", flags=re.IGNORECASE), "Дарбина-Уотсона"),
    (re.compile(r"Шапина-Лынка", flags=re.IGNORECASE), "Шапиро-Уилка"),
]


def fix_known_hallucinations(text: str) -> str:
    """Пост-replace частых галлюцинаций-терминов. Дешёвый фикс без переобучения."""
    for pat, repl in HALLUCINATION_FIXES:
        text = pat.sub(repl, text)
    return text


def _norm_word(w: str) -> str:
    return w.lower().strip(".,;:!?«»\"()")


def _collapse_repeated_phrases(text: str) -> str:
    """Схлопывает повторяющиеся фрагменты (результат beam-search без repetition_penalty).
    Ловит как непрерывные повторы (X X), так и разорванные союзом (X и X).
    Пример: 'особо экономической зоны и особо экономической зоны' -> 'особо экономической зоны'.
    Работает на повторах из 2–5 слов, в несколько проходов.
    """
    GLUE = {"и", "а", "но", "или", "да", "либо"}
    for _pass in range(3):
        words = text.split()
        if len(words) < 4:
            break
        changed = False
        for size in (5, 4, 3, 2):
            out: list[str] = []
            i = 0
            while i < len(words):
                # вариант 1: непрерывный повтор X X
                if i + 2 * size <= len(words):
                    a = [_norm_word(w) for w in words[i:i + size]]
                    b = [_norm_word(w) for w in words[i + size:i + 2 * size]]
                    if a == b and any(len(w) >= 4 for w in a):
                        out.extend(words[i:i + size])
                        i += 2 * size
                        changed = True
                        continue
                # вариант 2: повтор через союз X <союз> X
                if i + 2 * size + 1 <= len(words):
                    a = [_norm_word(w) for w in words[i:i + size]]
                    glue = _norm_word(words[i + size])
                    b = [_norm_word(w) for w in words[i + size + 1:i + 2 * size + 1]]
                    if glue in GLUE and a == b and any(len(w) >= 4 for w in a):
                        # оставляем X <союз>, второй X выбрасываем
                        # впрочем, если после второго X ничего нет — и союз выкинем
                        tail_after = i + 2 * size + 1
                        if tail_after >= len(words):
                            out.extend(words[i:i + size])
                        else:
                            out.extend(words[i:i + size + 1])
                        i = tail_after
                        changed = True
                        continue
                out.append(words[i])
                i += 1
            words = out
        text = " ".join(words)
        if not changed:
            break
    return text


def clean_body(text: str) -> str:
    """Удаляет из текста тезиса/заголовка эхо тайм-кода, мусорные ID-префиксы
    и схлопывает повторяющиеся фрагменты. Может срабатывать несколько раз подряд.
    """
    s = text.strip().strip("\"'«»“”")
    for _ in range(4):
        s = s.strip().strip("\"'«»“”")
        new_s = RE_PROMPT_RANGE_PREFIX.sub("", s, count=1)
        new_s = RE_NUMERIC_DASH_TC_PREFIX.sub("", new_s, count=1)
        new_s = RE_TC_ECHO_PREFIX.sub("", new_s, count=1)
        new_s = RE_NUMERIC_ID_PREFIX.sub("", new_s, count=1)
        new_s = RE_LEADING_NUM_DASH_WORD.sub("", new_s, count=1)
        new_s = RE_LEADING_NUM_WORD.sub("", new_s, count=1)
        new_s = new_s.lstrip(":-–—. ").strip().strip("\"'«»“”")
        if new_s == s:
            break
        s = new_s
    # Схлопываем повторяющиеся фрагменты («особой экономической зоны» x2 — x1)
    s = _collapse_repeated_phrases(s)
    # Исправляем известные галлюцинации-термины
    s = fix_known_hallucinations(s)
    # Убираем двойные пробелы
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def is_meta_hallucination(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in META_PHRASES)


def parse_model_line(text: str, default_start: float, default_end: float) -> tuple[float, float, str]:
    """Парсит сгенерированную моделью строку 'MM:SS - MM:SS: текст'. Если строка
    не соответствует формату — возвращает (default_start, default_end, очищенный текст).
    Тело тезиса/заголовка дополнительно прогоняется через clean_body() —
    срезаются повторные эхо тайм-кода и числовые ID-префиксы.
    """
    text = text.strip()
    m = RE_TC_LINE.match(text)
    if m:
        def _s(h, mn, sc):
            hh = int(h) if h else 0
            return hh * 3600 + int(mn) * 60 + int(sc)
        s = _s(m.group(1), m.group(2), m.group(3))
        e = _s(m.group(4), m.group(5), m.group(6))
        body = clean_body(m.group(7))
        return float(s), float(e), body
    body = clean_body(text)
    return default_start, default_end, body


# ============== модель ==============

@dataclass
class TwoAdapterModel:
    tokenizer: object
    model: object   # PEFT-модель с двумя подгруженными адаптерами
    has_thesis_adapter: bool

    @torch.inference_mode()
    def generate(self, prompt: str, adapter_name: str,
                 max_new_tokens: int = 96, max_source: int = 1950,
                 num_beams: int = 6, repetition_penalty: float = 1.15,
                 length_penalty: float = 0.8) -> str:
        # переключаемся на нужный адаптер
        self.model.set_adapter(adapter_name)
        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=max_source).to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            no_repeat_ngram_size=4,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=True,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def load_two_adapter_model(base_model: str, adapter_header: Path,
                           adapter_thesis: Path | None) -> TwoAdapterModel:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=torch.bfloat16).to("cuda")
    print(f">>> loading header adapter: {adapter_header}", flush=True)
    model = PeftModel.from_pretrained(base, str(adapter_header), adapter_name="header")
    has_thesis = adapter_thesis is not None and Path(adapter_thesis).exists()
    if has_thesis:
        print(f">>> loading thesis adapter: {adapter_thesis}", flush=True)
        model.load_adapter(str(adapter_thesis), adapter_name="thesis")
    else:
        print(">>> thesis adapter не задан или не найден — тезисы будут генериться адаптером 'header'",
              flush=True)
    model.eval()
    return TwoAdapterModel(tokenizer=tokenizer, model=model, has_thesis_adapter=has_thesis)


# ============== этап 1: окна -> заголовки ==============

def sliding_windows(t_start: float, t_end: float,
                    window_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    """Окна [start, start + window_sec] с шагом stride_sec, последнее — до t_end."""
    out: list[tuple[float, float]] = []
    s = t_start
    while s < t_end:
        e = min(s + window_sec, t_end)
        out.append((s, e))
        if e >= t_end:
            break
        s += stride_sec
    return out


def build_header_prompt(turns_window: list, ts: float, te: float,
                       tokenizer=None, max_input_tokens: int = 1950) -> str:
    window_text = "\n".join(
        f"[{fmt_tc(t.start)}-{fmt_tc(t.end)}] {t.speaker}: {t.text}"
        for t in turns_window
    )
    prompt = (
        "Задача: определи тематический заголовок сегмента дискуссии. "
        "Верни ровно одну строку формата 'MM:SS - MM:SS: <заголовок>' "
        "с указанным временным диапазоном. Без вводных фраз.\n\n"
        f"Диапазон: {fmt_tc(ts)} - {fmt_tc(te)}\n\n"
        "Фрагмент транскрипта:\n"
        f"{window_text}"
    )
    # Сжимаем длинные окна тем же способом, что при обучении (HEAD+MID+TAIL).
    if tokenizer is not None:
        head_part, body_part = prompt.split("Фрагмент транскрипта:\n", 1)
        prompt = prep._smart_truncate_window_lines(
            head_part, body_part.splitlines(), tokenizer, max_input_tokens
        )
    return prompt


def infer_headers(turns: list, model_pack: TwoAdapterModel,
                  window_sec: float, stride_sec: float) -> list[dict]:
    if not turns:
        return []
    t_start = turns[0].start
    t_end = turns[-1].end
    windows = sliding_windows(t_start, t_end, window_sec, stride_sec)
    print(f">>> окон: {len(windows)} (window={window_sec}c stride={stride_sec}c)", flush=True)

    results: list[dict] = []
    for i, (ws, we) in enumerate(windows):
        subset = [t for t in turns if t.end >= ws and t.start <= we]
        if not subset:
            continue
        prompt = build_header_prompt(subset, ws, we,
                                     tokenizer=model_pack.tokenizer,
                                     max_input_tokens=1950)
        out = model_pack.generate(prompt, adapter_name="header", max_new_tokens=96)
        s, e, body = parse_model_line(out, ws, we)
        # если модель вернула странный диапазон — используем границы окна
        if not (ws - 5 <= s <= we + 5) or not (ws - 5 <= e <= we + 60):
            s, e = ws, we
        results.append({"start": s, "end": e, "header": body, "win_idx": i,
                        "win_start": ws, "win_end": we})
        print(f"  [окно {i+1}/{len(windows)}] {fmt_tc(ws)}-{fmt_tc(we)} -> {body[:80]}",
              flush=True)
    return results


def dominant_speaker_with_share(turns: list, ts: float, te: float) -> tuple[str | None, float]:
    """Доминирующий спикер в окне + его доля от общего времени речи."""
    subset = [t for t in turns if t.end >= ts and t.start <= te]
    if not subset:
        return None, 0.0
    tally: dict[str, float] = {}
    for t in subset:
        ov = max(0.0, min(t.end, te) - max(t.start, ts))
        tally[t.speaker] = tally.get(t.speaker, 0.0) + ov
    if not tally:
        return None, 0.0
    total = sum(tally.values())
    top_spk, top_t = max(tally.items(), key=lambda x: x[1])
    return top_spk, (top_t / total if total > 0 else 0.0)


def dominant_speaker(turns: list, ts: float, te: float) -> str | None:
    """Backward-compat: возвращает только имя."""
    return dominant_speaker_with_share(turns, ts, te)[0]


def _has_real_diarization(turns: list | None) -> bool:
    """True если в транскрипте реально больше одного спикера.
    Если везде стоит один и тот же SPEAKER_XX (эффект отключённой диаризации) —
    speaker-сигнал нельзя использовать, иначе всё склеивается в один сегмент.
    """
    if not turns:
        return False
    speakers = {getattr(t, "speaker", None) for t in turns}
    speakers.discard(None)
    return len(speakers) >= 2


def render_boundary_context(turns: list, center: float, context_sec: float) -> str:
    left = center - context_sec
    right = center + context_sec
    subset = [t for t in turns if t.end >= left and t.start <= right]
    lines = [f"CENTER: {fmt_tc(center)}"]
    marker_added = False
    for turn in subset:
        if not marker_added and turn.start >= center:
            lines.append(f"<<<BOUNDARY {fmt_tc(center)}>>>")
            marker_added = True
        lines.append(f"[{fmt_tc(turn.start)}-{fmt_tc(turn.end)}] {turn.text}")
    if not marker_added:
        lines.append(f"<<<BOUNDARY {fmt_tc(center)}>>>")
    return "\n".join(lines)


@torch.inference_mode()
def predict_boundary_points(
    turns: list,
    boundary_model_path: Path,
    *,
    step_sec: float = 15.0,
    context_sec: float = 90.0,
    max_length: int = 512,
    device: str = "cuda",
) -> tuple[list[dict], list[str]]:
    from transformers import AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(boundary_model_path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(boundary_model_path))
    actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model.to(actual_device)
    model.eval()

    id2label_raw = getattr(model.config, "id2label", None) or {}
    id2label = {int(k): v for k, v in id2label_raw.items()} if id2label_raw else {}
    if not id2label:
        id2label = {0: "talk_start", 1: "talk_end", 2: "talk_inside", 3: "non_talk"}
    labels = [id2label[i] for i in sorted(id2label)]

    points: list[dict] = []
    start = turns[0].start
    end = turns[-1].end
    t = start
    while t <= end:
        text = render_boundary_context(turns, t, context_sec)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(actual_device)
        logits = model(**enc).logits[0]
        probs_tensor = torch.softmax(logits.float(), dim=-1).detach().cpu()
        probs = {
            id2label.get(i, str(i)): float(probs_tensor[i])
            for i in range(len(probs_tensor))
        }
        label = max(probs.items(), key=lambda item: item[1])[0]
        points.append({"time": float(t), "label": label, "probs": probs})
        t += step_sec

    del model
    if actual_device == "cuda":
        torch.cuda.empty_cache()
    return points, labels


def _score(point: dict, label: str) -> float:
    return float(point.get("probs", {}).get(label, 0.0))


def _snap_to_turn_start(turns: list, value: float, snap_sec: float) -> float:
    candidates = [t.start for t in turns if abs(t.start - value) <= snap_sec]
    if not candidates:
        return float(value)
    return float(min(candidates, key=lambda x: abs(x - value)))


def _snap_to_turn_end(turns: list, value: float, snap_sec: float) -> float:
    candidates = [t.end for t in turns if abs(t.end - value) <= snap_sec]
    if not candidates:
        return float(value)
    return float(min(candidates, key=lambda x: abs(x - value)))


def _merge_close_ranges(ranges: list[tuple[float, float]], gap_sec: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if out and start - out[-1][1] <= gap_sec:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


START_CUE_PATTERNS = (
    re.compile(r"(передаю|предоставляю)\s+слово", re.IGNORECASE),
    re.compile(r"слово\s+предоставляется", re.IGNORECASE),
    re.compile(r"переходим\s+к[о]?\s+.{0,80}доклад", re.IGNORECASE),
    re.compile(r"следующ(ий|ая|ее)\s+(доклад|выступлени)", re.IGNORECASE),
    re.compile(
        r"у\s+нас\s+.{0,40}(перв|втор|трет|треть|четверт|пят|следующ|остал[а-я]*|запланирован)\S*\s+.{0,50}доклад",
        re.IGNORECASE,
    ),
    re.compile(r"у\s+меня\s+.{0,30}доклад", re.IGNORECASE),
    re.compile(r"вашему\s+вниманию\s+.{0,80}доклад", re.IGNORECASE),
    re.compile(r"(тема|название)\s+(моего|нашего|данного|сегодняшнего)?\s*доклада", re.IGNORECASE),
    re.compile(r"сегодняшн(ий|яя|ее)\s+доклад\s+.{0,80}посвя[тщ]", re.IGNORECASE),
    re.compile(r"(мой|моя|наш|наша|совместн\w*)\s+.{0,60}доклад", re.IGNORECASE),
    re.compile(r"в\s+рамках\s+.{0,30}доклада", re.IGNORECASE),
    re.compile(r"(представить|представляется|представляю)\s+.{0,80}(доклад|работ[ау])", re.IGNORECASE),
    re.compile(r"работ[ауы]\s+на\s+тему", re.IGNORECASE),
    re.compile(r"будет\s+доклад", re.IGNORECASE),
    re.compile(r"докладчик\s+[А-ЯЁA-Z]", re.IGNORECASE),
    re.compile(r"доклад\s+(называется|посвящ[её]н|представляет)", re.IGNORECASE),
    re.compile(r"(уважаемые\s+коллеги|добрый\s+день).{0,120}(доклад|тема|исследован|работ)", re.IGNORECASE),
)

END_CUE_PATTERNS = (
    re.compile(r"спасибо\s+за\s+внимание", re.IGNORECASE),
    re.compile(r"спасибо\s+.{0,30}(за|вам)\s+.{0,30}доклад", re.IGNORECASE),
    re.compile(r"на\s+этом\s+.{0,50}(закончу|все|всё)", re.IGNORECASE),
    re.compile(r"будем\s+рады\s+ответить\s+на\s+.{0,40}вопрос", re.IGNORECASE),
    re.compile(r"(есть|будут|какие)\s+.{0,40}вопрос(ы|ов)?\s+(по|к)\s+доклад", re.IGNORECASE),
    re.compile(r"какие\s+.{0,30}вопрос", re.IGNORECASE),
    re.compile(r"есть\s+ли\s+.{0,20}вопрос", re.IGNORECASE),
    re.compile(r"у\s+кого\s+.{0,30}вопрос", re.IGNORECASE),
    re.compile(r"время\s+.{0,30}вышло", re.IGNORECASE),
    re.compile(r"поблагодарим\s+докладчика", re.IGNORECASE),
)


def _dedup_times(times: list[float], window_sec: float) -> list[float]:
    out: list[float] = []
    for value in sorted(times):
        if not out or value - out[-1] > window_sec:
            out.append(float(value))
    return out


def detect_talk_ranges_by_cues(
    turns: list,
    *,
    min_talk_sec: float = 180.0,
    merge_gap_sec: float = 45.0,
    snap_sec: float = 45.0,
    debug: bool = False,
) -> list[tuple[float, float]]:
    starts: list[float] = []
    ends: list[float] = []
    for turn in turns:
        text = turn.text or ""
        if any(pattern.search(text) for pattern in START_CUE_PATTERNS):
            starts.append(_snap_to_turn_start(turns, turn.start, snap_sec))
        if any(pattern.search(text) for pattern in END_CUE_PATTERNS):
            ends.append(_snap_to_turn_end(turns, turn.end, snap_sec))

    starts = _dedup_times(starts, 60.0)
    ends = _dedup_times(ends, 45.0)
    events = [(time, "start") for time in starts] + [(time, "end") for time in ends]
    events.sort()

    ranges: list[tuple[float, float]] = []
    current_start: float | None = None
    for time, kind in events:
        if kind == "start":
            if current_start is not None and time - current_start >= min_talk_sec:
                ranges.append((current_start, time))
            current_start = time
        elif kind == "end" and current_start is not None:
            if time - current_start >= min_talk_sec:
                ranges.append((current_start, time))
                current_start = None

    if current_start is not None and turns[-1].end - current_start >= min_talk_sec:
        ranges.append((current_start, turns[-1].end))

    ranges = _merge_close_ranges(ranges, merge_gap_sec)
    if debug:
        print(f">>> cue starts: {[fmt_tc(x) for x in starts]}", flush=True)
        print(f">>> cue ends: {[fmt_tc(x) for x in ends]}", flush=True)
        print(f">>> cue ranges: {len(ranges)}", flush=True)
        for start, end in ranges:
            print(f"     {fmt_tc(start)}-{fmt_tc(end)}", flush=True)
    return ranges


def _overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    if right <= left:
        return 0.0
    return (right - left) / max(1.0, min(a[1] - a[0], b[1] - b[0]))


def _range_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    if a[1] < b[0]:
        return b[0] - a[1]
    if b[1] < a[0]:
        return a[0] - b[1]
    return 0.0


def combine_model_and_cue_ranges(
    model_ranges: list[tuple[float, float]],
    cue_ranges: list[tuple[float, float]],
    *,
    merge_gap_sec: float = 45.0,
) -> list[tuple[float, float]]:
    if not cue_ranges:
        return model_ranges
    combined = list(cue_ranges)
    for model_range in model_ranges:
        # Cue ranges are high-precision anchors. The classifier is used mainly
        # to recover talks that have no explicit start/end phrases.
        if all(
            _overlap_ratio(model_range, cue_range) < 0.35
            and _range_gap(model_range, cue_range) > merge_gap_sec
            for cue_range in cue_ranges
        ):
            combined.append(model_range)
    return _merge_close_ranges(combined, merge_gap_sec)


LECTURE_MARKER_PATTERNS = (
    re.compile(
        r"\b(исследован\w*|анализ\w*|модел\w*|данн\w*|выборк\w*|регресси\w*|кластер\w*|"
        r"показател\w*|переменн\w*|гипотез\w*|результат\w*|метод\w*|оцен\w*|коэффициент\w*|"
        r"наблюдени\w*|фактор\w*|индекс\w*|слайд\w*|таблиц\w*|эксперимент\w*|вывод\w*|"
        r"задач\w*|цель\w*|проблем\w*|эффективн\w*|инструмент\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(в\s+нашем|в\s+данном|нами\s+был|мы\s+получили|мы\s+рассматриваем)\b", re.IGNORECASE),
    re.compile(r"\b(на\s+слайде|на\s+графике|в\s+таблице|здесь\s+представлен)\b", re.IGNORECASE),
)

DISCUSSION_MARKER_PATTERNS = (
    re.compile(r"\bвопрос(ы|ов|ам|ами|ах)?\b", re.IGNORECASE),
    re.compile(r"\bответ(ы|ов)?\b", re.IGNORECASE),
    re.compile(r"\bспасибо\b", re.IGNORECASE),
    re.compile(r"\bпоблагодарим\b", re.IGNORECASE),
    re.compile(r"\bкомментар", re.IGNORECASE),
)

ORGANIZATIONAL_MARKER_PATTERNS = (
    re.compile(r"\bрегламент\b", re.IGNORECASE),
    re.compile(r"\bперерыв\b", re.IGNORECASE),
    re.compile(r"\bкофе\b", re.IGNORECASE),
    re.compile(r"\bzoom\b|\bзум\b", re.IGNORECASE),
    re.compile(r"\bмикрофон\b|\bкамер", re.IGNORECASE),
    re.compile(r"\bв\s+чате\b|\bв\s+зале\b", re.IGNORECASE),
    re.compile(r"\bподключ", re.IGNORECASE),
)


def _has_any_pattern(text: str, patterns: tuple[re.Pattern, ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _range_turns(turns: list, start: float, end: float) -> list:
    return [t for t in turns if t.end >= start and t.start <= end]


def _joined_turn_text(turns: list) -> str:
    return " ".join((getattr(t, "text", "") or "") for t in turns)


def _range_text(turns: list, start: float, end: float) -> str:
    return _joined_turn_text(_range_turns(turns, start, end))


def _window_lecture_score(window_turns: list) -> float:
    if not window_turns:
        return -10.0
    text = _joined_turn_text(window_turns)
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    lecture_hits = sum(1 for pattern in LECTURE_MARKER_PATTERNS if pattern.search(text))
    discussion_hits = sum(1 for pattern in DISCUSSION_MARKER_PATTERNS if pattern.search(text))
    org_hits = sum(1 for pattern in ORGANIZATIONAL_MARKER_PATTERNS if pattern.search(text))
    long_turns = sum(1 for turn in window_turns if len((turn.text or "").split()) >= 10)
    score = float(lecture_hits) + min(long_turns, 5) * 0.55
    if _has_any_pattern(text, START_CUE_PATTERNS):
        score += 2.5
    score -= discussion_hits * 1.15
    score -= org_hits * 0.9
    if len(words) < 45:
        score -= 1.5
    return score


def _refine_rescue_start(turns: list, rough_start: float, rough_end: float) -> float:
    for turn in turns:
        if turn.end < rough_start - 90 or turn.start > rough_end:
            continue
        text = turn.text or ""
        if turn.start <= rough_start + 120.0 and _has_any_pattern(text, START_CUE_PATTERNS):
            return float(turn.start)
    return float(rough_start)


def _refine_rescue_end(turns: list, start: float, rough_end: float, gap_end: float,
                       min_talk_sec: float) -> float:
    search_end = min(gap_end, rough_end + 180.0)
    for turn in turns:
        if turn.end < start + min_talk_sec or turn.start > search_end:
            continue
        text = turn.text or ""
        if _has_any_pattern(text, END_CUE_PATTERNS):
            return float(turn.end)
    return float(min(search_end, gap_end))


def _complement_gaps(ranges: list[tuple[float, float]], full_start: float,
                     full_end: float) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = full_start
    for start, end in sorted(ranges):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < full_end:
        gaps.append((cursor, full_end))
    return gaps


def _merge_close_ranges_preserving_starts(
    turns: list,
    ranges: list[tuple[float, float]],
    gap_sec: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if out and start - out[-1][1] <= gap_sec:
            gap_text = _joined_turn_text(_range_turns(turns, out[-1][1], start))
            if _has_any_pattern(gap_text, START_CUE_PATTERNS):
                out.append((start, end))
            else:
                out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def rescue_talk_ranges_in_long_gaps(
    turns: list,
    ranges: list[tuple[float, float]],
    *,
    long_gap_sec: float = 600.0,
    min_talk_sec: float = 300.0,
    window_sec: float = 150.0,
    step_sec: float = 30.0,
    score_threshold: float = 1.6,
    merge_gap_sec: float = 60.0,
    debug: bool = False,
) -> list[tuple[float, float]]:
    if not ranges:
        return []
    rescued: list[tuple[float, float]] = []
    gaps = _complement_gaps(ranges, turns[0].start, turns[-1].end)
    for gap_start, gap_end in gaps:
        if gap_end - gap_start < long_gap_sec:
            continue
        gap_text = _joined_turn_text(_range_turns(turns, gap_start, gap_end))
        if gap_end >= turns[-1].end - 180.0 and not _has_any_pattern(gap_text, START_CUE_PATTERNS):
            continue
        likely_windows: list[tuple[float, float, float]] = []
        t = gap_start
        while t < gap_end:
            ws = t
            we = min(t + window_sec, gap_end)
            score = _window_lecture_score(_range_turns(turns, ws, we))
            if score >= score_threshold:
                likely_windows.append((ws, we, score))
            t += step_sec

        if not likely_windows:
            continue

        runs: list[tuple[float, float, float]] = []
        cur_s, cur_e, best_score = likely_windows[0]
        for ws, we, score in likely_windows[1:]:
            if ws - cur_e <= merge_gap_sec:
                cur_e = max(cur_e, we)
                best_score = max(best_score, score)
            else:
                runs.append((cur_s, cur_e, best_score))
                cur_s, cur_e, best_score = ws, we, score
        runs.append((cur_s, cur_e, best_score))

        for rough_start, rough_end, best_score in runs:
            start = _refine_rescue_start(turns, rough_start, rough_end)
            end = _refine_rescue_end(turns, start, rough_end, gap_end, min_talk_sec)
            if end - start < min_talk_sec:
                continue
            candidate = (start, end)
            if any(_overlap_ratio(candidate, existing) >= 0.25 for existing in ranges):
                continue
            rescued.append(candidate)
            if debug:
                print(
                    f">>> rescued talk in long gap: {fmt_tc(start)}-{fmt_tc(end)} "
                    f"(score={best_score:.2f})",
                    flush=True,
                )
    return rescued


def _range_tokens(turns: list, start: float, end: float) -> set[str]:
    text = _joined_turn_text(_range_turns(turns, start, end))
    return tokens_set(text)


def merge_fragmented_same_topic_ranges(
    turns: list,
    ranges: list[tuple[float, float]],
    *,
    max_gap_sec: float = 120.0,
    jaccard_threshold: float = 0.16,
    debug: bool = False,
) -> list[tuple[float, float]]:
    if len(ranges) < 2:
        return ranges
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        gap = start - prev_end
        if gap < 0:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        gap_text = _joined_turn_text(_range_turns(turns, prev_end, start))
        has_new_start = _has_any_pattern(gap_text, START_CUE_PATTERNS)
        left = _range_tokens(turns, prev_start, prev_end)
        right = _range_tokens(turns, start, end)
        sim = jaccard(left, right)
        common = len(left & right)
        if gap <= max_gap_sec and not has_new_start and sim >= jaccard_threshold and common >= 4:
            if debug:
                print(
                    f">>> merge fragmented same-topic talks: {fmt_tc(prev_start)}-{fmt_tc(end)} "
                    f"(gap={gap:.0f}s, jacc={sim:.2f}, common={common})",
                    flush=True,
                )
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def detect_talk_ranges_with_boundary_model(
    turns: list,
    boundary_model_path: Path,
    *,
    step_sec: float = 15.0,
    context_sec: float = 90.0,
    start_threshold: float = 0.45,
    end_threshold: float = 0.45,
    inside_threshold: float = 0.55,
    min_talk_sec: float = 180.0,
    merge_gap_sec: float = 45.0,
    snap_sec: float = 45.0,
    rescue_long_gap_sec: float = 600.0,
    rescue_min_talk_sec: float = 300.0,
    rescue_window_sec: float = 150.0,
    rescue_score_threshold: float = 1.6,
    topic_merge_gap_sec: float = 120.0,
    device: str = "cuda",
    debug: bool = False,
) -> list[tuple[float, float]]:
    points, labels = predict_boundary_points(
        turns,
        boundary_model_path,
        step_sec=step_sec,
        context_sec=context_sec,
        device=device,
    )
    has_event_labels = "talk_start" in labels and "talk_end" in labels
    ranges: list[tuple[float, float]] = []
    in_talk = False
    current_start: float | None = None
    inside_streak = 0

    for point in points:
        t = point["time"]
        if has_event_labels:
            p_start = _score(point, "talk_start")
            p_end = _score(point, "talk_end")
            p_inside = _score(point, "talk_inside")
            if not in_talk:
                if p_start >= start_threshold:
                    current_start = _snap_to_turn_start(turns, t, snap_sec)
                    in_talk = True
                    inside_streak = 0
                elif p_inside >= inside_threshold:
                    inside_streak += 1
                    if inside_streak >= 2:
                        current_start = _snap_to_turn_start(turns, t - step_sec, snap_sec)
                        in_talk = True
                else:
                    inside_streak = 0
            else:
                if (
                    p_start >= start_threshold
                    and current_start is not None
                    and t - current_start >= min_talk_sec
                ):
                    end = _snap_to_turn_end(turns, t, snap_sec)
                    if end - current_start >= min_talk_sec:
                        ranges.append((current_start, end))
                    current_start = _snap_to_turn_start(turns, t, snap_sec)
                elif (
                    p_end >= end_threshold
                    and current_start is not None
                    and t - current_start >= min_talk_sec
                ):
                    end = _snap_to_turn_end(turns, t, snap_sec)
                    if end - current_start >= min_talk_sec:
                        ranges.append((current_start, end))
                    current_start = None
                    in_talk = False
                    inside_streak = 0
        else:
            # Backward-compatible binary mode for old talk/non_talk classifiers.
            label = point["label"]
            is_talk = label == "talk" or _score(point, "talk") >= inside_threshold
            if is_talk and not in_talk:
                current_start = _snap_to_turn_start(turns, t, snap_sec)
                in_talk = True
            elif not is_talk and in_talk and current_start is not None:
                end = _snap_to_turn_end(turns, t, snap_sec)
                if end - current_start >= min_talk_sec:
                    ranges.append((current_start, end))
                current_start = None
                in_talk = False

    if in_talk and current_start is not None:
        end = _snap_to_turn_end(turns, turns[-1].end, snap_sec)
        if end - current_start >= min_talk_sec:
            ranges.append((current_start, end))

    ranges = _merge_close_ranges(ranges, merge_gap_sec)
    cue_ranges = detect_talk_ranges_by_cues(
        turns,
        min_talk_sec=min_talk_sec,
        merge_gap_sec=merge_gap_sec,
        snap_sec=snap_sec,
        debug=debug,
    )
    ranges = combine_model_and_cue_ranges(ranges, cue_ranges, merge_gap_sec=merge_gap_sec)
    rescued_ranges = rescue_talk_ranges_in_long_gaps(
        turns,
        ranges,
        long_gap_sec=rescue_long_gap_sec,
        min_talk_sec=rescue_min_talk_sec,
        window_sec=rescue_window_sec,
        score_threshold=rescue_score_threshold,
        merge_gap_sec=max(merge_gap_sec, 60.0),
        debug=debug,
    )
    if rescued_ranges:
        ranges = _merge_close_ranges_preserving_starts(turns, ranges + rescued_ranges, merge_gap_sec)
    ranges = merge_fragmented_same_topic_ranges(
        turns,
        ranges,
        max_gap_sec=topic_merge_gap_sec,
        debug=debug,
    )
    if debug:
        print(f">>> boundary labels: {labels}", flush=True)
        print(f">>> boundary ranges: {len(ranges)}", flush=True)
        for start, end in ranges:
            print(f"     {fmt_tc(start)}-{fmt_tc(end)}", flush=True)
    return ranges


def infer_header_for_range(turns: list, model_pack: TwoAdapterModel,
                           start: float, end: float) -> dict:
    subset = [t for t in turns if t.end >= start and t.start <= end]
    if not subset:
        return {"start": start, "end": end, "header": "Доклад", "win_start": start, "win_end": end}
    prompt = build_header_prompt(subset, start, end,
                                 tokenizer=model_pack.tokenizer,
                                 max_input_tokens=1950)
    out = model_pack.generate(prompt, adapter_name="header", max_new_tokens=96)
    _, _, body = parse_model_line(out, start, end)
    if not body or is_bad_header(body):
        body = "Доклад"
    return {"start": start, "end": end, "header": body, "win_start": start, "win_end": end}


def default_gap_header(start: float, end: float, full_start: float, full_end: float) -> str:
    if start <= full_start + 120:
        return "Вступление и организационные сообщения"
    if end >= full_end - 120:
        return "Заключение и обсуждение"
    return "Обсуждение и вопросы к докладу"


def build_segments_from_boundary_ranges(
    turns: list,
    model_pack: TwoAdapterModel,
    ranges: list[tuple[float, float]],
    *,
    include_gaps: bool = True,
    min_gap_sec: float = 60.0,
) -> list[dict]:
    if not ranges:
        return []
    segments: list[dict] = []
    full_start = turns[0].start
    full_end = turns[-1].end
    cursor = full_start

    for start, end in ranges:
        if include_gaps and start - cursor >= min_gap_sec:
            segments.append(
                {
                    "start": cursor,
                    "end": start,
                    "header": default_gap_header(cursor, start, full_start, full_end),
                    "win_start": cursor,
                    "win_end": start,
                }
            )
        talk_seg = infer_header_for_range(turns, model_pack, start, end)
        segments.append(talk_seg)
        cursor = max(cursor, end)

    if include_gaps and full_end - cursor >= min_gap_sec:
        segments.append(
            {
                "start": cursor,
                "end": full_end,
                "header": default_gap_header(cursor, full_end, full_start, full_end),
                "win_start": cursor,
                "win_end": full_end,
            }
        )
    return segments


def detect_event_mode(transcript_path: Path, turns: list, requested: str = "auto") -> str:
    """Определяет жанр записи. Стандартный режим не меняется для обычных секций."""
    if requested != "auto":
        return requested
    name = transcript_path.stem.lower().replace("_nd1", "")
    if "заключ" in name:
        return "closing"
    if "дискуссион" in name or "панель" in name or "круглый стол" in name:
        return "panel"
    first_text = _joined_turn_text(turns[: min(len(turns), 40)]).lower()
    if "заключительное засед" in first_text or "подвед" in first_text and "итог" in first_text:
        return "closing"
    if "панельн" in first_text or "кругл" in first_text:
        return "panel"
    return "standard"


CLOSING_REPORT_PATTERNS = (
    re.compile(r"\bотчет\b|\bотч[её]т\b", re.IGNORECASE),
    re.compile(r"\bсекци[ия]\s*\d+|\bпо\s+секци[ии]\b", re.IGNORECASE),
    re.compile(r"\bпленарн\w+\s+заседан", re.IGNORECASE),
    re.compile(r"\bкругл\w+\s+стол", re.IGNORECASE),
    re.compile(r"\bучастник\w+.{0,60}\bдоклад", re.IGNORECASE),
    re.compile(r"\bдоклад\w+.{0,60}\bучастник", re.IGNORECASE),
)

CLOSING_AWARD_PATTERNS = (
    re.compile(r"\bлауреат", re.IGNORECASE),
    re.compile(r"\bдиплом", re.IGNORECASE),
    re.compile(r"\bнагражд", re.IGNORECASE),
    re.compile(r"\bсертификат", re.IGNORECASE),
    re.compile(r"\bконкурс\b", re.IGNORECASE),
)

CLOSING_ORG_PATTERNS = (
    re.compile(r"\bорганизацион", re.IGNORECASE),
    re.compile(r"\bматериал\w*.{0,60}(подать|прислать|срок|октябр)", re.IGNORECASE),
    re.compile(r"\b(срок|до)\s+30\s+октябр", re.IGNORECASE),
    re.compile(r"\bредколлег", re.IGNORECASE),
    re.compile(r"\bфотограф|\bвидеоматериал|\bдиск\b", re.IGNORECASE),
    re.compile(r"\bпубликаци|\bстать[яи]", re.IGNORECASE),
)

CLOSING_FUTURE_PATTERNS = (
    re.compile(r"\bследующ\w+\s+(конференц|школ|семинар)", re.IGNORECASE),
    re.compile(r"\bуф[аеу]\b", re.IGNORECASE),
    re.compile(r"\bповысить\s+статус", re.IGNORECASE),
    re.compile(r"\bпросветительск", re.IGNORECASE),
)

CLOSING_SPLIT_CUE_PATTERNS = (
    re.compile(r"\b(итоги|отчет|отч[её]т)\s+.{0,50}\bсекци", re.IGNORECASE),
    re.compile(r"\bсекци[ия]\s*\d+", re.IGNORECASE),
    re.compile(r"\bпленарн\w+\s+заседан", re.IGNORECASE),
    re.compile(r"\bкругл\w+\s+стол", re.IGNORECASE),
    re.compile(r"\bлауреат|\bдиплом|\bнагражд", re.IGNORECASE),
    re.compile(r"\bорганизацион|\bматериал\w*.{0,60}(подать|прислать|срок|октябр)", re.IGNORECASE),
    re.compile(r"\bследующ\w+\s+(конференц|школ|семинар)", re.IGNORECASE),
    re.compile(r"\bхотел(а|и)?\s+бы\s+сказать", re.IGNORECASE),
    re.compile(r"\bеще\s+раз\s+спасибо|\bещ[её]\s+раз\s+благодар", re.IGNORECASE),
)


def _has_patterns(text: str, patterns: tuple[re.Pattern, ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _dedup_boundary_candidates(candidates: list[float], min_gap_sec: float = 240.0) -> list[float]:
    out: list[float] = []
    for value in sorted(candidates):
        if not out or value - out[-1] >= min_gap_sec:
            out.append(float(value))
    return out


def _closing_split_points(turns: list, start: float, end: float,
                          *, fallback_sec: float = 900.0) -> list[float]:
    points = [float(start), float(end)]
    for turn in turns:
        if turn.start <= start + 180.0 or turn.start >= end - 180.0:
            continue
        if _has_patterns(turn.text or "", CLOSING_SPLIT_CUE_PATTERNS):
            points.append(float(turn.start))

    if len(points) <= 2 and end - start > fallback_sec * 1.6:
        t = start + fallback_sec
        while t < end - 240.0:
            points.append(float(t))
            t += fallback_sec

    points = _dedup_boundary_candidates(points, 240.0)
    if points[0] > start:
        points.insert(0, float(start))
    if points[-1] < end:
        points.append(float(end))
    return points


def _header_from_closing_text(header: str, text: str, start: float, end: float,
                              full_end: float) -> str:
    text_l = text.lower()
    duration = end - start
    late_closing = start > full_end * 0.55
    if late_closing and _has_patterns(text_l, CLOSING_AWARD_PATTERNS):
        return "Вручение дипломов и награждение участников"
    if late_closing and _has_patterns(text_l, CLOSING_ORG_PATTERNS):
        return "Организационные вопросы по материалам конференции"
    if late_closing and _has_patterns(text_l, CLOSING_FUTURE_PATTERNS):
        return "Обсуждение дальнейшего развития конференции"
    # Отчеты по секциям часто содержат слова про публикации, дипломы или будущие
    # планы, поэтому проверяем их первыми. Иначе отчет превращается в non-talk.
    if _has_patterns(text_l, CLOSING_REPORT_PATTERNS):
        if "пленар" in text_l:
            return "Общий отчет о пленарных заседаниях и круглых столах"
        m = re.search(r"секци[ия]\s*(\d+)", text_l)
        if m:
            return f"Отчет секции {m.group(1)}"
        return "Отчет о работе секций и итогах конференции"
    if _has_patterns(text_l, CLOSING_AWARD_PATTERNS):
        return "Вручение дипломов и награждение участников"
    if _has_patterns(text_l, CLOSING_ORG_PATTERNS):
        return "Организационные вопросы по материалам конференции"
    if _has_patterns(text_l, CLOSING_FUTURE_PATTERNS):
        return "Обсуждение дальнейшего развития конференции"
    if duration > 900.0 or start > full_end * 0.45:
        return "Заключительное обсуждение итогов конференции"
    return header


def _maybe_rewrite_panel_header(seg: dict, text: str) -> str:
    header = seg.get("header", "")
    duration = float(seg["end"] - seg["start"])
    score = _window_lecture_score(_range_turns([], 0, 0)) if False else 0.0
    # Длинные фрагменты панелей часто получают заголовок "обсуждение",
    # хотя внутри идет полноценное выступление. Решение ниже только снимает
    # слово "обсуждение", а тему оставляет за T5-заголовком.
    if duration >= 600.0 and is_non_talk_header(header):
        score = _window_lecture_score_from_text(text)
        if score >= 2.0:
            cleaned = re.sub(r"^\s*обсуждени[ея]\s+(доклада|выступления)\s*", "Доклад ", header, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\s*обсуждени[ея]\s*", "Доклад ", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^\s*открыти[ея]\s+панельной\s+дискуссии\s+о\s+", "Доклад о ", cleaned, flags=re.IGNORECASE)
            if cleaned and not is_non_talk_header(cleaned):
                return cleaned
    return header


def _window_lecture_score_from_text(text: str) -> float:
    class _Turn:
        def __init__(self, text: str):
            self.text = text
    # Используем существующую скоринговую функцию, подавая текст как один
    # большой псевдо-turn. Этого достаточно для жанрового решения.
    return _window_lecture_score([_Turn(text)])


def _scan_likely_talk_ranges_in_interval(
    turns: list,
    start: float,
    end: float,
    *,
    min_talk_sec: float = 300.0,
    window_sec: float = 150.0,
    step_sec: float = 30.0,
    score_threshold: float = 2.0,
) -> list[tuple[float, float]]:
    likely: list[tuple[float, float, float]] = []
    t = start
    while t < end:
        ws = t
        we = min(t + window_sec, end)
        score = _window_lecture_score(_range_turns(turns, ws, we))
        if score >= score_threshold:
            likely.append((ws, we, score))
        t += step_sec
    if not likely:
        return []

    runs: list[tuple[float, float, float]] = []
    cur_s, cur_e, best = likely[0]
    for ws, we, score in likely[1:]:
        if ws - cur_e <= 90.0:
            cur_e = max(cur_e, we)
            best = max(best, score)
        else:
            runs.append((cur_s, cur_e, best))
            cur_s, cur_e, best = ws, we, score
    runs.append((cur_s, cur_e, best))

    out: list[tuple[float, float]] = []
    for rough_s, rough_e, _ in runs:
        rs = _refine_rescue_start(turns, rough_s, rough_e)
        re_ = _refine_rescue_end(turns, rs, rough_e, end, min_talk_sec)
        if re_ - rs >= min_talk_sec:
            out.append((rs, re_))
    return _merge_close_ranges(out, 90.0)


def apply_event_mode_segments(
    segments: list[dict],
    turns: list,
    model_pack: TwoAdapterModel,
    *,
    event_mode: str,
    rescue_long_non_talk_sec: float = 600.0,
    rescue_score_threshold: float = 2.0,
    closing_split_long_sec: float = 1500.0,
    debug: bool = False,
) -> list[dict]:
    if event_mode == "standard" or not segments:
        return segments

    full_start = turns[0].start
    full_end = turns[-1].end
    refined: list[dict] = []

    for seg in segments:
        seg = dict(seg)
        start = float(seg["start"])
        end = float(seg["end"])
        text = _range_text(turns, start, end)
        duration = end - start

        if event_mode == "panel" and is_non_talk_header(seg["header"]) and duration >= rescue_long_non_talk_sec:
            rescued = _scan_likely_talk_ranges_in_interval(
                turns,
                start,
                end,
                min_talk_sec=300.0,
                score_threshold=rescue_score_threshold,
            )
            if rescued:
                cursor = start
                for rs, re_ in rescued:
                    if rs - cursor >= 60.0:
                        refined.append({
                            "start": cursor,
                            "end": rs,
                            "header": seg["header"],
                            "win_start": cursor,
                            "win_end": rs,
                        })
                    talk_seg = infer_header_for_range(turns, model_pack, rs, re_)
                    refined.append(talk_seg)
                    cursor = max(cursor, re_)
                if end - cursor >= 60.0:
                    refined.append({
                        "start": cursor,
                        "end": end,
                        "header": default_gap_header(cursor, end, full_start, full_end),
                        "win_start": cursor,
                        "win_end": end,
                    })
                if debug:
                    print(f">>> event rescue panel: {fmt_tc(start)}-{fmt_tc(end)} -> {len(rescued)} talk ranges", flush=True)
                continue

        if event_mode == "closing" and duration >= closing_split_long_sec:
            points = _closing_split_points(turns, start, end)
            if len(points) > 2:
                for a, b in zip(points, points[1:]):
                    if b - a < 60.0:
                        continue
                    chunk_text = _range_text(turns, a, b)
                    refined.append({
                        "start": a,
                        "end": b,
                        "header": _header_from_closing_text(seg["header"], chunk_text, a, b, full_end),
                        "win_start": a,
                        "win_end": b,
                    })
                if debug:
                    print(f">>> event split closing: {fmt_tc(start)}-{fmt_tc(end)} -> {len(points)-1} chunks", flush=True)
                continue

        if event_mode == "closing":
            seg["header"] = _header_from_closing_text(seg["header"], text, start, end, full_end)
        elif event_mode == "panel":
            seg["header"] = _maybe_rewrite_panel_header(seg, text)
        refined.append(seg)

    return fix_overlapping_segments(refined)


NON_TALK_HEADER_PATTERNS = (
    re.compile(r"\bвопрос(ы|ов|ам|ами|ах)?\b", re.IGNORECASE),
    re.compile(r"\bответ(ы|ов)?\b", re.IGNORECASE),
    re.compile(r"\bобсуждени[ея]\s+(доклада|выступления)", re.IGNORECASE),
    re.compile(r"\bдискусси", re.IGNORECASE),
    re.compile(r"\bвступительн", re.IGNORECASE),
    re.compile(r"\bвступлени[ея]\b", re.IGNORECASE),
    re.compile(r"\bоткрыти[ея]\b", re.IGNORECASE),
    re.compile(r"\bзаключительн", re.IGNORECASE),
    re.compile(r"\bзаключени[ея]\b", re.IGNORECASE),
    re.compile(r"\bзавершени[ея]\b", re.IGNORECASE),
    re.compile(r"\bподведени[ея]\s+итог", re.IGNORECASE),
    re.compile(r"\bорганизационн", re.IGNORECASE),
    re.compile(r"\bслужебн", re.IGNORECASE),
    re.compile(r"\bперерыв\b", re.IGNORECASE),
    re.compile(r"\bрегламент\b", re.IGNORECASE),
    re.compile(r"\bвручени[ея]\b", re.IGNORECASE),
    re.compile(r"\bнагражден", re.IGNORECASE),
    re.compile(r"\bдиплом", re.IGNORECASE),
    re.compile(r"\bсертификат", re.IGNORECASE),
    re.compile(r"\bсрок[и]?\s+подач", re.IGNORECASE),
    re.compile(r"\bобсуждени[ея]\s+дальнейш", re.IGNORECASE),
    re.compile(r"\bзаключительн\w+\s+обсуждени", re.IGNORECASE),
)

BAD_HEADER_PATTERNS = (
    re.compile(r"^\s*\d{1,3}\s*$"),
    re.compile(r"^\s*0?3\s*$", re.IGNORECASE),
    re.compile(r"^\s*сегмент\s+конференции\s*$", re.IGNORECASE),
)

HARD_SPLIT_BOUNDARY_PATTERNS = (
    re.compile(r"спасибо\s+(за\s+внимание|большое|коллеги)", re.IGNORECASE),
    re.compile(r"есть\s+ли\s+вопрос", re.IGNORECASE),
    re.compile(r"\bвопросы?\s+(к\s+докладу|докладчику)", re.IGNORECASE),
    re.compile(r"можно\s+вопрос", re.IGNORECASE),
    re.compile(r"следующ(ий|ая|ее)\s+(доклад|выступа|участник|секци)", re.IGNORECASE),
    re.compile(r"переходим\s+к\s+(следующ|доклад|выступ)", re.IGNORECASE),
    re.compile(r"(передаю|предоставляю)\s+слово", re.IGNORECASE),
    re.compile(r"слово\s+предоставляется", re.IGNORECASE),
    re.compile(r"(тема|название)\s+(моего\s+)?доклада", re.IGNORECASE),
    re.compile(r"доклад\s+(называется|посвящ[её]н)", re.IGNORECASE),
    re.compile(r"разрешите\s+представить\s+доклад", re.IGNORECASE),
    re.compile(r"добрый\s+день.{0,80}\b(доклад|тема|представ)", re.IGNORECASE | re.DOTALL),
)


def is_bad_header(header: str) -> bool:
    text = (header or "").strip()
    if len(text) < 5:
        return True
    return any(p.search(text) for p in BAD_HEADER_PATTERNS)


def is_non_talk_header(header: str) -> bool:
    text = (header or "").strip().lower()
    if not text:
        return True
    return any(p.search(text) for p in NON_TALK_HEADER_PATTERNS)


def should_generate_theses(header: str, force_all: bool = False) -> bool:
    """Тезисы нужны только для докладов, а не для вопросов/вступлений/заключений."""
    if force_all:
        return True
    return not is_non_talk_header(header)


def _header_quality(header: str) -> float:
    h = (header or "").strip()
    if is_bad_header(h):
        return -1000.0
    score = min(len(h), 120) / 4.0
    hl = h.lower()
    if "доклад" in hl:
        score += 10.0
    if is_non_talk_header(h):
        score -= 15.0
    if "модель" in hl or "анализ" in hl or "исследован" in hl:
        score += 3.0
    return score


def _choose_best_header(headers: list[str]) -> str:
    good = [h for h in headers if h and h.strip()]
    if not good:
        return "Сегмент конференции"
    return max(good, key=_header_quality)


def _boundary_text(turns: list | None, center_sec: float, context_sec: float) -> str:
    if not turns:
        return ""
    a = center_sec - context_sec
    b = center_sec + context_sec
    parts = [t.text for t in turns if t.end >= a and t.start <= b]
    return " ".join(parts)


def _has_hard_split_cue(turns: list | None, boundary_sec: float,
                        context_sec: float) -> bool:
    text = _boundary_text(turns, boundary_sec, context_sec)
    if not text:
        return False
    return any(p.search(text) for p in HARD_SPLIT_BOUNDARY_PATTERNS)


def _merge_candidate(prev: dict, cur: dict, max_merged_segment_sec: float) -> bool:
    start = min(prev["start"], cur["start"])
    end = max(prev["end"], cur["end"])
    return (end - start) <= max_merged_segment_sec


def merge_adjacent_headers(headers: list[dict], jacc_thresh: float = 0.35,
                            turns: list | None = None,
                            speaker_share_thresh: float = 0.5,
                            votes_needed: int = 2,
                            debug: bool = False,
                            boundary_context_sec: float = 120.0,
                            max_merged_segment_sec: float = 1800.0) -> list[dict]:
    """Соседние окна с похожими заголовками склеиваем в один сегмент.
    Логика votes (по умолчанию нужно >= 2):
      1) Jaccard заголовков (с вырезанными стоп-словами) >= jacc_thresh — 1 голос;
      2) пересечение хотя бы 2 «содержательных» токенов — 1 голос;
      3) одинаковый доминирующий спикер в обоих окнах:
           - оба с долей >= speaker_share_thresh — STRONG, 2 голоса;
           - иначе (один и тот же топ-спикер, но доли ниже порога) — WEAK, 1 голос.
    Вето: разные доминирующие спикеры, оба с долей >= speaker_share_thresh.
    Если debug=True — печатает диагностику по каждой паре.
    """
    if not headers:
        return []

    has_diar = _has_real_diarization(turns)
    if debug and not has_diar:
        print(
            "   [merge] диаризация отсутствует — включена smart-склейка "
            f"(boundary={boundary_context_sec:.0f}c, max={max_merged_segment_sec:.0f}c)",
            flush=True,
        )

    first = dict(headers[0])
    first["_headers"] = [first.get("header", "")]
    merged: list[dict] = [first]

    for h in headers[1:]:
        prev = merged[-1]
        a = tokens_set(prev["header"])
        b = tokens_set(h["header"])
        ja = jaccard(a, b)
        common = a & b

        vote_jaccard = ja >= jacc_thresh
        vote_common = len(common) >= 2
        speaker_weight = 0  # 0 / 1 (weak) / 2 (strong)
        veto_speaker = False
        sp_prev = sp_cur = None
        sh_prev = sh_cur = 0.0

        boundary = float(h.get("start", h.get("win_start", prev["end"])))
        hard_split = _has_hard_split_cue(turns, boundary, boundary_context_sec)
        header_split = (
            is_non_talk_header(prev["header"]) != is_non_talk_header(h["header"])
            or is_non_talk_header(h["header"])
        )
        too_long = not _merge_candidate(prev, h, max_merged_segment_sec)

        if not has_diar:
            # Без спикеров главный сигнал — текст вокруг предполагаемой границы.
            # Если явного перехода нет, а блоки похожи на докладные, склеиваем
            # даже при слабом сходстве заголовков: модель часто меняет формулировку
            # внутри одного длинного доклада.
            should_merge = (
                not hard_split
                and not header_split
                and not too_long
                and (
                    vote_jaccard
                    or vote_common
                    or is_bad_header(prev["header"])
                    or is_bad_header(h["header"])
                    or (not is_non_talk_header(prev["header"]) and not is_non_talk_header(h["header"]))
                )
            )
        elif turns is not None:
            sp_prev, sh_prev = dominant_speaker_with_share(turns, prev["win_start"], prev["win_end"])
            sp_cur, sh_cur = dominant_speaker_with_share(turns, h["win_start"], h["win_end"])
            if sp_prev is not None and sp_cur is not None:
                if sp_prev == sp_cur:
                    if sh_prev >= speaker_share_thresh and sh_cur >= speaker_share_thresh:
                        speaker_weight = 2  # STRONG: один спикер уверенно в обоих окнах
                    else:
                        speaker_weight = 1  # WEAK: тот же топ-спикер, но доли ниже порога
                else:
                    if sh_prev >= speaker_share_thresh and sh_cur >= speaker_share_thresh:
                        veto_speaker = True  # явная смена докладчика

            votes = int(vote_jaccard) + int(vote_common) + speaker_weight
            should_merge = (
                votes >= votes_needed
                and not veto_speaker
                and not hard_split
                and not too_long
            )
        else:
            votes = int(vote_jaccard) + int(vote_common)
            should_merge = votes >= votes_needed and not hard_split and not too_long

        if debug:
            print(
                f"   [merge?] prev={fmt_tc(prev['win_start'])}-{fmt_tc(prev['win_end'])} "
                f"cur={fmt_tc(h['win_start'])}-{fmt_tc(h['win_end'])} | "
                f"jacc={ja:.2f}({int(vote_jaccard)}) common={len(common)}({int(vote_common)}) "
                f"sp_prev={sp_prev}/{sh_prev:.2f} sp_cur={sp_cur}/{sh_cur:.2f} "
                f"sp_w={speaker_weight} veto={veto_speaker} hard={hard_split} "
                f"header_split={header_split} too_long={too_long} votes={int(vote_jaccard) + int(vote_common) + speaker_weight} -> "
                f"{'MERGE' if should_merge else 'KEEP'}",
                flush=True,
            )

        if should_merge:
            prev.setdefault("_headers", [prev.get("header", "")])
            prev["_headers"].append(h.get("header", ""))
            prev["end"] = max(prev["end"], h["end"])
            prev["win_end"] = max(prev["win_end"], h["win_end"])
            prev["header"] = _choose_best_header(prev["_headers"])
        else:
            cur = dict(h)
            cur["_headers"] = [cur.get("header", "")]
            merged.append(cur)
    return merged


def fix_overlapping_segments(segments: list[dict]) -> list[dict]:
    """После merge соседние сегменты могут перекрываться (за-за прыжков модели
    по границам). Двигаемся по парам (i, i+1):
      • если seg[i+1].start < seg[i].end:
        - если перекрытие больше 50% длины короткого — склеим в один (оставим более длинный заголовок);
        - иначе — сдвинем границу: seg[i].end = seg[i+1].start.
    """
    if len(segments) <= 1:
        return segments
    out: list[dict] = [dict(segments[0])]
    for seg in segments[1:]:
        prev = out[-1]
        if seg["start"] < prev["end"]:
            overlap = prev["end"] - seg["start"]
            shorter = min(prev["end"] - prev["start"], seg["end"] - seg["start"])
            if shorter > 0 and overlap / shorter > 0.5:
                # сильное перекрытие — склеим
                prev["end"] = max(prev["end"], seg["end"])
                prev["win_end"] = max(prev.get("win_end", prev["end"]),
                                       seg.get("win_end", seg["end"]))
                if len(seg["header"]) > len(prev["header"]) and not is_meta_hallucination(seg["header"]):
                    prev["header"] = seg["header"]
                continue
            else:
                # лёгкое перекрытие — обрежем предыдущий
                prev["end"] = seg["start"]
        out.append(dict(seg))
    return out


# ============== этап 2: подсегменты -> тезисы ==============

def _split_by_pauses(subset: list, min_sec: float, max_sec: float,
                     target_sec: float, pause_thresh: float = 1.2) -> list[tuple[float, float]]:
    """Режем подсегменты по «большим» паузам между репликами WhisperX,
    целясь в ~target_sec, но всегда на границе реплик (не рвём фразы).
    Подходит для транскриптов без диаризации (один спикер у всех реплик).
    Алгоритм:
      1) Идём по репликам, накапливаем чанк.
      2) Когда длительность >= target_sec и между текущей и следующей репликой
         есть пауза >= pause_thresh — закрываем (это лучшее место для реза).
      3) Если длительность >= max_sec — принудительно закрываем на границе текущей реплики.
    """
    chunks: list[tuple[float, float]] = []
    if not subset:
        return chunks
    cur_start = subset[0].start
    last_added_end = subset[0].start  # конец последней «включённой» реплики
    for i, t in enumerate(subset):
        # Превышение max_sec ESCALATING: если добавление текущей реплики выведет
        # длительность за max_sec, и в чанке уже есть хотя бы 1 реплика — закрываем
        # предыдущим резом до текущей реплики.
        if last_added_end > cur_start and (t.end - cur_start) > max_sec:
            chunks.append((cur_start, last_added_end))
            cur_start = t.start
        last_added_end = t.end
        dur = t.end - cur_start
        # пауза до следующей реплики (или +inf в конце)
        gap = (subset[i + 1].start - t.end) if i + 1 < len(subset) else float("inf")
        if dur >= target_sec and gap >= pause_thresh:
            chunks.append((cur_start, t.end))
            cur_start = subset[i + 1].start if i + 1 < len(subset) else t.end
            last_added_end = cur_start
            continue
    # хвост
    last_end = subset[-1].end
    if cur_start < last_end:
        chunks.append((cur_start, last_end))

    # сливаем слишком короткие хвосты
    cleaned: list[tuple[float, float]] = []
    for ch in chunks:
        if cleaned and (ch[1] - ch[0]) < min_sec:
            cleaned[-1] = (cleaned[-1][0], ch[1])
        else:
            cleaned.append(ch)
    cleaned = [c for c in cleaned if c[1] - c[0] >= 10.0]
    return cleaned


def split_segment_for_theses(turns: list, seg_start: float, seg_end: float,
                             min_sec: float, max_sec: float) -> list[tuple[float, float]]:
    """
    Разбивает сегмент на подсегменты для генерации тезисов.

    Два режима:
      • Есть реальная диаризация (>=2 спикера): закрываем чанк по смене доминантного
        спикера (когда длительность в пределах [min_sec, max_sec]).
      • Диаризации нет (все SPEAKER_00): режем по большим паузам между репликами
        WhisperX (>= 1.2c), целясь в (min_sec+max_sec)/2 и всегда на границе реплик.
        Не рвём посреди предложения.
    """
    subset = [t for t in turns if t.end >= seg_start and t.start <= seg_end]
    if not subset:
        return []

    # Режим без диаризации — по паузам.
    if not _has_real_diarization(turns):
        target = (min_sec + max_sec) / 2.0
        return _split_by_pauses(subset, min_sec=min_sec, max_sec=max_sec,
                                target_sec=target, pause_thresh=1.2)

    # Режим с диаризацией — по смене доминирующего спикера.
    chunks: list[tuple[float, float]] = []
    cur_start = subset[0].start
    cur_speaker_time: dict[str, float] = {}
    last_end = subset[0].start
    for t in subset:
        dur_in_chunk = t.end - cur_start
        cur_speaker_time[t.speaker] = cur_speaker_time.get(t.speaker, 0.0) + (t.end - t.start)
        last_end = t.end
        if dur_in_chunk >= max_sec:
            chunks.append((cur_start, t.end))
            cur_start = t.end
            cur_speaker_time = {}
            continue
        if dur_in_chunk >= min_sec:
            total = sum(cur_speaker_time.values())
            if total > 0:
                top_spk, top_t = max(cur_speaker_time.items(), key=lambda x: x[1])
                if top_spk != t.speaker and (t.end - t.start) / max(total, 1e-6) > 0.4:
                    chunks.append((cur_start, t.end))
                    cur_start = t.end
                    cur_speaker_time = {}
                    continue
    if cur_start < last_end:
        chunks.append((cur_start, last_end))
    cleaned: list[tuple[float, float]] = []
    for ch in chunks:
        if cleaned and (ch[1] - ch[0]) < min_sec / 2.0:
            cleaned[-1] = (cleaned[-1][0], ch[1])
        else:
            cleaned.append(ch)
    cleaned = [c for c in cleaned if c[1] - c[0] >= 10.0]
    return cleaned


def build_thesis_prompt(turns_window: list, ts: float, te: float, seg_header: str,
                       tokenizer=None, max_input_tokens: int = 1950) -> str:
    window_text = "\n".join(
        f"[{fmt_tc(t.start)}-{fmt_tc(t.end)}] {t.speaker}: {t.text}"
        for t in turns_window
    )
    prompt = (
        "Задача: сформулируй один ключевой тезис, прозвучавший во фрагменте. "
        "Верни ровно одну строку формата 'MM:SS - MM:SS: <тезис>.' с указанным "
        "временным диапазоном. Тезис должен быть содержательным (что было сказано), "
        "без мета-речи и без вводных слов.\n\n"
        f"Заголовок сегмента: {seg_header}\n"
        f"Диапазон тезиса: {fmt_tc(ts)} - {fmt_tc(te)}\n\n"
        "Фрагмент транскрипта:\n"
        f"{window_text}"
    )
    if tokenizer is not None:
        head_part, body_part = prompt.split("Фрагмент транскрипта:\n", 1)
        prompt = prep._smart_truncate_window_lines(
            head_part, body_part.splitlines(), tokenizer, max_input_tokens
        )
    return prompt


def infer_theses_for_segment(turns: list, seg: dict, model_pack: TwoAdapterModel,
                             min_sec: float, max_sec: float,
                             dedup_thresh: float = 0.5) -> list[dict]:
    sub_ranges = split_segment_for_theses(turns, seg["start"], seg["end"], min_sec, max_sec)
    if not sub_ranges:
        return []
    adapter = "thesis" if model_pack.has_thesis_adapter else "header"
    theses: list[dict] = []
    seen_token_sets: list[set] = []
    n_meta_drop = 0
    n_dedup_drop = 0
    for ts, te in sub_ranges:
        subset = [t for t in turns if t.end >= ts and t.start <= te]
        if not subset:
            continue
        prompt = build_thesis_prompt(subset, ts, te, seg["header"],
                                     tokenizer=model_pack.tokenizer,
                                     max_input_tokens=1950)
        out = model_pack.generate(prompt, adapter_name=adapter, max_new_tokens=96)
        s, e, body = parse_model_line(out, ts, te)
        if not (ts - 5 <= s <= te + 5) or not (ts - 5 <= e <= te + 60):
            s, e = ts, te
        s = max(float(seg["start"]), min(float(s), float(seg["end"])))
        e = max(s, min(float(e), float(seg["end"])))
        if not body or len(body) < 8:
            continue
        # отсев мета-галлюцинаций (инструкция -> тезис)
        if is_meta_hallucination(body):
            n_meta_drop += 1
            continue
        # дедуп по Jaccard
        tk = tokens_set(body)
        is_dup = any(jaccard(tk, prev) >= dedup_thresh for prev in seen_token_sets)
        if is_dup:
            n_dedup_drop += 1
            continue
        seen_token_sets.append(tk)
        theses.append({"start": s, "end": e, "text": body})
    if n_meta_drop or n_dedup_drop:
        print(f"     [фильтр: meta={n_meta_drop}, dedup={n_dedup_drop}]", flush=True)
    return theses


# ============== этап 3: сборка Markdown ==============

def assemble_markdown(segments: list[dict], theses_by_seg: list[list[dict]],
                      *, theses_for_all: bool = False) -> str:
    lines: list[str] = []
    for seg, theses in zip(segments, theses_by_seg):
        header_line = f"{fmt_tc(seg['start'])} - {fmt_tc(seg['end'])}: {seg['header']}"
        lines.append(header_line)
        lines.append("")
        if not should_generate_theses(seg["header"], force_all=theses_for_all):
            continue
        lines.append("Важные моменты темы")
        if theses:
            for th in theses:
                t = th["text"].rstrip(".")
                lines.append(f"{fmt_tc(th['start'])} - {fmt_tc(th['end'])}: {t}.")
        else:
            lines.append("(тезисы не сформированы)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ============== main ==============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_header", required=True, type=Path,
                    help="Путь к LoRA-адаптеру для заголовков сегментов")
    ap.add_argument("--adapter_thesis", default=None, type=Path,
                    help="Путь к LoRA-адаптеру для тезисов (опционально)")
    ap.add_argument("--transcript", required=True, type=Path)
    ap.add_argument("--out_md", required=True, type=Path)
    ap.add_argument("--window_sec", type=float, default=720.0,
                    help="Длина скользящего окна (с) для генерации заголовков сегментов")
    ap.add_argument("--stride_sec", type=float, default=240.0,
                    help="Шаг скользящего окна (с); должен быть <= window_sec")
    ap.add_argument("--thesis_min_sec", type=float, default=60.0,
                    help="Минимальная длительность подсегмента для тезиса")
    ap.add_argument("--thesis_max_sec", type=float, default=180.0,
                    help="Максимальная длительность подсегмента для тезиса")
    ap.add_argument("--merge_jaccard", type=float, default=0.35,
                    help="Порог Jaccard для склейки соседних окон с похожими заголовками")
    ap.add_argument("--dedup_jaccard", type=float, default=0.5,
                    help="Порог Jaccard для дедупа тезисов внутри сегмента")
    ap.add_argument("--speaker_share", type=float, default=0.5,
                    help="Минимальная доля времени доминирующего спикера для STRONG-сигнала и вето")
    ap.add_argument("--votes_needed", type=int, default=2,
                    help="Сколько голосов нужно для склейки (jaccard=1, common=1, speaker=1 weak / 2 strong)")
    ap.add_argument("--boundary_context_sec", type=float, default=120.0,
                    help="Сколько секунд вокруг границы смотреть для поиска фраз перехода между докладами")
    ap.add_argument("--max_merged_segment_sec", type=float, default=1800.0,
                    help="Защита от чрезмерной склейки: максимальная длительность объединенного сегмента")
    ap.add_argument("--boundary_model", default=None, type=Path,
                    help="Optional boundary classifier path. If set, it finds real talk ranges before T5.")
    ap.add_argument("--boundary_model_step_sec", type=float, default=15.0)
    ap.add_argument("--boundary_model_context_sec", type=float, default=90.0)
    ap.add_argument("--boundary_start_threshold", type=float, default=0.45)
    ap.add_argument("--boundary_end_threshold", type=float, default=0.45)
    ap.add_argument("--boundary_inside_threshold", type=float, default=0.55)
    ap.add_argument("--boundary_min_talk_sec", type=float, default=180.0)
    ap.add_argument("--boundary_merge_gap_sec", type=float, default=45.0)
    ap.add_argument("--boundary_snap_sec", type=float, default=45.0)
    ap.add_argument("--boundary_rescue_long_gap_sec", type=float, default=600.0,
                    help="If a non-talk gap is longer than this, scan it again for missed talks.")
    ap.add_argument("--boundary_rescue_min_talk_sec", type=float, default=300.0)
    ap.add_argument("--boundary_rescue_window_sec", type=float, default=150.0)
    ap.add_argument("--boundary_rescue_score_threshold", type=float, default=1.6)
    ap.add_argument("--boundary_topic_merge_gap_sec", type=float, default=120.0,
                    help="Merge same-topic talk fragments separated by a short gap.")
    ap.add_argument("--boundary_device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no_boundary_gaps", action="store_true",
                    help="Do not add generic non-talk gaps between detected talks.")
    ap.add_argument("--debug_boundary", action="store_true")
    ap.add_argument("--theses_for_all", action="store_true",
                    help="Старый режим: выводить важные моменты даже для вопросов/вступлений/заключений")
    ap.add_argument("--debug_merge", action="store_true",
                    help="Печатать диагностику склейки по каждой паре соседних окон")
    ap.add_argument("--event_mode", default="auto",
                    choices=["auto", "standard", "panel", "closing"],
                    help="Жанровый режим: auto определяет панель/заключительное заседание по имени и началу транскрипта")
    ap.add_argument("--event_rescue_long_non_talk_sec", type=float, default=600.0,
                    help="В panel-режиме: искать пропущенные доклады внутри длинных non-talk блоков")
    ap.add_argument("--event_rescue_score_threshold", type=float, default=2.0,
                    help="Порог эвристического спасения докладов внутри non-talk блоков")
    ap.add_argument("--event_closing_split_long_sec", type=float, default=1500.0,
                    help="В closing-режиме: разбивать слишком длинные заключительные блоки")
    args = ap.parse_args()

    turns = prep.parse_transcript(args.transcript)
    if not turns:
        raise RuntimeError(f"Транскрипт пуст или формат не распознан: {args.transcript}")
    print(f">>> реплик: {len(turns)}, длительность {fmt_tc(turns[-1].end)}", flush=True)
    event_mode = detect_event_mode(args.transcript, turns, args.event_mode)
    print(f">>> event mode: {event_mode}", flush=True)

    boundary_ranges: list[tuple[float, float]] = []
    if args.boundary_model is not None:
        print(f">>> boundary model: {args.boundary_model}", flush=True)
        boundary_ranges = detect_talk_ranges_with_boundary_model(
            turns,
            args.boundary_model,
            step_sec=args.boundary_model_step_sec,
            context_sec=args.boundary_model_context_sec,
            start_threshold=args.boundary_start_threshold,
            end_threshold=args.boundary_end_threshold,
            inside_threshold=args.boundary_inside_threshold,
            min_talk_sec=args.boundary_min_talk_sec,
            merge_gap_sec=args.boundary_merge_gap_sec,
            snap_sec=args.boundary_snap_sec,
            rescue_long_gap_sec=args.boundary_rescue_long_gap_sec,
            rescue_min_talk_sec=args.boundary_rescue_min_talk_sec,
            rescue_window_sec=args.boundary_rescue_window_sec,
            rescue_score_threshold=args.boundary_rescue_score_threshold,
            topic_merge_gap_sec=args.boundary_topic_merge_gap_sec,
            device=args.boundary_device,
            debug=args.debug_boundary,
        )

    model_pack = load_two_adapter_model(args.base_model, args.adapter_header,
                                        args.adapter_thesis)

    # 1) окна -> заголовки
    if boundary_ranges:
        raw_headers = []
    else:
        raw_headers = infer_headers(turns, model_pack, args.window_sec, args.stride_sec)

    # 2) склейка соседних окон (votes-логика) + снятие перекрытий
    if args.debug_merge:
        print(">>> raw headers (до склейки):", flush=True)
        for rh in raw_headers:
            print(f"     {fmt_tc(rh['win_start'])}-{fmt_tc(rh['win_end'])}: {rh['header'][:100]}",
                  flush=True)
    segments = merge_adjacent_headers(raw_headers,
                                       jacc_thresh=args.merge_jaccard,
                                       turns=turns,
                                       speaker_share_thresh=args.speaker_share,
                                       votes_needed=args.votes_needed,
                                       debug=args.debug_merge,
                                       boundary_context_sec=args.boundary_context_sec,
                                       max_merged_segment_sec=args.max_merged_segment_sec)
    segments = fix_overlapping_segments(segments)
    if boundary_ranges:
        print(">>> using boundary ranges for talk segmentation", flush=True)
        segments = build_segments_from_boundary_ranges(
            turns,
            model_pack,
            boundary_ranges,
            include_gaps=not args.no_boundary_gaps,
        )
    segments = apply_event_mode_segments(
        segments,
        turns,
        model_pack,
        event_mode=event_mode,
        rescue_long_non_talk_sec=args.event_rescue_long_non_talk_sec,
        rescue_score_threshold=args.event_rescue_score_threshold,
        closing_split_long_sec=args.event_closing_split_long_sec,
        debug=args.debug_boundary or args.debug_merge,
    )
    print(f">>> сегментов после склейки: {len(segments)}", flush=True)
    for seg in segments:
        print(f"     {fmt_tc(seg['start'])}-{fmt_tc(seg['end'])}: {seg['header'][:100]}",
              flush=True)

    # 3) тезисы по сегментам
    theses_by_seg: list[list[dict]] = []
    for i, seg in enumerate(segments):
        if not should_generate_theses(seg["header"], force_all=args.theses_for_all):
            print(f">>> сегмент {i+1}/{len(segments)} без тезисов: {seg['header'][:100]}",
                  flush=True)
            theses_by_seg.append([])
            continue
        print(f">>> тезисы для сегмента {i+1}/{len(segments)}...", flush=True)
        theses = infer_theses_for_segment(
            turns, seg, model_pack,
            min_sec=args.thesis_min_sec, max_sec=args.thesis_max_sec,
            dedup_thresh=args.dedup_jaccard,
        )
        print(f"     -> {len(theses)} тезисов", flush=True)
        theses_by_seg.append(theses)

    # 4) сборка Markdown
    md = assemble_markdown(segments, theses_by_seg,
                           theses_for_all=args.theses_for_all)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"\nСохранено: {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
