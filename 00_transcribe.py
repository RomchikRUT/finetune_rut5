"""
Шаг 0: аудио -> текстовый транскрипт в формате, совместимом с 01/02/03.

Использует WhisperX (large-v3 + word-level alignment; диаризация pyannote опциональна).
Выход — текстовый файл со строками:

    [12.34s -> 56.78s] SPEAKER_00: текст реплики

Именно этот формат ожидает `prep.parse_transcript()` в 01_prepare_dataset.py
(RE_TURN на строке 47-49), поэтому его потребляют без правок 01 и 03.

ВАЖНО:
  - На Windows перед запуском обязательно:
      $env:KMP_DUPLICATE_LIB_OK="TRUE"
  - bitsandbytes НЕ используется.
  - HuggingFace токен нужен только при включении диаризации pyannote/speaker-diarization-3.1
    (требуется один раз принять пользовательское соглашение на странице модели).
    Токен можно передать через --hf_token или переменную окружения HF_TOKEN.

Пример:
    python 00_transcribe.py ^
        --audio   ".\raw\audio\Секция 7 (14 октября).m4a" ^
        --out_txt ".\raw\transcripts\Секция 7 (14 октября).txt" ^
        --hf_token hf_XXX
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch


def format_turn_line(start: float, end: float, speaker: str, text: str) -> str:
    """Формирует строку в формате, который ожидает 01/03."""
    speaker = speaker or "SPEAKER_00"
    text = (text or "").strip().replace("\n", " ")
    return f"[{start:.2f}s -> {end:.2f}s] {speaker}: {text}"


def transcribe_audio(
    audio_path: Path,
    out_path: Path,
    *,
    hf_token: str | None,
    model_name: str = "large-v3",
    compute_type: str = "float16",
    batch_size: int = 16,
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "cuda",
    diarize: bool = False,
) -> Path:
    """Транскрибирует audio_path и сохраняет результат в out_path.

    Возвращает путь к сохранённому транскрипту.
    """
    # Импорт здесь, чтобы 00_transcribe.py можно было импортировать как модуль
    # без падения, если whisperx не установлен на текущей машине.
    import whisperx

    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 1. Whisper large-v3 ---
    print(f">>> [1/3] Загрузка модели Whisper '{model_name}'...", flush=True)
    model = whisperx.load_model(model_name, device, compute_type=compute_type)

    print(">>> загрузка аудио...", flush=True)
    audio = whisperx.load_audio(str(audio_path))

    print(">>> транскрипция...", flush=True)
    if language:
        result = model.transcribe(audio, batch_size=batch_size, language=language)
    else:
        result = model.transcribe(audio, batch_size=batch_size)
    detected_lang = result["language"]
    print(f">>> обнаружен язык: {detected_lang}", flush=True)

    # Освобождаем VRAM до загрузки align-модели
    del model
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. Word-level alignment ---
    print(">>> [2/3] Выравнивание (word-level timestamps)...", flush=True)
    align_model, align_meta = whisperx.load_align_model(
        language_code=detected_lang, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, align_meta, audio, device,
        return_char_alignments=False,
    )
    del align_model
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3. Диаризация (опционально) ---
    final_result = result
    if diarize:
        print(">>> [3/3] Диаризация (pyannote)...", flush=True)
        hf_token_eff = os.environ.get("HF_TOKEN")
        if not hf_token_eff:
            raise RuntimeError(
                "Для диаризации нужен HF_TOKEN. Передайте --hf_token или установите "
                "переменную окружения HF_TOKEN (и примите соглашение модели "
                "pyannote/speaker-diarization-3.1 на huggingface.co)."
            )
        # В разных версиях whisperx сигнатура немного отличается; поддержим обе.
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=hf_token_eff, device=device
            )
        except TypeError:
            diarize_model = whisperx.diarize.DiarizationPipeline(device=device)

        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        final_result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        print(">>> [3/3] Диаризация отключена — всем репликам будет присвоен SPEAKER_00",
              flush=True)

    # --- Сохранение в формате 01/03 ---
    print(f">>> запись транскрипта: {out_path}", flush=True)
    lines: list[str] = []
    for seg in final_result["segments"]:
        speaker = seg.get("speaker") or "SPEAKER_00"
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(format_turn_line(start, end, speaker, text))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f">>> готово: {len(lines)} реплик, {out_path}", flush=True)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="WhisperX -> текстовый транскрипт для 01/03")
    ap.add_argument("--audio", required=True, type=Path,
                    help="Входной аудиофайл (m4a/mp3/wav/...)")
    ap.add_argument("--out_txt", required=True, type=Path,
                    help="Куда сохранить транскрипт в формате 01/03")
    ap.add_argument("--hf_token", default=None,
                    help="HuggingFace токен (или установите HF_TOKEN)")
    ap.add_argument("--model", default="large-v3",
                    help="Имя модели WhisperX (по умолчанию large-v3)")
    ap.add_argument("--compute_type", default="float16",
                    help="float16 / int8 (для слабой VRAM возьмите int8)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--language", default=None,
                    help="Принудительно задать язык (ru/en/...); если не задан — авто")
    ap.add_argument("--min_speakers", type=int, default=None)
    ap.add_argument("--max_speakers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diarize", action="store_true",
                    help="Включить диаризацию pyannote. По умолчанию выключена.")
    args = ap.parse_args()

    transcribe_audio(
        audio_path=args.audio,
        out_path=args.out_txt,
        hf_token=args.hf_token,
        model_name=args.model,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        diarize=args.diarize,
    )


if __name__ == "__main__":
    main()
