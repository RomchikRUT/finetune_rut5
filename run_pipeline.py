"""
Единый пайплайн: аудио -> транскрипт (WhisperX) -> Markdown-аннотация (03).

Обёртка над:
  - 00_transcribe.py  (WhisperX large-v3 + alignment; диаризация опциональна)
  - 03_inference.py   (header/thesis LoRA-адаптеры -> Markdown)

Шаг 01 (подготовка датасета) и 02 (обучение QLoRA) — это OFFLINE-фаза,
которая делается один раз перед инференсом и здесь НЕ вызывается.
Для дообучения запускайте 01_prepare_dataset.py и 02_train_qlora.py
вручную (см. README в проекте).

Использование (PowerShell, Windows):

    cd E:\\finetune_rut5
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    $env:HF_TOKEN="hf_XXX"

    python -u run_pipeline.py `
      --audio          ".\\raw\\audio\\Секция 7 (14 октября).m4a" `
      --transcript_out ".\\raw\\transcripts\\Секция 7 (14 октября).txt" `
      --out_md         ".\\summaries\\Секция 7.md" `
      --base_model     utrobinmv/t5_summary_en_ru_zh_large_2048 `
      --adapter_header .\\checkpoints\\header\\best `
      --adapter_thesis .\\checkpoints\\thesis\\best

Если транскрипт уже есть — пропустите этап WhisperX:

    python -u run_pipeline.py `
      --transcript     ".\\raw\\transcripts\\Секция 7 (14 октября).txt" `
      --out_md         ".\\summaries\\Секция 7.md" `
      --base_model     utrobinmv/t5_summary_en_ru_zh_large_2048 `
      --adapter_header .\\checkpoints\\header\\best `
      --adapter_thesis .\\checkpoints\\thesis\\best
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _ensure_kmp_workaround() -> None:
    """Дублируем KMP_DUPLICATE_LIB_OK на случай если пользователь забыл."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _load_module_from_path(name: str, path: Path):
    """Импорт модуля с именем файла, содержащим цифры в начале (00_..., 03_...)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль из {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_transcribe(args) -> Path:
    """Шаг 0: WhisperX -> текстовый транскрипт."""
    transcribe_mod = _load_module_from_path("transcribe_step", HERE / "00_transcribe.py")
    out_txt = Path(args.transcript_out)
    transcribe_mod.transcribe_audio(
        audio_path=Path(args.audio),
        out_path=out_txt,
        hf_token=args.hf_token,
        model_name=args.whisper_model,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        language=args.language,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        diarize=args.diarize,
    )
    return out_txt


def run_inference(transcript_path: Path, args) -> None:
    """Шаг 3: 03_inference.py через subprocess (у него собственный CLI).

    Запускаем как отдельный процесс, потому что 03 загружает T5+LoRA и
    держит GPU долго; чистый процесс гарантирует освобождение VRAM от WhisperX.
    """
    cmd = [
        sys.executable, "-u", str(HERE / "03_inference.py"),
        "--base_model", args.base_model,
        "--adapter_header", str(args.adapter_header),
        "--adapter_thesis", str(args.adapter_thesis),
        "--transcript", str(transcript_path),
        "--out_md", str(args.out_md),
        "--window_sec", str(args.window_sec),
        "--stride_sec", str(args.stride_sec),
        "--thesis_min_sec", str(args.thesis_min_sec),
        "--thesis_max_sec", str(args.thesis_max_sec),
        "--merge_jaccard", str(args.merge_jaccard),
        "--dedup_jaccard", str(args.dedup_jaccard),
        "--speaker_share", str(args.speaker_share),
        "--votes_needed", str(args.votes_needed),
        "--boundary_context_sec", str(args.boundary_context_sec),
        "--max_merged_segment_sec", str(args.max_merged_segment_sec),
        "--event_mode", args.event_mode,
        "--event_rescue_long_non_talk_sec", str(args.event_rescue_long_non_talk_sec),
        "--event_rescue_score_threshold", str(args.event_rescue_score_threshold),
        "--event_closing_split_long_sec", str(args.event_closing_split_long_sec),
    ]
    if args.boundary_model is not None:
        cmd.extend([
            "--boundary_model", str(args.boundary_model),
            "--boundary_model_step_sec", str(args.boundary_model_step_sec),
            "--boundary_model_context_sec", str(args.boundary_model_context_sec),
            "--boundary_start_threshold", str(args.boundary_start_threshold),
            "--boundary_end_threshold", str(args.boundary_end_threshold),
            "--boundary_inside_threshold", str(args.boundary_inside_threshold),
            "--boundary_min_talk_sec", str(args.boundary_min_talk_sec),
            "--boundary_merge_gap_sec", str(args.boundary_merge_gap_sec),
            "--boundary_snap_sec", str(args.boundary_snap_sec),
            "--boundary_rescue_long_gap_sec", str(args.boundary_rescue_long_gap_sec),
            "--boundary_rescue_min_talk_sec", str(args.boundary_rescue_min_talk_sec),
            "--boundary_rescue_window_sec", str(args.boundary_rescue_window_sec),
            "--boundary_rescue_score_threshold", str(args.boundary_rescue_score_threshold),
            "--boundary_topic_merge_gap_sec", str(args.boundary_topic_merge_gap_sec),
            "--boundary_device", args.boundary_device,
        ])
    if args.no_boundary_gaps:
        cmd.append("--no_boundary_gaps")
    if args.debug_boundary:
        cmd.append("--debug_boundary")
    if args.theses_for_all:
        cmd.append("--theses_for_all")
    if args.debug_merge:
        cmd.append("--debug_merge")

    print(f">>> запуск инференса: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"03_inference.py завершился с кодом {rc}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Аудио -> Markdown-аннотация (единый пайплайн)")
    # --- источник входа ---
    ap.add_argument("--audio", type=Path, default=None,
                    help="Входной аудиофайл. Если не задан, требуется --transcript.")
    ap.add_argument("--transcript", type=Path, default=None,
                    help="Готовый транскрипт (пропустить WhisperX).")
    ap.add_argument("--transcript_out", type=Path, default=None,
                    help="Куда сохранить транскрипт из WhisperX. По умолчанию рядом с аудио.")

    # --- WhisperX ---
    ap.add_argument("--hf_token", default=None, help="HF_TOKEN для pyannote-диаризации")
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--compute_type", default="float16",
                    help="float16 или int8 (если мало VRAM)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--language", default=None,
                    help="ru/en/...; не задан = автодетект")
    ap.add_argument("--min_speakers", type=int, default=None)
    ap.add_argument("--max_speakers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--diarize", action="store_true",
                    help="Включить диаризацию pyannote. По умолчанию выключена.")
    ap.add_argument("--skip_transcribe", action="store_true",
                    help="Не запускать WhisperX (использовать --transcript как есть)")

    # --- 03 inference ---
    ap.add_argument("--out_md", required=True, type=Path)
    ap.add_argument("--base_model", required=True,
                    help="Например: utrobinmv/t5_summary_en_ru_zh_large_2048")
    ap.add_argument("--adapter_header", required=True, type=Path)
    ap.add_argument("--adapter_thesis", required=True, type=Path)
    ap.add_argument("--window_sec", type=float, default=720.0)
    ap.add_argument("--stride_sec", type=float, default=240.0)
    ap.add_argument("--thesis_min_sec", type=float, default=60.0)
    ap.add_argument("--thesis_max_sec", type=float, default=180.0)
    ap.add_argument("--merge_jaccard", type=float, default=0.35)
    ap.add_argument("--dedup_jaccard", type=float, default=0.5)
    ap.add_argument("--speaker_share", type=float, default=0.5)
    ap.add_argument("--votes_needed", type=int, default=2)
    ap.add_argument("--boundary_context_sec", type=float, default=120.0)
    ap.add_argument("--max_merged_segment_sec", type=float, default=1800.0)
    ap.add_argument("--boundary_model", default=None, type=Path)
    ap.add_argument("--boundary_model_step_sec", type=float, default=15.0)
    ap.add_argument("--boundary_model_context_sec", type=float, default=90.0)
    ap.add_argument("--boundary_start_threshold", type=float, default=0.45)
    ap.add_argument("--boundary_end_threshold", type=float, default=0.45)
    ap.add_argument("--boundary_inside_threshold", type=float, default=0.55)
    ap.add_argument("--boundary_min_talk_sec", type=float, default=180.0)
    ap.add_argument("--boundary_merge_gap_sec", type=float, default=45.0)
    ap.add_argument("--boundary_snap_sec", type=float, default=45.0)
    ap.add_argument("--boundary_rescue_long_gap_sec", type=float, default=600.0)
    ap.add_argument("--boundary_rescue_min_talk_sec", type=float, default=300.0)
    ap.add_argument("--boundary_rescue_window_sec", type=float, default=150.0)
    ap.add_argument("--boundary_rescue_score_threshold", type=float, default=1.6)
    ap.add_argument("--boundary_topic_merge_gap_sec", type=float, default=120.0)
    ap.add_argument("--boundary_device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no_boundary_gaps", action="store_true")
    ap.add_argument("--debug_boundary", action="store_true")
    ap.add_argument("--theses_for_all", action="store_true")
    ap.add_argument("--debug_merge", action="store_true")
    ap.add_argument("--event_mode", default="auto",
                    choices=["auto", "standard", "panel", "closing"])
    ap.add_argument("--event_rescue_long_non_talk_sec", type=float, default=600.0)
    ap.add_argument("--event_rescue_score_threshold", type=float, default=2.0)
    ap.add_argument("--event_closing_split_long_sec", type=float, default=1500.0)

    return ap.parse_args()


def main() -> None:
    _ensure_kmp_workaround()
    args = parse_args()

    # --- решаем, нужно ли транскрибировать ---
    if args.skip_transcribe or (args.transcript and not args.audio):
        if not args.transcript:
            raise SystemExit("--skip_transcribe требует --transcript")
        transcript_path = Path(args.transcript)
        if not transcript_path.exists():
            raise SystemExit(f"Транскрипт не найден: {transcript_path}")
        print(f">>> [0/2] WhisperX пропущен, использую готовый транскрипт: {transcript_path}",
              flush=True)
    else:
        if not args.audio:
            raise SystemExit("Нужен либо --audio, либо --transcript (+ --skip_transcribe)")
        # дефолт пути для транскрипта = рядом с аудио
        if args.transcript_out is None:
            args.transcript_out = args.audio.with_suffix(".txt")
        print(f">>> [0/2] WhisperX: {args.audio} -> {args.transcript_out}", flush=True)
        transcript_path = run_transcribe(args)

    print(f">>> [1/2] Инференс 03 -> {args.out_md}", flush=True)
    run_inference(transcript_path, args)
    print(f"\n>>> [2/2] Готово. Аннотация: {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
