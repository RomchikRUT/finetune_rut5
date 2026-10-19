# finetune_rut5

Локальная система для автоматической аннотации лекций и конференций. На вход подаётся аудиозапись или готовый транскрипт, на выходе формируется Markdown-документ с границами докладов, заголовками и ключевыми тезисами с таймкодами.

## Как работает система

1. **WhisperX** распознаёт аудио и выравнивает слова по времени. По умолчанию используется `large-v3`; диаризация спикеров через pyannote включается отдельно.
2. **ru-T5** с двумя отдельными LoRA-адаптерами формирует заголовки сегментов (`header`) и тезисы (`thesis`). Базовая модель: `utrobinmv/t5_summary_en_ru_zh_large_2048`.
3. **BERT-классификатор границ** на базе `cointegrated/rubert-tiny2` можно обучить и подключить для дополнительного определения границ докладов.
4. Скрипт инференса объединяет сегменты, удаляет похожие повторы и сохраняет результат в Markdown.

Обработка выполняется локальными моделями. При первом запуске веса моделей могут скачиваться из Hugging Face; после загрузки аудио не отправляется во внешний облачный API.

## Структура проекта

| Файл | Назначение |
| --- | --- |
| `00_transcribe.py` | Транскрибация аудио и выравнивание таймкодов через WhisperX |
| `01_prepare_dataset.py` | Подготовка JSONL-датасета для адаптеров заголовков и тезисов |
| `02_train_qlora.py` | Обучение адаптера `header` или `thesis` |
| `03_inference.py` | Построение аннотации из готового транскрипта |
| `04_prepare_boundary_dataset.py` | Подготовка датасета для классификатора границ |
| `05_train_boundary_classifier.py` | Обучение BERT-классификатора границ |
| `06_evaluate_metrics.py` | Оценка качества аннотации |
| `07_evaluate_transcription_metrics.py` | Оценка качества транскрипции |
| `run_pipeline.py` | Полный запуск: аудио → транскрипт → Markdown-аннотация |

> Имя `02_train_qlora.py` сохранено в репозитории, но скрипт обучает LoRA-адаптеры без `bitsandbytes` и 4-битной загрузки весов.

## Требования

- Python и PyTorch.
- Для инференса и обучения ru-T5-адаптеров в текущем коде нужна CUDA-совместимая видеокарта с поддержкой BF16: модель загружается на устройство CUDA с типом данных bfloat16.
- WhisperX, Transformers, PEFT, Accelerate и SentencePiece.
- ffmpeg для чтения аудиофайлов.
- Для диаризации: доступ к модели pyannote/speaker-diarization-3.1 на Hugging Face и токен Hugging Face.

Версии библиотек в репозитории не зафиксированы. Подберите PyTorch под конфигурацию CUDA и используйте совместимые версии остальных пакетов.

## Установка

```powershell
git clone https://github.com/RomchikRUT/finetune_rut5.git
cd finetune_rut5
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Установите зависимости после активации виртуального окружения. Выберите сборку PyTorch, соответствующую вашей версии CUDA:

```powershell
python -m pip install torch
python -m pip install whisperx transformers peft accelerate sentencepiece
```


Скрипты инференса и обучения задают совместимую настройку OpenMP автоматически. При необходимости перед запуском её можно задать вручную:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

Файлы весов `.safetensors` и `.bin` настроены на хранение через Git LFS. Если такие файлы используются в репозитории, установите Git LFS, выполните `git lfs install`, затем `git lfs pull`.

## Быстрый запуск

Для инференса нужны два обученных адаптера: `checkpoints/header/best` и `checkpoints/thesis/best`. Если их нет, сначала обучите адаптеры по инструкции ниже.

### Из аудиофайла

```powershell
New-Item -ItemType Directory -Force .\summaries | Out-Null

python -u run_pipeline.py `
  --audio ".\raw\audio\lecture.m4a" `
  --transcript_out ".\raw\transcripts\lecture.txt" `
  --out_md ".\summaries\lecture.md" `
  --base_model "utrobinmv/t5_summary_en_ru_zh_large_2048" `
  --adapter_header ".\checkpoints\header\best" `
  --adapter_thesis ".\checkpoints\thesis\best"
```

Поддерживаются распространённые аудиоформаты, включая M4A, MP3 и WAV. По умолчанию WhisperX определяет язык автоматически. Для транскрипции можно задать язык явно параметром `--language ru`. Если видеозапись уже содержит готовую аудиодорожку, предварительно извлеките её в аудиофайл.

### Из готового транскрипта

```powershell
New-Item -ItemType Directory -Force .\summaries | Out-Null

python -u run_pipeline.py `
  --transcript ".\raw\transcripts\lecture.txt" `
  --skip_transcribe `
  --out_md ".\summaries\lecture.md" `
  --base_model "utrobinmv/t5_summary_en_ru_zh_large_2048" `
  --adapter_header ".\checkpoints\header\best" `
  --adapter_thesis ".\checkpoints\thesis\best"
```

Транскрипт должен содержать по одной реплике на строку, с временными границами и меткой спикера:

```text
[12.34s -> 56.78s] SPEAKER_00: Текст распознанной реплики.
```

### Необязательная диаризация

Чтобы включить разметку реплик по спикерам, примите условия использования модели pyannote на Hugging Face, задайте токен и добавьте флаг `--diarize` к команде запуска:

```powershell
$env:HF_TOKEN = "hf_ваш_токен"
```

Токен не следует записывать в README или добавлять в Git. Без флага `--diarize` скрипт назначает репликам общий идентификатор `SPEAKER_00`.

## Подготовка данных и обучение

Данные для обучения не входят в репозиторий. Скрипты ожидают пары файлов с одинаковым именем в каталогах транскриптов и эталонных аннотаций, например:

```text
raw/
├── transcripts/
│   └── lecture.txt
└── summaries/
    └── lecture.md
```

Файл транскрипта содержит строки вида `[12.34s -> 56.78s] SPEAKER_00: ...`. В эталонной аннотации задаются интервалы и заголовки докладов, а также раздел `Важные моменты темы` с интервалами и текстом тезисов. Пример строки аннотации:

```text
00:00 - 04:30: Название доклада

Важные моменты темы
00:15 - 00:50: Первый ключевой тезис.
```

Подготовьте общий датасет для двух задач:

```powershell
python 01_prepare_dataset.py `
  --transcripts .\raw\transcripts `
  --summaries .\raw\summaries `
  --out .\data\dataset.jsonl `
  --tokenizer "utrobinmv/t5_summary_en_ru_zh_large_2048" `
  --max_input_tokens 1500 `
  --context_sec 10
```

Обучите отдельный адаптер для заголовков и отдельный адаптер для тезисов:

```powershell
python 02_train_qlora.py `
  --base_model "utrobinmv/t5_summary_en_ru_zh_large_2048" `
  --dataset .\data\dataset.jsonl `
  --output .\checkpoints\header `
  --task header `
  --epochs 25

python 02_train_qlora.py `
  --base_model "utrobinmv/t5_summary_en_ru_zh_large_2048" `
  --dataset .\data\dataset.jsonl `
  --output .\checkpoints\thesis `
  --task thesis `
  --epochs 25
```

После обучения скрипт сохраняет лучшие версии адаптеров в `checkpoints/header/best` и `checkpoints/thesis/best`.

### Дополнительный классификатор границ

Классификатор различает начало доклада, конец доклада, фрагмент внутри доклада и недокладный фрагмент. Его можно подготовить и обучить отдельно:

```powershell
python 04_prepare_boundary_dataset.py `
  --transcripts .\raw\transcripts `
  --summaries .\raw\summaries `
  --out .\data\boundary_dataset.jsonl

python 05_train_boundary_classifier.py `
  --base_model "cointegrated/rubert-tiny2" `
  --dataset .\data\boundary_dataset.jsonl `
  --output .\checkpoints\boundary `
  --epochs 8 `
  --device cuda
```

Чтобы использовать обученную модель при инференсе, передайте каталог классификатора через `--boundary_model` в `run_pipeline.py`. Параметр необязателен: базовый пайплайн работает без него.

## Результаты

В презентации к выпускной квалификационной работе приведены следующие показатели экспериментального прогона:

| Метрика | Значение |
| --- | ---: |
| F1 докладов | 62,5% |
| Покрытие докладов | 76,0% |
| Покрытие тезисов | 71,2% |
| chrF тезисов | 55,3% |

Это результаты конкретного прогона и набора данных, а не гарантия качества на произвольных аудиозаписях. Для собственной выборки используйте скрипты `06_evaluate_metrics.py` и `07_evaluate_transcription_metrics.py`.

## Ограничения и примечания

- Ошибки распознавания речи могут повлиять на сегментацию и качество тезисов; итоговую аннотацию рекомендуется проверять.
- Большие модели требуют значительных ресурсов. Для снижения нагрузки WhisperX поддерживает `--compute_type int8`; также можно уменьшить `--batch_size`.
- Веса базовых моделей загружаются из Hugging Face при первом запуске, если отсутствуют в локальном кеше. После загрузки моделей обработка выполняется локально.
- Каталоги `raw/`, `data/`, `summaries/` и `metrics/` исключены из Git. Храните исходные аудиозаписи, персональные данные и локальные результаты в соответствии с применимыми правилами доступа и распространения.


