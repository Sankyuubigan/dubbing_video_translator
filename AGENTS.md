# Documentation

## ОБЯЗАТЕЛЬНО: Чтение глобальной документации (core §3.1)

Перед любой работой над этим проектом ОБЯЗАТЕЛЬНО прочитать:

1. `D:\Projects\docusaurus-starter\docs\Sega Mega Note\Моя картотека\software\настройки\global_ai_docs\core\rules.md` — базовые правила (универсальные, для всех проектов)
2. `D:\Projects\docusaurus-starter\docs\Sega Mega Note\Моя картотека\software\настройки\global_ai_docs\desktop_rust_tauri\rules.md` — правила для Rust + Tauri

### Ключевая выжимка, обязательная к соблюдению
- **Git**: любые git-операции (commit/push/add/checkout/...) — ТОЛЬКО с явного письменного разрешения пользователя. Разрешены только `git diff/status/log`.
- **Временные файлы и логи**: ТОЛЬКО в проектных папках `temp/`, `test/` или `target/`. Системный `%TEMP%` и корень репо — запрещены. Лог последней сессии — `test/last_logs.txt` (пишет `tauri-plugin-logs`).
- **Поиск корня бага**: запрещено чинить симптомы/костылями, только root cause. Догадка ≠ причина: проверять кодом/логами/тестами.
- **Поиск в интернете**: только keyless-сервисы (DuckDuckGo/Wikipedia/...). Использование API-ключей (Tavily/Exa/Brave/Bing/SerpAPI) — ЗАПРЕЩЕНО (core §2.6).
- **Крупные задачи**: обязателен план-файл в `tasks/` с именем `ДД.ММ.ГГ <название>.md` и чекбоксами (core §1.8).
- **Сборка/тесты**: НЕ вызывать `cargo`/`npx tauri` напрямую — только через `.bat`-обёртки из папки проекта (desktop §2).

## Stack
- **Backend:** Rust + Tauri 2.0
- **LLM (перевод):** плагин `tauri-plugin-llama-engine` — инференс отдельным процессом `llama-server.exe` (desktop §6.7). Нативного `llama-cpp-2` в проекте НЕТ.
- **Речь (STT/TTS):** плагин `tauri-plugin-speech` (CrispASR)
- **Логи:** плагин `tauri-plugin-logs` (core §2.5.2). Свой логгер в хосте запрещён.
- **Frontend:** React/TypeScript
- **Media:** FFmpeg (external, audio extraction + subtitle muxing)
- **Модели:** Gemma-4-12B GGUF (перевод), CrispASR parakeet (STT), CosyVoice3 (TTS)

## Architecture
Modules are isolated — they don't import each other directly. All communication goes through `comm.rs` (PipelineContext hub).

```
Pipeline order (DON'T CHANGE):
1. audio_extractor — extract WAV from video via FFmpeg
2. diarization — CrispASR one-pass: parakeet ASR + VAD + speaker diarization (returns speaker segments AND transcript)
3. stt — takes the ready transcript from diarization (does NOT re-run the engine)
4. translation — LLM translation (Gemma-4-12B GGUF через плагин `tauri-plugin-llama-engine`)
5. verification — проверка изохронии/мусора, ретраи через тот же движок
6. tts — CrispASR CosyVoice3
7. output — burn subtitles into video via FFmpeg
```

Каждый LLM-этап сам поднимает и роняет движок (`llama-server.exe`): при выходе
из этапа процесс убивается и VRAM освобождается до следующего (desktop §6.5).
`pipeline.rs` о движке ничего не знает.

**No Silero VAD module and no sherpa-onnx fallback diarization** — CrispASR is the single source of truth for ASR+diarization. If CrispASR gives no segments, the pipeline fails with a clear error (no silent fallback).

## Translation Module (`src-tauri/src/translation/mod.rs`)

### Model
Выбирается пользователем в **выпадающем списке на главной вкладке**
(«Модель перевода») либо в **Настройках → «Локальные модели перевода»**
(`<llama-models-panel>` плагина). Оба пути пишут в конфиг плагина
(`%APPDATA%\com.deedub.desktop\app_config.json`, ключ `last_model`) — это
единственный источник правды (core §2.1). Своего «выбранной модели» в
`config.toml` нет намеренно: два места правды разъедутся, и UI станет врать.
Движок ставится и обновляется плашкой «Движок перевода (LLM)»
(`<llama-engine-panel>`), папку можно сменить там же («Изменить путь»).

Поддерживаются **два семейства**, и разница между ними не в качестве, а в
формате промпта (см. «Prompt format»):

| Модель | Файл | Размер | Формат промпта | translate (36 чанков) |
|---|---|--:|---|--:|
| **Index-Translate-9B** (рекомендуется) | `D:\nn\models\translation\Index-Translate-9B\Index-Translate-9B.IQ4_XS.gguf` | 5.0 GiB | `instTrans` | 18–19 с |
| Gemma-4-12B (baseline) | `D:\nn\models\llm\uncen\gemma-4-12B\gemma-4-12B-it-qat-q4_0-unquantized-heretic-ja-v2.i1-IQ4_NL.gguf` | 6.5 GiB | `chat` | 32–33 с |

- **Index-Translate-9B:** `Index-Translate-9B` (qwen35), IQ4_XS, IFscore 0.821,
  WMT24++ 0.8601, FLORES 0.8789. Думатель выключен, `enable_thinking: false`.
  Декод ~99 tok/s, полный оффлоад на 16 ГБ (пик VRAM 6.5 ГиБ).
- **Gemma-4-12B:** `google/gemma-4` (Gemma4ForCausalLM), 12B dense (not MoE),
  IQ4_NL, Vocab 256000 SPM (no BPE Cyrillic bug).
- **Context window:** 8192 токенов (`llm.rs: CONTEXT_SIZE`) у обеих.
- В GGUF есть `tokenizer.chat_template` — движок рендерит промпт сам.
- **Index-Translate-2B** (`D:\nn\models\llm\index-translate\Index-Translate-2B.Q4_K_M.gguf`)
  быстрее всех (11 с), но теряет 3 чанка из 37 на проверке цифр — как основная
  модель не годится.

### Prompt format
Раньше промпт собирался вручную в теги Gemma-4 (`<|turn>…<|channel>thought`).
**Этого кода нет**: хост отдаёт обычные роли, а промпт рендерит сам движок по
`tokenizer.chat_template` модели.

**Формат выбирается явно, и это не косметика.** У моделей разные «родные»
формы, и промпт, придуманный под одну, для другой выходит из распределения:
Index-Translate в нашем chat-промпте дублировал соседние реплики и переводил
текст из lookahead вместо текущего (воспроизводилось при temperature 0 и 0.6,
то есть это формат входа, а не сэмплирование).

- `PromptStyle::Chat` — наш `system` + `user` с блоками «Previous context»
  (8 пар EN/RU) и «Future context» (3 EN). Проверен на Gemma-4-12B.
- `PromptStyle::InstTrans` — канонический формат семейства Index-Translate
  (`inference/llm/translate.py`, функция `trans_prompt`): исходник в
  огороженном блоке `Source text:`, требования нумерованным списком с метками
  `[hard]`/`[soft]`, в конце запрет на пояснения. **Без** блоков контекста.
- Приоритет: env `DEEDUB_LLM_PROMPT` → настройка `prompt_style` (`auto` по
  умолчанию) → определение по имени файла модели. Env первым, иначе
  `run_mt_ab.bat` / `run_pipeline_mt.bat` не смогли бы сравнивать модели.
- Значение приходит в `PipelineConfig::prompt_style` и читается и переводом, и
  верификацией **из одного места**, а не разрешается дважды: разойтись они не
  могут, а разойтись могли — и тогда ретраи собирали бы промпт в чужом формате.

Промпт `chat` (Gemma, проверен):
```
system:
You are an expert audiovisual translator adapting English video subtitles into Russian for voiceover.

STRICT RULES:
1. CONTEXT & ASR FIXES: ...
2. ISOCHRONY (TIMING): ...
3. NUMBERS TO WORDS: ... (TTS engine cannot read digits)
4. METRIC: ...
5. GENDER: Current speaker is [SPEAKER_GENDER]. ...
6. NO CENSORSHIP: ...
7. OUTPUT FORMAT: Output ONLY the final spoken Russian text. Do NOT think out loud, do NOT analyze, do NOT offer options or variants. ...

user:
Previous context:
EN: prev text
RU: prev translation

Future context (DO NOT translate yet):
EN: next text

Translate this current text:
EN: source text
```

- Контекст: до 8 предыдущих пар (EN, RU) + до 3 lookahead (только EN)
- Source-side контекст (EN) важнее целевого — модель чинит ASR-ошибки по нему
- **Думатель выключен**: `REASONING_BUDGET = 0` + `disable_reasoning = true`, движок
  получает `chat_template_kwargs {"enable_thinking": false}` и поднимается без
  `--reasoning-*` флагов. Модель отдаёт перевод сразу в `GenerationResult.text`
- `clean_output` прогоняет ответ через пост-обработку: CoT-блоки `<channel|>`,
  обрезка до первой кириллицы, префиксы («Russian:», «Перевод:»), хвост латиницы

### Верификация (`src-tauri/src/verification/mod.rs`)
- Проверок в `assess` — 14. Тринадцать из них про текст, и только одна про цифры:
  `digits_present` добавлена после того, как выяснилось, что правило «пиши числа
  словами» было прописано и в промпте, и в ретрае, но **не проверялось нигде**:
  «Сезон 10» и «9.5» доезжали до SRT, где CosyVoice3 цифры не читает.
- **Ретрай получает список дефектов.** Раньше `build_strict_messages` его не
  принимал: модель просто переспрашивала то же самое при меньшей температуре,
  и детерминированно воспроизводила ту же ошибку — на прогоне это давало
  `RETRY=0 FAIL=2`. Сейчас коды `assess` переводятся в человеческие требования
  (`issue_hint`) и уходят в промпт как `[hard, fix] …`, а перед каждой попыткой
  список пересобирается из фактического текста последней генерации.
- Формат промпта ретрая **тот же**, что у основного перевода, и определяется
  один раз на весь проход. Ретрай в чужом формате починил бы цифры и сломал
  остальное (Index вне instTrans начинает дублировать соседние чанки).
- Изохрония проверяется по символам (`CHARS_PER_SEC_MAX = 22` + допуск 10,
  чанки короче `MIN_LEN_FOR_BUDGET = 10` не проверяются), тогда как планировщик
  TTS считает по гласным (`гласные / 4.4`). Критерии не сверены — это известное
  расхождение, а не баг конкретного прогона.
- Измеренная изохрония (гласные/4.4 ÷ длительность): Gemma-4-12B **1.70**,
  Index-9B + instTrans **1.81**, Index-2B + instTrans 1.83. То есть переполнение
  окон есть у обеих моделей, Index-9B чуть сильнее.

### Цензура слов: `censored_symbol` и `sanitize_for_output`
Index-Translate-9B маскирует мат звёздочками: чанк с «F you broke cheaters»
приходил как `Бл*ть, обманщики…`. Наблюдалось **только на 9B** — 2B и Gemma в
том же тесте не дали ни одной звёздочки, и воспроизводилось одинаково в chat-
и в instTrans-промпте, то есть формат промпта ни при чём. Механизм (safety
alignment) не доказан, но и не нужен: чинится тремя независимыми слоями.

1. **Промпт** (оба формата): требование писать слово целиком, в обычных буквах,
   и прямо объясняет, зачем — «the speech engine cannot pronounce symbols».
2. **Верификация**: `assess` ловит слово с одиночной `*` между буквами →
   `censored_symbol` → `issue_hint` просит написать слово полностью. Проверка
   `**` на markdown это не ловила.
3. **Санитар** `comm::sanitize_for_output` (вызывается из
   `translation::clean_output`): выбрасывает замаскированное слово, потому что
   восстановить заменённую букву детерминированно нельзя. Правильное слово
   приходит с ретрая; санитар — страховка, чтобы `*` физически не дошёл до
   CosyVoice3, где небуквенный токен рвёт слово.

**Известная граница фильтра:** маска из двух звёздочек (`f**k`) под `**` не
попадает — в проекте `**` считается markdown. Наблюдалась только одиночная
маска. Словарь «`бл*ть` → `блять»` сознательно не делаем: он покрыл бы только
известные маски, и первое же новое ругательство снова поехало бы в озвучку.

Проверяется `build_test_censor_filter.bat` (бинарник, т.к. харнесс `cargo test`
на этой машине не стартует) и 9 unit-тестов в `translation::tests` /
`verification::tests`. Санитар лежит в `comm`, а не в `translation`, чтобы
тест-бинарник звал production-функцию, а не свою копию.

### `SKIP_MARKER` — служебная метка «чанк выкинут» (3 потребителя)
Чанк, не прошедший верификацию после 2 ретраев, получает
`text = SKIP_MARKER` (`"(-)"`, `comm.rs:5`). Это **внутренняя метка, а не
текст для показа**, и её обязаны уважать все, кто читает `chunk.text`:

| Потребитель | Проверка | Смысл |
|---|---|---|
| `verification::assess` | `trimmed == SKIP_MARKER → без дефектов` | метка сама не считается браком |
| `tts` | `text == SKIP_MARKER → continue` | молчит в озвучке |
| `output::build_srt_content` | `text == SKIP_MARKER → continue` | не занимает экран |

Раньше `write_srt` проверял только `is_empty()`, и маркер **уезжала в
пользовательский SRT**: на экране 11 секунд висело `(-)`. Замер (04.10.26,
Index-9B, chunk 36 = `FAIL` по `9.5`): RU SRT содержал 37 cue, из них
последний — `[Speaker_1] (-)`; после правки 36 cue, маркера 0, нумерация
сплошная.

Фильтр живёт в чистой функции `output::build_srt_content` (без файлов и
побочных эффектов), а `write_srt` только пишет её результат на диск — так
тест-бинарник проверяет production-код. `idx` растёт **после** `continue`,
поэтому пропуск не оставляет дыры в нумерации.

Проверяется `build_test_srt_filter.bat` (бинарник) и 5 unit-тестов в
`output::tests`.

**Не путать с отказом от FAIL.** `SKIP_MARKER` означает «молча и без
субтитра», а не «показать исходник». Если чанк падает регулярно (как 36-й с
`9.5`), это проблема верификации, а не формата вывода — метку нельзя
разворачивать в фолбэк на английский текст молча.

### Параметры сэмплинга (`src-tauri/src/llm.rs`)
Нативных сэмплеров больше нет — параметры уходят в `llama-server` как параметры запроса (`ModelParams` плагина). База берётся из самого GGUF (`tokenizer.ggml.*`), сверху накладываются значения проекта:
1. `temperature = 0.6`
2. `top_k = 64`
3. `top_p = 0.95`
4. `min_p = 0.0`
5. `repetition_penalty = 1.05`
6. `presence_penalty = 0.0`, `dry_* = 0`, `xtc_* = 0` (раньше не использовались)

Фиксированного seed больше нет: `llama-server` получает `seed: -1` (случайный).

### Генерация
- Контекст: 8192 токенов (`llm.rs: CONTEXT_SIZE`), передаётся движку при старте
- Бюджет: `n_predict = MAX_ANSWER_TOKENS` (512) — **константный потолок**, не функция
  от промпта (`llm.rs: generate`)
  - Резерв «на размышления» не нужен: думатель выключен (см. Prompt format)
  - Потолок не влияет на сэмплирование: он ограничивает только момент остановки,
    а она и так по EOS (замер: средний ответ 22 токена, `stop_reason=EOS`)
  - Раньше бюджет считался как `clamp(prompt_tokens/2, 64, 512)`, но точное число
    токенов бралось отдельным HTTP-запросом `POST /tokenize` на КАЖДЫЙ чанк. Теперь
    вместимость промпта в контекст проверяется по **консервативной верхней оценке
    из символов** (`prompt_tokens_upper_bound`: токен не короче символа + 32 токена
    на сообщение — ложное «поместится» невозможно), а фактическое число токенов
    сверяется по `result.metrics.prompt_tokens` из ответа движка.
- Замер (ролик 192 с / 38 чанков, RTX 4070 Ti SUPER, полный оффлоад, 03.10.26):
  - `translate: 32.4s`, `verify: 2.5s` (OK=36 RETRY=2 FAIL=0), `tts: 52.1s`,
    **TOTAL 99.6s** (было 177.5s — см. `tasks/03.10.26 Ускорение пайплайна дубляжа.md`)
- **Замер с Index-Translate-9B + instTrans** (37 чанков, RTX 4070 Ti SUPER,
  тот же тест, 04.10.26): `translate: 18.2s`, `verify: 1.8s`
  (**OK=35 RETRY=2 FAIL=0**), `tts: 52.1s`, **TOTAL 83.9s**. То есть
  перевод в 1.8 раза быстрее Gemma при том же времени озвучки и без потери
  контента.
- **Почему думатель выключен:** при бюджете 1500 модель писала 600-1500 токенов
  рассуждений на каждый чанк (на «Bam. Right now.» — 611 токенов думателя) →
  25-30 с на реплику, 238 чанков ≈ 100 минут. Выигрыша в качестве не было:
  ответ одинаковый, а ошибки распознавания чинит source-side контекст, а не
  длинный внутренний монолог. Старый нативный код тоже «думал», но стоп по
  закрытию `<channel|>` ограничивал весь вывод 64-512 токенами
- Отмена пайплайна передаётся в движок напрямую (`Arc<AtomicBool>` в
  `generate_chat`), так что отмена работает и на середине генерации

### Одна LLM-сессия на translate + verify
`translate_with_session` возвращает живой `LlmSession` наружу, `verify_with_session`
принимает его и уничтожает на выходе (`pipeline.rs`). Раньше `translate` ронял
сессию на границе фаз, а `verify` поднимал **второй** `llama-server.exe` (~5 с
загрузки 6.5 ГиБ в VRAM) ради десятка запросов по бракованным чанкам.
Сессия умирает строго ДО старта TTS — VRAM освобождается по desktop §6.5.

## TTS Module (`src-tauri/src/tts/mod.rs`)

### Модель по умолчанию: `cosyvoice3-tts-rl` (RL)
- **Файл LLM:** `D:\nn\models\tts\cosyvoice3-tts-rl\cosyvoice3-llm-rl-q4_k.gguf` (~366 МБ)
- **Прочее (общие GGUF в той же папке):** flow-q8_0 + campplus-f16 + s3tok-f16 + hift-f16 + voices.gguf
- Base-версия `cosyvoice3-tts` (т.е. `cosyvoice3-llm-q4_k.gguf`) УДАЛЕНА с диска и из `speech_models.json` — она глючила (обрезание финальных слогов в cross-lingual режиме). RL-постобучение стабильнее: на тесте 38 чанков обрезаний нет.
- Выбор модели: env `DEEDUB_TTS_PRESET` (приоритет, для headless A/B) → `tts_settings.json` → `preset` → дефолт `cosyvoice3-tts-rl`.
- **Пресеты — ЕДИНЫЙ источник: `tauri-plugin-speech/speech_models.json`.** Плагин подключается через path-зависимость (`Cargo.toml:42`), его `include_str!` компилируется из этой папки при каждой сборке. `src-tauri/speech_models.json` УДАЛЁН (и из `tauri.conf.json` resources тоже) — внешний файл рядом с exe больше не создаётся. Любая правка пресетов делается только в плагине (`D:\Projects\my-tauri-plugins\tauri-plugin-speech\speech_models.json`).

### Retry-костыль и declick (только для base `cosyvoice3-tts`)
- Историческая причина: base-модель в cross-lingual (EN-голос → RU-текст) выбрасывает LM reference-tokens и «обрезает» финальные слоги; RAS-семплер seed-зависим и даёт щелчки.
- Retry-механизм: `RetryTracker` + `TTS_RETRY_SEEDS` + `TTS_FULL_FLOOR=0.62` — повторы с разными seed, пока ДВЕ попытки не сойдутся на полноте ≥ 0.62×ожидаемой длительности (`expected_spoken_secs` по числу гласных, темп 4.4 гласных/сек).
- Деклик: `declick_spikes` — сглаживание 1-сэмпловых выбросов в финальных сэмплах (O(n), без перегенераций). Клики — артефакт base (фикстуры ч.3/ч.27 сняты с её прогонов 17.09 01:58, до установки RL).
- **Оба включаются строго по бэкенду:** `attempt_count = if tts_backend == "cosyvoice3-tts" { TTS_MAX_ATTEMPTS } else { 1 }`, `if tts_backend == "cosyvoice3-tts" { declick_spikes(...) }`.
- Сейчас пресета `cosyvoice3-tts` в JSON нет → оба костыля фактически выключены. Оставлены в коде на будущее: если base вернётся в плагинский `speech_models.json`, retry и declick восстановятся автоматически. RL синтезирует 1 попыткой (seed=None) и без кликов → костыли для неё — zero overhead.

### Обход PASS 1 ПО СПИКЕРАМ (обязательное правило)
Движок cosyvoice3 в CrispASR получает клон-референс **только** из CLI-аргумента
`--voice <ref.wav>` при старте процесса. Смена голоса на конкретный запрос не
поддерживается: внутренний `cosyvoice3_tts::synth` умеет только 8 голосов из
`voices.gguf` и печатает `voice '%s' not found (have %zu)`, а WAV-клон идёт другим
путём — `synth_from_wav` от стартового `--voice`. Значит **один запуск движка = один голос**.

Поэтому PASS 1 обходит чанки **по спикерам** (`speaker_order` + `speaker_chunks`),
а не по хронологии. На тестовом ролике (4 спикера, 12 переключений): 4 запуска
движка вместо 13, TTS 119.9 с → 52.1 с. На 22-минутном ролике (8 спикеров,
46 переключений) было 47 запусков, станет 8.

**На результат не влияет:** чанк `i` синтезируется из тех же данных (текст +
голос), а PASS 2 (`plan_timeline`) и PASS 3 (`place_samples`) работают по исходному
`index` в хронологическом порядке. **Хронологический обход PASS 1 возвращать
нельзя** — это ровно то, из-за чего движок перезапускался на каждой смене голоса.

### Референс спикера: `MIN_CLONE_REF_SEC` и слияние коротких кластеров
Авто-кластеризация CrispASR (`--diarize-speakers auto`) недетерминирована: на
одном и том же аудио число спикеров гуляет 4 ↔ 5, и лишний кластер собирается из
шума, смеха и кашля. Такие сегменты проходят фильтр диаризации (≥ 0.15 с), но для
клона непригодны — CosyVoice3 строит референс из одного непрерывного отрезка, и
короче секунды эмбеддинг неустойчив.

`MIN_CLONE_REF_SEC = 1.0` живёт в `comm.rs` и это **единственный** источник для
обоих потребителей:
- `comm::merge_unclonable_speakers` (вызывается из `parse_engine_diarization`)
  присоединяет спикера без сегмента ≥ порога к ближайшему по времени валидному,
  сохраняя его речь в субтитрах, и пересобирает нумерацию без пропусков;
- `tts::pick_ref_segment` не выбирает для клона сегмент короче порога.

Инвариант: после разбора у каждого спикера есть референс. Проверяется
`build_test_diar_merge.bat` (отдельный бинарник — харнесс `cargo test` на этой
машине не стартует, 0xc0000139). Править это в TTS «мягкой деградацией» нельзя:
причиной был артефакт кластеризации, и лечить её надо до перевода.

Телеметрия в логе: `TTS: синтез N чанков, M спикер(ов), K переключений в хронологии
→ M запусков движка` и `TTS: движок запущен M раз, сумма генерации по движку Xs`.

### Подгонка и ресемплинг — один проход FFmpeg
`fit_and_resample_wav(input, output, tempo, out_sr, ffmpeg)` объединяет
`rubberband=tempo=…` (подгонка длительности) и `-ar … -ac 1` (24 кГц → 44.1 кГц)
в ОДИН запуск FFmpeg на чанк. Раньше это были два последовательных прохода плюс
промежуточный файл `deedub_tts_stretched_<N>.wav` (файла больше нет).
`tempo > 1.02` — единственное условие включения rubberband: планировщик отдаёт
ровно 1.0, когда речь влезает в окно натуральной скоростью, и >1 только когда
нужно ужимать.

## Build & Release (Tauri Build Toolkit)

Сборка/тесты выполняются ТОЛЬКО через `.bat`-обёртки (desktop §2 — прямой вызов `cargo`/`npx tauri` запрещён).
`build.bat`, `test.bat`, `generate_installer.bat`, `release.bat` — шаблоны общего тулкита
`..\my-tauri-plugins\tauri-build-toolkit` (единый источник правды для всех проектов). Настройки
проекта — `.build-config.json` в корне (repo, appExe, productName, signing, ...).

**ВАЖНО:** команды `build`/`prep`/`installer`/`release` автоматически бампят версию по схеме YY.M.P
в `src-tauri/tauri.conf.json` + `Cargo.toml` при каждом запуске. Dev-сборка (`build`) собирает без
бандла (dev-override), `installer`/`release` включают `bundle.active`.

Read-only диагностика: `node ..\my-tauri-plugins\tauri-build-toolkit\cli.cjs doctor --project "%CD%"`.

### Commands

```batch
REM Dev-сборка: prep (бамп версии + npm install + иконки) + tauri build без бандла + запуск deedub.exe
build.bat

REM Unit-тесты: cargo test [фильтр] (компиляция+прогон)
test.bat
REM То же в release-профиле. ВАЖНО: харнесс cargo test на этой машине не
REM стартует ни в debug, ни в release — 0xc0000139 STATUS_ENTRYPOINT_NOT_FOUND
REM (ошибка загрузчика ДО выполнения тестов). Поэтому рабочими точками
REM проверки TTS-логики остаются отдельные бинарники build_test_retry /
REM build_test_declick + полный run_tts_pipeline.
test_release.bat

REM Установщик NSIS: prep + tauri build --bundles nsis + верификация установщика и .sig
generate_installer.bat

REM Полный релиз на GitHub: build + sign + gh release + latest.json + commit/push
release.bat

REM Полный pipeline-тест (TTS-дубляж, сохраняет raw/final WAV в temp/)
run_tts_pipeline.bat

REM Полный pipeline-тест без дубляжа
run_pipeline.bat

REM Только компиляция тест-бинаря
build_test_pipeline.bat

REM Деклик red-тест (бинарь, работает): клики против порога <0.20 FS (костыль base, функция)
build_test_declick.bat

REM RetryTracker-тесты по всем сценариям обрезания (бинарь, работает)
build_test_retry.bat
```

### Batch files
| File | Purpose |
|------|---------|
| `build.bat` | Toolkit: dev build (`prep` + `npx tauri build` без бандла + запуск `deedub.exe`) |
| `test.bat` | Toolkit: `cargo test [фильтр]` (компиляция + прогон unit-тестов) |
| `test_release.bat` | Toolkit: `cargo test --release`. Обёртка над `test.bat --release` (флаг cargo, проброшен тулкитом). **На этой машине харнесс не стартует даже в release** — 0xc0000139; для проверки TTS-логики используй `build_test_retry.bat` / `build_test_declick.bat` / `run_tts_pipeline.bat` |
| `generate_installer.bat` | Toolkit: сборка NSIS-установщика + верификация `.sig` |
| `release.bat` | Toolkit: полный релиз (build + sign + `gh release` + `latest.json` + commit/push) |
| `.build-config.json` | Конфиг тулкита (repo, appExe, productName, signing, ...) |
| `run_pipeline.bat` | Full pipeline test (no dubbing) |
| `run_tts_pipeline.bat` | Full pipeline with TTS dubbing on `test/test_TTS_dubbing.mp4`, keeps raw/final WAVs (`DEEDUB_KEEP_TTS_WAV=1`) |
| `build_test_pipeline.bat` | Compile `test-pipeline` binary |
| `build_test_declick.bat` | Build+run `test-tts-declick`: red test for base-костыля click spikes (ch03/ch27 must drop below 0.20 FS, controls ch01/05/14 untouched). **Падает и это ожидаемо:** фикстуры сняты с base-модели до установки RL, костыль declick гейтом отключён для RL |
| `build_test_retry.bat` | Build+run `test-retry-tracker`: all truncation scenarios (ch7/ch27/ch34/ch35/ch38/ch36/monotonic) via real RetryTracker |
| `build_test_diar_merge.bat` | Build+run `test-diar-merge`: слияние спикеров без референса + сплошная нумерация. Зелёный |
| `build_test_censor_filter.bat` | Build+run `test-censor-filter`: замаскированные слова (`бл*ть`) не доходят до SRT/TTS, markdown и обычный мат не режутся |
| `build_test_srt_filter.bat` | Build+run `test-srt-filter`: `SKIP_MARKER` не попадает в SRT, нумерация cue сплошная |
| `run_mt_ab.bat` | A/B перевода на фиксированном входе `temp\mt_ab_input.json`: `run_mt_ab.bat <gguf> [temp] [chat\|insttrans]` |
| `run_pipeline_mt.bat` | Полный пайплайн с переопределением модели: `run_pipeline_mt.bat <gguf> [temp] [1=с дубляжом] [chat\|insttrans]` |

Пайплайн-батники (`run_pipeline`, `run_tts_pipeline`, `build_test_pipeline`, `build_test_declick`,
`build_test_retry`, `run_gemma4`) используют тот же MSVC-прелюд, что и шаблоны тулкита
(vswhere → vcvarsall x64 + сброс `RUSTC_WRAPPER`/`CC`/`CXX`/`CARGO_PROFILE_*`), но запускают
проектные `cargo run`/`cargo build` — у тулкита нет команды для произвольного бинаря.
`run_gemma4.bat` — тонкий алиас на `run_pipeline.bat`.

### Output files (проектные папки, core §1.2)
- Video with subtitles: `test/<name>_subbed.mp4` (рядом с исходником)
- Dubbed WAV: `temp/deedub_dubbed.wav`
- Per-chunk raw/final WAV: `temp/deedub_tts_{raw,final}_<N>.wav` (промежуточного `stretched` больше нет — rubberband и ресемплинг идут одним проходом FFmpeg)
- English SRT: `temp/deedub_subtitles_en.srt`
- Russian SRT: `temp/deedub_subtitles_ru.srt`
- Session log: `test/last_logs.txt` (зеркало от `tauri-plugin-logs`), рядом с exe — `deedub.log`

## Config
File: `~/.deedub/config.toml`
Fields: `ffmpeg_path`, `output_format`, `enable_dubbing`, `mix_volume`, `prompt_style`
Модель перевода здесь НЕ хранится — она в конфиге плагина движка
(`%APPDATA%\com.deedub.desktop\app_config.json`, ключ `last_model`), туда же
команда `set_translation_model` пишет выбор из выпадающего списка. Формат
промпта — app-specific, поэтому живёт здесь (`auto` | `insttrans` | `chat`).

⚠️ `save_config` на фронте шлёт **весь** структур целиком. Любое новое поле
`AppConfig` обязано появиться и в литерале `save_config`, и в типе ответа
`get_config` в `src/App.tsx`, иначе оно пропадёт из `config.toml`. Парсинг
`config.toml` при ошибке логируется в `config::load` (раньше молча отдавался
пустой дефолт, и «сброс настроек» выглядел как забывчивость приложения).

**Известное ограничение плагина:** `add_model` безусловно выставляет
`last_model = Some(path)`, даже если модель уже в реестре. Поэтому добавление
модели в панели сбрасывает выбор из селекта — выбирать активную модель стоит
после добавления. Сам плагин не меняем (общий репозиторий).

## Known Issues
- Occasional STT errors not corrected (e.g., "sea" → "море" instead of "sight" → "зрелище") — 12B model limitation
- "спотыкалась" instead of "соскальзывал" for "slipped" context — model doesn't infer the tube top slip meaning
- CoT occasionally produces "thought: ..." prefix instead of proper `<|channel>thought` format — handled by `clean_output`
- `9.5` (рейтинг) Index-Translate пишет цифрами даже после ретрая с прямым
  требованием — ретрай срабатывает в ~50% случаев. Chunk 36 на тестовом ролике
  поэтому иногда даёт `FAIL`. Конвертер цифр в слова на стороне Rust
  сознательно не делаем: русские числительные согласуются с существительным
  («два»/«две»), а модель знает контекст, регулярка — нет.

## Relevant Source Files
| File | Purpose |
|------|---------|
| `src-tauri/src/translation/mod.rs` | Translation pipeline |
| `src-tauri/src/llm.rs` | Фасад LLM: `LlmSession` поверх `tauri-plugin-llama-engine` (движок, модель, параметры, бюджет) |
| `src-tauri/src/lib.rs` | App init, плагины (logs/downloader/llama-engine), PIPELINE_CANCEL |
| `src-tauri/src/pipeline.rs` | Pipeline orchestrator |
| `src-tauri/src/comm.rs` | PipelineContext, SubtitleChunk, `MIN_CLONE_REF_SEC`, `merge_unclonable_speakers` |
| `src-tauri/src/config/mod.rs` | Config load/save |
| `src-tauri/src/audio_extractor/mod.rs` | WAV extraction |
| `src-tauri/src/diarization/mod.rs` | CrispASR ASR + speaker diarization (single source) |
| `src-tauri/src/stt/mod.rs` | Consumes diarization transcript |
| `src-tauri/src/output/mod.rs` | Subtitle muxing, `build_srt_content` — чистая сборка SRT с фильтром `SKIP_MARKER` |
| `src-tauri/src/tts/mod.rs` | TTS дubляж: синтез + retry и declick только для base (`RetryTracker`, `TTS_FULL_FLOOR`, `declick_spikes`) |
| `src-tauri/src/bin/test_retry_tracker.rs` | Автономный тест сценариев обрезания (работает на этой машине) |
| `src-tauri/src/bin/test_diar_merge.rs` | Слияние спикеров без референса (работает на этой машине) |
| `src-tauri/src/bin/test_censor_filter.rs` | Фильтр цензуры слов (работает на этой машине) |
| `src-tauri/src/bin/test_srt_filter.rs` | `SKIP_MARKER` не попадает в SRT (работает на этой машине) |
| `src-tauri/src/bin/test_mt_ab.rs` | Фиксированный вход для A/B перевода (без диаризации и TTS) |
| `src-tauri/src/paths.rs` | Единый источник путей (core §1.2/§2.5.1): `temp/`, `test/` |
| `src/App.tsx` | Главная вкладка: выбор файла, селекты модели и формата промпта, прогресс |
| `src-tauri/Cargo.toml` | Rust dependencies |

## Research
- `research_tower_plus_9b/` — full research findings on Tower-Plus-9B prompt format, translation prompting, and alternative approaches
- Key finding: source-side context (previous EN text) is more important than target-side context for translation quality
- Key finding: temperature 0.1-0.3 is optimal for factual translation tasks
- Key finding: GBNF grammar not needed for standard instruction-tuned models like Tower-Plus-9B