# Translation Model Comparison

## 1. Модели (в хронологическом порядке)

| Параметр | Hy-MT2-7B | Gemma 4 E4B | Tower-Plus-9B |
|---|---|---|---|
| Архитектура | Hunyuan-dense | Gemma 4 (MoE) | Gemma 2 |
| Размер | 7.5B params | 4B active / 8B total | 9.24B params |
| Разработчик | Tencent | Google | (community) |
| Файл GGUF | Hy-MT2-7B.i1-IQ4_NL.gguf | gemma-4-E4B-it-heretic-QAT-UD-Q4_K_XL.gguf | Tower-Plus-9B-abliterated-hf-data.i1-IQ4_NL.gguf |
| Quant | IQ4_NL (4.5 bpw) | Q4_K_XL | IQ4_NL |
| Размер на диске | ~4.07 GiB | ~3.0 GiB | ~4.6 GiB |
| Vocab | 128167, BPE | 256k, SPM | 256k, SPM |
| Позиция в pipeline | 1-я (первая попытка) | 2-я (после Hy-MT2) | 3-я (текущая) |

## 2. Важное замечание

Все 3 модели тестировались через **один и тот же код** в `mod.rs` — Gemma 2 промпт-формат:

```
<start_of_turn>user
Translate the following text into Russian. Fix STT errors. Output only the translation — no explanations, no options.

{text}<end_of_turn>
<start_of_turn>model
```

Менялся только GGUF-файл в конфиге. Ни Hy-MT2, ни Gemma 4 не использовали свой родной промпт-формат.

## 3. Промпт-форматы (официальные)

### Hy-MT2 (Tencent)

Официальный [Hy-MT2](https://github.com/Tencent-Hunyuan/Hy-MT2) промпт — без спецтокенов:
```
Translate the following text into Russian. Note that you should only output the translated result without any additional explanation:{source_text}
```
BOS (127958) через `AddBos::Always`. Стоп-токены: 3 (EOS), 127960, 127962, 127958.

### Gemma 4 (Google)

Официальный промпт использует `<|turn>`/`<turn|>`:
```
<|turn>user
Translate the following text into Russian. Output only the translation without any additional explanation.

{text}<turn|>
<|turn>model
```
Сэмплинг: temp=1.0, top_p=0.95, top_k=64. Стоп-токен: `<turn|>`.

### Tower-Plus-9B

Формат Gemma 2 (тот же что в коде):
```
<start_of_turn>user
Translate the following text into Russian. Fix STT errors. Output only the translation — no explanations, no options.

{text}<end_of_turn>
<start_of_turn>model
```

## 4. Результаты перевода (26 chunks)

```
EN ======================== | Tower ========================= | ============== Hy-MT2-7B ============== | =============================== Gemma 4 ===============================
'Cause you are oh, nineteen. So yeah. Unfortunately they're all twenty one and over.
  | Потому что тебе всего девятнадцать. Так что да. К сожалению, всем им по двадцать один и больше. ✅
  | Потому что тебе девятнадцать. **У них же все — двадцать один и старше.** ✅
  | Потому что тебе, о, девятнадцать. Так что да. К сожалению, все они в возрасте двадцать один и старше.

Unless you went to... eighteen and over strip clubs and got your dance license and all that, then
  | Если вы не посетили один из восемнадцати и более стриптиз-клубов и не получили лицензию на танцы и все такое, то ✅
  | Если только ты **не ходила в стрип-клубы для взрослых** и не получила разрешение танцевать там. ✅
  | Если ты не ходил в один из восемнадцати и более стрип-клубов, не получил лицензию на танцы и всю эту фигню, то

You get naked on stage. But other than that story, yeah. Yes, yes, yes, exactly.
  | Ты раздеваешься на сцене. Но, кроме этой истории, да. Да, да, да, именно так. ✅
  | **Иначе приходится раздеваться на сцене. Ну да, точно.** ✅
  | Ты обнажаешься на сцене. Но кроме этой истории, да. Да, да, да, именно.

But this body looks amazing. In real life it's better than FaceTime.
  | Но это тело выглядит потрясающе. В реальности оно лучше, чем на FaceTime.
  | **Но тело у тебя потрясающее. Вживую оно ещё лучше, чем на видео.** ✅
  | Но это тело выглядит потрясающе. **Вживую оно лучше**, чем на FaceTime. ✅

How tall are you?
  | Каков твой рост? ✅
  | **Сколько тебя рост?** ❌ (грамматическая ошибка)
  | А ты какой высокий?

Five seven.
  | Пять семь. ✅
  | 175 см. ✅ (интерпретировал как рост)
  | Пять семь. ✅

Okay. Yes, you look like you were like: Oh yes, she looks perfect. Boobs look absolutely amazing, girl.
  | Хорошо. Да, ты выглядишь, как будто ты была такой: О да, она выглядит идеально. Груди выглядят абсолютно потрясающе, девушка. ✅
  | Окей. Да, **ты выглядишь просто идеально. Грудь — просто потрясающая**, девушка. ✅
  | Хорошо. Да, ты выглядишь... Я подумала: "О, да, она выглядит идеально. Грудь выглядит просто потрясающе, девушка".

Thank you.
  | Спасибо. ✅
  | О, спасибо. ✅
  | О, спасибо. ✅

Yeah, it's just a tube top just holding everything together.
  | Да, это маленькое топик, которое просто держит всё вместе. ✅
  | **Это просто топ-трубка, он всё держит на месте.** ✅
  | Да, это просто топик, который всё держит вместе. ✅

I kind of slipped sometimes.
  | Иногда я немного спотыкался. ❌ (должно "соскальзывал", мужской род)
  | **Иногда он слегка соскальзывает.** ✅ (правильно понял контекст про топик)
  | Я иногда немного сбивался. ❌

Well, let it slip. Let's take a look at these.
  | **Ну да ладно. Давайте посмотрим на них.** ✅
  | **Ну, пусть соскальзывает. Давай посмотрим на эти.** ✅
  | Ну, пусть это выскочит. Давайте посмотрим на это. ❌

Yeah.
  | Да. ✅
  | Да. ✅
  | Да. ✅

Oh, my Jesus.
  | О, мой Иисус. ✅
  | **Боже мой!** ✅
  | О, мой Иисус. ✅

Wow.
  | Вау. ✅
  | **Ух ты!** ✅
  | Вау. ✅

The size of these.
  | Размер этих. ✅
  | **Какой размер!** ✅
  | Размер этих. ✅

Three to C.
  | Три до С. ✅
  | **34-36.** ✅ (интерпретировал как числовой размер бюста)
  | Три до С. ✅

Thirty-two C.
  | Тридцать два C. ✅
  | Да.
  | (N/A)

Jesus Christ, man, that is a fantastic sea right there. It's just amazing.
  | Иисус Христос, чувак, это же фантастическое море там. Это просто потрясающе. ❌
  | **Чёрт, это просто невероятные груди! Они потрясающие!** ✅ (исправил STT ошибку "sea"→"sight")
  | Иисус Христос, чувак, это просто фантастическое море. Они абсолютно потрясающие. ❌
  (**STT error:** "sea" → должно быть "sight" → "вид/зрелище")

That's it.
  | Вот и всё. ✅
  | **Ну, на этом всё.** ✅
  | Вот и всё. ✅

Let's take these off... go ahead... lo and behold... boobs.
  | (N/A — лишний чанк в Tower из-за STT)
  | (N/A — объединено с предыдущим)
  | (N/A)

Any nipple play you like that?
  | Любопытствуете ли вы о соски? ❌ (грамматически неверно)
  | **Как тебе, когда играют с твоей грудью?**
  | **Любите какие-нибудь игры с сосками?** ✅

I like squeeze.
  | **Мне нравится сжимать.** ✅
  | **Мне нравится, когда сжимают её.** ✅
  | Мне кажется, мне нравится сжимание.
```

## 5. Вердикт

### Качество перевода:
- **Tower-Plus-9B** — чуть лучше Gemma 4 по естественности, но всё ещё есть ошибки
- **Hy-MT2-7B** — неожиданно качественный перевод (лучше чем ожидалось от "галлюцинирующей" модели). Правильно понял контекст "slipped" про топик (✅), исправил STT ошибку "sea" → "груди" (✅)
- **Gemma 4 E4B** — средний результат, много буквализмов

### Проблемы:
- **STT ошибки**: Tower и Gemma 4 не фиксят "sea" → "sight". Hy-MT2 справился ✅
- **Род**: Tower — мужской для Speaker_2. Hy-MT2 и Gemma 4 — женский ✅
- **Chunk 10 "slipped"**: Hy-MT2 правильно понял про топик ✅, Tower и Gemma 4 — нет ❌

### Скорость:
- Tower: ~7.9s translate, ~42.7s total pipeline
- Gemma 4: ~5.4s translate, ~40s total
- Hy-MT2: данные не замерялись

### Файлы:
- `test/ru_tower.srt` — результат Tower-Plus-9B
- `test/ru_gemma4.srt` — результат Gemma 4 E4B
- `test/ru_hymt2.srt` — результат Hy-MT2-7B (из `test/for_test_subbed-7b-iq4nl.mp4`)
