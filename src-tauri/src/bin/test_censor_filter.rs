//! Тест фильтра цензуры: замаскированные слова (`бл*ть`) не должны доходить
//! ни до субтитров, ни до CosyVoice3.
//!
//! Отдельный бинарник, а не `#[test]`, потому что харнесс `cargo test` на этой
//! машине не стартует ни в debug, ни в release (0xc0000139
//! STATUS_ENTRYPOINT_NOT_FOUND — ошибка загрузчика ДО выполнения тестов).
//! Рабочие точки проверки в проекте устроены так же: `test_retry_tracker`,
//! `test_tts_declick`, `test_diar_merge`.
//!
//! Санитар зовётся из `comm`, то есть это production-функция, а не её копия:
//! копия в тесте проверяла бы саму себя и разошлась бы с кодом при первой же
//! правке. Юнит-тесты с теми же сценариями лежат в `translation::tests` и
//! `verification::tests` — они для сред, где харнесс работает.

use deedub_lib::comm::sanitize_for_output as clean_output;
use std::process::exit;

/// Реальный вывод Index-Translate-9B из A/B-прогона, чанк 5.
/// Источник: `temp/mt_ab_out_Index-Translate-9B.IQ4_XS_insttrans.json`.
const REAL_MASKED: &str = "Не могут позволить себе ПК. Бл*ть, обманщики всё равно остаются обманщиками. Играть против снайперских персонажей — это не фига не весело.";

static mut FAILED: usize = 0;

fn check(name: &str, cond: bool) {
    if cond {
        println!("  OK   {name}");
    } else {
        println!("  FAIL {name}");
        unsafe { FAILED += 1 };
    }
}

fn main() {
    println!("Фильтр цензуры (Index-Translate-9B)\n");

    // 1. Реальный случай из прогона.
    println!("real_masked_word_removed");
    let out = clean_output(REAL_MASKED);
    check("звёздочек в выходе нет", !out.contains('*'));
    check("замаскированного слова нет", !out.contains("Бл*ть"));
    check("остальной текст сохранён", out.contains("обманщики"));
    check("хвост предложения на месте", out.contains("не фига не весело"));
    println!();

    // 2. Мат без цензуры резать нельзя — это валидный перевод.
    println!("plain_profanity_kept");
    let plain = "Чёрт, это не весело";
    check("текст не изменился", clean_output(plain) == plain);
    println!();

    // 3. Markdown — не цензура. Production-код снимает `**` только с краёв
    //    строки, поэтому `**Привет**` в середине фразы остаётся как есть, и
    //    санитар не должен выбрасывать содержимое.
    println!("markdown_not_confused_with_censorship");
    check("краевой **Привет** снят", clean_output("**Привет**") == "Привет");
    check(
        "слово с ** внутри не выброшено",
        clean_output("**Привет** мир").contains("Привет"),
    );
    println!();

    // 4. Несколько замаскированных слов подряд.
    println!("all_masked_words_removed");
    let many = clean_output("Бл*ть, п*здец и ещё");
    check("звёздочек не осталось", !many.contains('*'));
    check("осталось только «и ещё»", many == "и ещё");
    println!();

    // 5. Одиночная звёздочка как отдельный токен — не замаскированное слово.
    println!("standalone_star_kept");
    let star = "звёздочка * отдельно";
    check("текст не изменился", clean_output(star) == star);
    println!();

    // 6. Чистый текст не трогаем вообще.
    println!("clean_text_untouched");
    let clean = "У вас два страйка, и это уже не смешно";
    check("текст не изменился", clean_output(clean) == clean);
    println!();

    // 7. Двойных пробелов не остаётся после выбрасывания слова.
    println!("no_double_spaces_after_drop");
    let mid = clean_output("ПК. Бл*ть, обманщики");
    check("нет двойного пробела", !mid.contains("  "));
    check("стало «ПК. обманщики»", mid == "ПК. обманщики");
    println!();

    // 8. Известная граница фильтра: маска из ДВУХ звёздочек (`f**k`) под `**`
    //    не попадает — в этом проекте `**` считается markdown. Наблюдалась
    //    только одиночная маска, поэтому ловим её; двойная остаётся зоной
    //    проверки `markdown_bold`. Тест фиксирует границу, чтобы её не потерять.
    println!("double_star_is_markdown_not_censorship");
    check(
        "f**k не выбрасывается санитаром",
        clean_output("ты f**k").contains("f**k"),
    );
    println!();

    let failed = unsafe { FAILED };
    if failed == 0 {
        println!("ALL PASS");
    } else {
        println!("FAILED: {failed}");
        exit(1);
    }
}