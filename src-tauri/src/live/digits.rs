//! Цифры → русские слова перед TTS (PRP шаг 4: конвертер ловит хвост, который
//! промпт-правило «пиши числа словами» пропустил; регулярка не знает контекста,
//! поэтому именительный падеж и поцифровое чтение дробной части).
//!
//! Применяется ТОЛЬКО к тексту перед озвучкой: субтитры остаются как перевела
//! модель («Сезон 10» глазами читается, а CosyVoice3 цифры не произносит).
//!
//! Границы v1 (сознательные): дробная часть по цифрам («2.25» → «два и два
//! пять»); числа до 999 999 999; группировка «1 500 000» с пробелами — три
//! отдельных числа; падежи — именительный (PRP: «им. падеж»).

/// Таблицы слов (индексный доступ — компактнее матчей и не ломается).
const U: &str = "ноль один два три четыре пять шесть семь восемь девять";
/// Женские единицы для тысяч: «одна/две тысячи». Индекс 0 — заглушка
/// (таблица читается с нуля, а ноль в this позиции не встречается: ветки
/// `n > 0` и `n % 10 > 0` отсекают его).
const UF: &str = "- одна две три четыре пять шесть семь восемь девять";
const TEEN: &str = "десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать";
const T: &str = "двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто";
const H: &str = "сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот";

fn w(table: &str, i: usize) -> String {
    table.split_whitespace().nth(i).unwrap_or("").to_string()
}

/// Все числа `\d+([.,]\d+)?` в тексте → русские слова (именительный падеж).
pub fn digits_to_words(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 32);
    let mut chars = text.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            // Число: цифры + опционально «. | ,» + цифры. Разделитель, за
            // которым не цифра («1.» в конце, «3-4») — не часть числа.
            let mut int_part = String::new();
            let mut frac: Option<String> = None;
            while let Some(&c2) = chars.peek() {
                if c2.is_ascii_digit() {
                    // После разделителя цифры идут в дробную часть, до него —
                    // в целую (раньше они всегда уезжали в int_part: «9.5»
                    // превращалось в «девяносто пять и»).
                    match &mut frac {
                        Some(f) => f.push(c2),
                        None => int_part.push(c2),
                    }
                    chars.next();
                } else if frac.is_none() && (c2 == '.' || c2 == ',') {
                    let mut probe = chars.clone();
                    probe.next(); // съели разделитель
                    if probe.peek().is_some_and(|d| d.is_ascii_digit()) {
                        frac = Some(String::new());
                        chars.next();
                        continue;
                    }
                    break;
                } else {
                    break;
                }
            }
            out.push_str(&int_to_words(&int_part));
            if let Some(f) = frac {
                out.push_str(" и ");
                for (i, d) in f.chars().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(&w(U, d.to_digit(10).unwrap_or(0) as usize));
                }
            }
        } else {
            out.push(c);
            chars.next();
        }
    }
    out
}

/// Целое (≤9 цифр) → слова; длиннее — возвращается как было (не ломать текст).
fn int_to_words(token: &str) -> String {
    let n: u64 = match token.parse() {
        Ok(v) if token.len() <= 9 => v,
        _ => return token.to_string(),
    };
    if n == 0 {
        return "ноль".into();
    }
    let m = n / 1_000_000;
    let th = (n % 1_000_000) / 1_000;
    let rest = n % 1_000;
    let mut parts: Vec<String> = Vec::new();
    if m > 0 {
        parts.push(format!("{} {}", below_1000(m, false), count_word(m, "миллион", "миллиона", "миллионов")));
    }
    if th > 0 {
        parts.push(format!("{} {}", below_1000(th, true), count_word(th, "тысяча", "тысячи", "тысяч")));
    }
    if rest > 0 {
        parts.push(below_1000(rest, false));
    }
    parts.join(" ")
}

/// 1..999 → слова; `fem` — женские единицы (для «одна/две тысячи»).
fn below_1000(n: u64, fem: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut n = n;
    if n >= 100 {
        parts.push(w(H, (n / 100 - 1) as usize));
        n %= 100;
    }
    if (10..20).contains(&n) {
        parts.push(w(TEEN, (n - 10) as usize));
    } else if n >= 20 {
        parts.push(w(T, (n / 10 - 2) as usize));
        if n % 10 > 0 {
            parts.push(w(if fem { UF } else { U }, (n % 10) as usize));
        }
    } else if n > 0 {
        parts.push(w(if fem { UF } else { U }, n as usize));
    }
    parts.join(" ")
}

/// Форма счётного слова: 1 → «миллион», 2-4 → «миллиона», прочее → «миллионов».
fn count_word(n: u64, one: &'static str, few: &'static str, many: &'static str) -> &'static str {
    let d = n % 10;
    let dd = n % 100;
    if (11..=14).contains(&dd) {
        return many;
    }
    match d {
        1 => one,
        2..=4 => few,
        _ => many,
    }
}
