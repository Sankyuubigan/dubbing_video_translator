//! Единый источник правды для путей проекта (core §1.2, §2.5.1).
//!
//! Все временные файлы и логи обязаны жить внутри проекта (`temp/`, `test/`),
//! а не в системной temp-папке. Путь вычисляется от `current_exe()` без хардкода
//! абсолютных путей (core §1.4).

use std::path::{Path, PathBuf};

/// Корень проекта. Поднимаемся от `current_exe()` вверх:
/// 1) первый предок с существующей папкой `test/`;
/// 2) иначе первый предок — корень Tauri-проекта (маркер: рядом есть `src-tauri/`);
/// 3) иначе — папка рядом с exe (упакованный бинарь), либо cwd.
pub fn project_root() -> PathBuf {
    let exe = std::env::current_exe().ok();
    if let Some(exe) = exe.as_ref() {
        let mut tauri_root: Option<PathBuf> = None;
        let mut cur = exe.parent();
        while let Some(dir) = cur {
            if dir.join("test").is_dir() {
                return dir.to_path_buf();
            }
            if tauri_root.is_none() && dir.join("src-tauri").is_dir() {
                tauri_root = Some(dir.to_path_buf());
            }
            cur = dir.parent();
        }
        if let Some(root) = tauri_root {
            return root;
        }
        if let Some(dir) = exe.parent() {
            return dir.to_path_buf();
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Папка `test/` в корне проекта (авто-создаётся). Здесь лежат логи последней сессии.
pub fn test_dir() -> PathBuf {
    ensure(project_root().join("test"))
}

/// Папка `temp/` в корне проекта (авто-создаётся). Здесь лежат временные WAV/файлы.
pub fn temp_dir() -> PathBuf {
    ensure(project_root().join("temp"))
}

/// Путь к временному файлу внутри проектной `temp/`.
pub fn temp_file(name: &str) -> PathBuf {
    temp_dir().join(name)
}

/// Файл лога последней сессии: `<root>/test/last_logs` (core §2.5.1).
pub fn last_logs_file() -> PathBuf {
    test_dir().join("last_logs")
}

fn ensure(dir: PathBuf) -> PathBuf {
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("paths: не удалось создать {}: {}", dir.display(), e);
    }
    dir
}

#[allow(dead_code)]
pub fn is_under_project(p: &Path) -> bool {
    p.starts_with(project_root())
}
