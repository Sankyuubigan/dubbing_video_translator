use std::os::windows::process::CommandExt;
use std::path::Path;
use std::process::Command;

const CREATE_NO_WINDOW: u32 = 0x08000000;

pub fn resolve(cfg_path: &Option<String>) -> String {
    if let Some(path) = cfg_path {
        if !path.is_empty() && Path::new(path).exists() {
            return path.clone();
        }
    }

    let candidates = vec![
        std::env::current_exe().ok().and_then(|p| {
            p.parent().map(|d| d.join("ffmpeg.exe"))
        }),
        Some(Path::new("ffmpeg.exe").to_path_buf()),
        Some(Path::new("ffmpeg").to_path_buf()),
    ];

    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            return candidate.to_string_lossy().to_string();
        }
    }

    "ffmpeg".to_string()
}

#[allow(dead_code)]
pub fn check_available(cfg_path: &Option<String>) -> bool {
    let exe = resolve(cfg_path);
    Command::new(&exe)
        .creation_flags(CREATE_NO_WINDOW)
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
