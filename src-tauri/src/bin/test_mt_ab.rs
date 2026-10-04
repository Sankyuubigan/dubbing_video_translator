use std::path::PathBuf;

/// A/B-прогон перевода на фиксированном входе (см. `deedub_lib::run_mt_ab`).
///
///   test-mt-ab --video test/test_TTS_dubbing.mp4 --chunks temp/mt_ab_input.json
///
/// Модель берётся из `DEEDUB_LLM_MODEL`, результат пишется в
/// `temp/mt_ab_out.json`. Диаризация и TTS не запускаются.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();

    let video = pick(&args, &["--video", "-v"], &root.join("test/test_TTS_dubbing.mp4"));
    let chunks = pick(&args, &["--chunks", "-c"], &root.join("temp/mt_ab_input.json"));

    let video = video.to_string_lossy().to_string();
    let chunks = chunks.to_string_lossy().to_string();

    eprintln!("test-mt-ab: video  = {video}");
    eprintln!("test-mt-ab: chunks = {chunks}");

    let code = deedub_lib::run_mt_ab(video, chunks);
    eprintln!("\n=== EXIT CODE {code} ===");
    std::process::exit(code);
}

fn pick(args: &[String], names: &[&str], default: &PathBuf) -> PathBuf {
    for name in names {
        if let Some(i) = args.iter().position(|a| a == name) {
            if let Some(v) = args.get(i + 1) {
                let p = PathBuf::from(v);
                return if p.is_absolute() {
                    p
                } else {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .unwrap()
                        .join(p)
                };
            }
        }
    }
    default.to_path_buf()
}