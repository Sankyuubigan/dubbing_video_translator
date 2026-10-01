use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let video_path = parse_arg(&args, "--video")
        .or_else(|| parse_arg(&args, "-v"))
        .or_else(|| {
            // Default: project_root/test/for_test.mp4
            let root = project_root();
            let path = root.join("test").join("for_test.mp4");
            if path.exists() { Some(path.to_string_lossy().to_string()) } else { None }
        });

    let video = match video_path {
        Some(v) => {
            let p = PathBuf::from(&v);
            if p.is_relative() {
                let abs = project_root().join(&p);
                if abs.exists() {
                    abs.to_string_lossy().to_string()
                } else {
                    eprintln!("ERROR: video not found: {} (tried: {})", v, abs.display());
                    std::process::exit(1);
                }
            } else if p.exists() {
                v
            } else {
                eprintln!("ERROR: video not found: {}", v);
                std::process::exit(1);
            }
        }
        None => {
            eprintln!("Usage: test-pipeline --video <path> [--dub]");
            eprintln!("       (defaults to test/for_test.mp4 in project root)");
            std::process::exit(1);
        }
    };

    let enable_dubbing = args.iter().any(|a| a == "--dub");
    log::info!("test-pipeline: video = {}, dub = {}", video, enable_dubbing);

    eprintln!("test-pipeline: headless (dub={})", enable_dubbing);
    let code = deedub_lib::run_headless(video, enable_dubbing);

    eprintln!("\n=== EXIT CODE {} ===", code);
    std::process::exit(code);
}

fn parse_arg(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            if let Some(val) = iter.next() {
                return Some(val.clone());
            }
        }
    }
    None
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}