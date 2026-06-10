use std::path::PathBuf;

fn main() {
    app_lib::truncate_logs();
    app_lib::setup_logger();

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
            eprintln!("Usage: test-pipeline --video <path>");
            eprintln!("       (defaults to test/for_test.mp4 in project root)");
            std::process::exit(1);
        }
    };

    log::info!("test-pipeline: video = {}", video);

    let cfg = app_lib::config::load();
    let sherpa_dir = cfg.sherpa_onnx_dir.clone().unwrap_or_default();
    let translate = cfg.gguf_model_path.clone().unwrap_or_default();

    if sherpa_dir.is_empty() && !cfg.stt_model.as_deref().unwrap_or("").contains("qwen") {
        log::error!("Missing sherpa_onnx_dir in config");
        std::process::exit(1);
    }
    if translate.is_empty() {
        log::error!("Missing gguf_model_path in config");
        std::process::exit(1);
    }

    let pcfg = app_lib::comm::PipelineConfig {
        input_path: video,
        output_format: "mp4".to_string(),
        gguf_model_path: Some(translate),
        ffmpeg_path: cfg.ffmpeg_path.clone(),
        vad_threshold_db: cfg.vad_threshold_db.clone(),
        sherpa_onnx_dir: cfg.sherpa_onnx_dir.clone(),
        stt_model: cfg.stt_model.clone(),
        diarization_threshold: cfg.diarization_threshold,
        diarization_num_speakers: cfg.diarization_num_speakers,
    };
    let ctx = app_lib::comm::PipelineContext::new(pcfg);

    log::info!("test-pipeline: running...");
    let result = app_lib::pipeline::run(ctx);

    match result {
        Ok(res) => {
            let out = res.output_path.unwrap_or_default();
            log::info!("test-pipeline: SUCCESS output={}", out);
            eprintln!("\n=== SUCCESS ===");
            eprintln!("Output: {}", out);
            std::process::exit(0);
        }
        Err(e) => {
            let err_msg = format!("{:#}", e);
            log::error!("test-pipeline: {}", err_msg);
            eprintln!("\n=== ERROR ===");
            eprintln!("{}", err_msg);
            std::process::exit(1);
        }
    }
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
