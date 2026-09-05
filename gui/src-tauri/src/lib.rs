use std::collections::{HashMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use tauri::State;

/// Maximum size we will read for a single censor-plan JSON blob.
/// Plans are small JSON documents - anything larger almost certainly
/// indicates the wrong file or an attempt to exhaust memory.
const MAX_PLAN_BYTES: u64 = 16 * 1024 * 1024; // 16 MiB

/// Video file extensions the desktop app is willing to thumbnail.
const ALLOWED_VIDEO_EXTS: &[&str] = &[
    "mkv", "mp4", "mov", "avi", "webm", "m4v", "ts", "wmv",
];

/// Log lines kept per job. Progress bars emit a redraw per frame tick, so
/// the buffer must hold the last full redraws to make the tail readable.
const LOG_TAIL_LINES: usize = 400;

/// Log lines returned by `job_status` - enough for the UI to show the
/// active progress line plus recent context without shipping the world.
const LOG_TAIL_REPORT: usize = 24;

const VALID_PROFILES: &[&str] = &["AUTO", "HIGH", "MEDIUM", "LOW", "CPU"];
const VALID_CONTENT_TYPES: &[&str] = &["live-action", "animation", "anime", "low-light"];

#[derive(Serialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
enum JobState {
    Running,
    Done,
    Failed,
}

#[derive(Serialize)]
struct JobStatusResponse {
    state: JobState,
    exit_code: Option<i32>,
    mode: String,
    output: Option<String>,
    log_tail: Vec<String>,
}

struct Job {
    child: Child,
    state: JobState,
    exit_code: Option<i32>,
    mode: String,
    output: Option<String>,
    log: Arc<Mutex<VecDeque<String>>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StartJobOptions {
    output: Option<String>,
    no_audio: bool,
    no_clip: bool,
    /// One of: "plan", "apply", "process". Validated server-side.
    mode: String,
    /// Optional hardware profile override; AUTO (or absent) lets the CLI decide.
    profile: Option<String>,
    /// Optional content-type preset for detection thresholds.
    content_type: Option<String>,
    /// Optional nudity detection threshold in [0, 1].
    threshold: Option<f64>,
}

/// Normalize and validate a user-supplied filesystem path.
///
/// The frontend can call into Tauri commands with arbitrary strings, so each
/// path is canonicalized and then checked against an extension allow-list.
/// This blocks the trivial path-traversal/wildcard-read failure modes that
/// the previous implementation had by using `std::fs::*` on unsanitized
/// arguments.
fn validated_path(raw: &str, allowed_exts: &[&str], must_exist: bool) -> Result<PathBuf, String> {
    if raw.is_empty() {
        return Err("path is empty".into());
    }
    let path = PathBuf::from(raw);

    let resolved = if must_exist {
        path.canonicalize()
            .map_err(|e| format!("invalid path '{}': {}", raw, e))?
    } else {
        // Canonicalize the parent so we still reject ".." sneakiness even
        // when creating a new file.
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let canon_parent = parent
            .canonicalize()
            .map_err(|e| format!("invalid parent directory: {}", e))?;
        let fname = path
            .file_name()
            .ok_or_else(|| "missing file name".to_string())?;
        canon_parent.join(fname)
    };

    let ext_ok = resolved
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let lower = e.to_ascii_lowercase();
            allowed_exts.iter().any(|a| a.eq_ignore_ascii_case(&lower))
        })
        .unwrap_or(false);

    if !ext_ok {
        return Err(format!(
            "path '{}' has disallowed or missing extension",
            resolved.display()
        ));
    }

    Ok(resolved)
}

/// Consume one output pipe of a job and append complete lines to the shared
/// log ring. Progress bars redraw with `\r` (and rich also uses ANSI codes),
/// so lines are split on both `\r` and `\n`; the UI reads the tail to show
/// live progress. Without a reader here, the CLI fills the OS pipe buffer
/// (~64 KiB) and blocks mid-render.
fn drain_pipe(mut pipe: impl Read + Send + 'static, log: Arc<Mutex<VecDeque<String>>>) {
    let mut buf = [0u8; 4096];
    let mut carry = String::new();
    loop {
        match pipe.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                carry.push_str(&String::from_utf8_lossy(&buf[..n]));
                while let Some(pos) = carry.find(['\r', '\n']) {
                    let line: String = carry.drain(..=pos).collect();
                    let line = line.trim_end_matches(['\r', '\n']).trim().to_string();
                    if line.is_empty() {
                        continue;
                    }
                    if let Ok(mut g) = log.lock() {
                        if g.len() >= LOG_TAIL_LINES {
                            g.pop_front();
                        }
                        g.push_back(line);
                    }
                }
                // A pathological no-newline stream must not grow forever.
                if carry.len() > 16 * 1024 {
                    carry.clear();
                }
            }
            Err(_) => break,
        }
    }
}

#[tauri::command]
fn start_job(
    input_path: String,
    options: StartJobOptions,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let mode = match options.mode.as_str() {
        "plan" | "apply" | "process" => options.mode.clone(),
        other => return Err(format!("invalid mode '{}'", other)),
    };

    if let Some(profile) = &options.profile {
        if !VALID_PROFILES
            .iter()
            .any(|p| p.eq_ignore_ascii_case(profile))
        {
            return Err(format!("invalid profile '{}'", profile));
        }
    }
    if let Some(ct) = &options.content_type {
        if !VALID_CONTENT_TYPES
            .iter()
            .any(|c| c.eq_ignore_ascii_case(ct))
        {
            return Err(format!("invalid content type '{}'", ct));
        }
    }
    if let Some(t) = options.threshold {
        if !(0.0..=1.0).contains(&t) {
            return Err(format!("threshold {} outside [0, 1]", t));
        }
    }

    let input = validated_path(&input_path, ALLOWED_VIDEO_EXTS, true)?;

    let mut args = vec![mode.clone(), input.to_string_lossy().into_owned()];

    // The CLI derives the plan path as "<input>.censorplan.json" when no
    // --output is given. Record the same derivation here so the UI can open
    // the plan the moment the job finishes.
    let effective_output: Option<PathBuf> = match &options.output {
        Some(out) => {
            let exts: &[&str] = if mode == "plan" {
                &["json"]
            } else {
                ALLOWED_VIDEO_EXTS
            };
            Some(validated_path(out, exts, false)?)
        }
        None if mode == "plan" => Some(input.with_file_name(format!(
            "{}.censorplan.json",
            input
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default()
        ))),
        None => None,
    };
    if let Some(out) = &effective_output {
        args.push("--output".to_string());
        args.push(out.to_string_lossy().into_owned());
    }

    if matches!(&options.profile, Some(p) if !p.eq_ignore_ascii_case("AUTO")) {
        args.push("--profile".to_string());
        args.push(options.profile.clone().unwrap());
    }
    if let Some(ct) = &options.content_type {
        args.push("--content-type".to_string());
        args.push(ct.clone());
    }
    if let Some(t) = options.threshold {
        args.push("--threshold".to_string());
        args.push(format!("{}", t));
    }
    if options.no_audio {
        args.push("--no-audio".to_string());
    }
    if options.no_clip {
        args.push("--no-clip".to_string());
    }

    let mut child = Command::new("pureframe")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start pureframe: {}", e))?;

    let log = Arc::new(Mutex::new(VecDeque::with_capacity(LOG_TAIL_LINES)));
    // Take the pipes BEFORE the child is stored so nothing else can race on
    // them; each gets its own reader thread.
    if let Some(stdout) = child.stdout.take() {
        let log = Arc::clone(&log);
        thread::spawn(move || drain_pipe(stdout, log));
    }
    if let Some(stderr) = child.stderr.take() {
        let log = Arc::clone(&log);
        thread::spawn(move || drain_pipe(stderr, log));
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    let mut jobs = state.jobs.lock().map_err(|e| e.to_string())?;
    jobs.insert(
        job_id.clone(),
        Job {
            child,
            state: JobState::Running,
            exit_code: None,
            mode,
            output: effective_output.map(|p| p.to_string_lossy().into_owned()),
            log,
        },
    );

    Ok(job_id)
}

#[tauri::command]
fn job_status(id: String, state: State<'_, AppState>) -> Result<JobStatusResponse, String> {
    let mut jobs = state.jobs.lock().map_err(|e| e.to_string())?;
    let job = jobs.get_mut(&id).ok_or_else(|| "unknown job".to_string())?;

    if job.state == JobState::Running {
        if let Some(status) = job.child.try_wait().map_err(|e| e.to_string())? {
            job.exit_code = status.code();
            job.state = if status.success() {
                JobState::Done
            } else {
                JobState::Failed
            };
        }
    }

    let log_tail: Vec<String> = if let Ok(g) = job.log.lock() {
        g.iter()
            .rev()
            .take(LOG_TAIL_REPORT)
            .rev()
            .cloned()
            .collect()
    } else {
        Vec::new()
    };

    Ok(JobStatusResponse {
        state: job.state.clone(),
        exit_code: job.exit_code,
        mode: job.mode.clone(),
        output: job.output.clone(),
        log_tail,
    })
}

#[tauri::command]
fn cancel_job(id: String, state: State<'_, AppState>) -> Result<(), String> {
    let mut jobs = state.jobs.lock().map_err(|e| e.to_string())?;
    if let Some(mut job) = jobs.remove(&id) {
        // Best-effort: send SIGKILL/TerminateProcess, then reap so the
        // child does not become a zombie. We deliberately ignore errors
        // from `kill` (the process may have already exited) but we DO want
        // to wait so the OS reclaims the PID.
        let _ = job.child.kill();
        let _ = job.child.wait();
    }
    Ok(())
}

#[tauri::command]
async fn load_plan(path: String) -> Result<String, String> {
    let resolved = validated_path(&path, &["json"], true)?;

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let metadata = std::fs::metadata(&resolved).map_err(|e| e.to_string())?;
        if metadata.len() > MAX_PLAN_BYTES {
            return Err(format!(
                "plan file {} exceeds size cap ({} bytes)",
                resolved.display(),
                MAX_PLAN_BYTES
            ));
        }
        std::fs::read_to_string(&resolved).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn save_plan(path: String, plan_json: String) -> Result<(), String> {
    if plan_json.len() as u64 > MAX_PLAN_BYTES {
        return Err("plan JSON exceeds size cap".into());
    }
    // Validate JSON structure before persisting so we never write garbage.
    serde_json::from_str::<serde_json::Value>(&plan_json)
        .map_err(|e| format!("plan JSON is invalid: {}", e))?;
    let resolved = validated_path(&path, &["json"], false)?;

    tauri::async_runtime::spawn_blocking(move || {
        std::fs::write(&resolved, plan_json).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn extract_thumbnail(video_path: String, frame_idx: usize) -> Result<String, String> {
    let resolved = validated_path(&video_path, ALLOWED_VIDEO_EXTS, true)?;

    // ffmpeg call happens off the runtime thread so the UI stays responsive.
    let bytes = tauri::async_runtime::spawn_blocking(move || -> Result<Vec<u8>, String> {
        let output = Command::new("ffmpeg")
            .args([
                "-loglevel",
                "error",
                "-i",
                resolved.to_string_lossy().as_ref(),
                "-vf",
                &format!("select=eq(n\\,{})", frame_idx),
                "-vframes",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-",
            ])
            .output()
            .map_err(|e| e.to_string())?;

        if output.stdout.is_empty() {
            return Err("No frame data generated by ffmpeg".to_string());
        }
        Ok(output.stdout)
    })
    .await
    .map_err(|e| e.to_string())??;

    let b64 = general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:image/jpeg;base64,{}", b64))
}

struct AppState {
    jobs: Mutex<HashMap<String, Job>>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            jobs: Mutex::new(HashMap::new()),
        })
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            start_job,
            cancel_job,
            job_status,
            load_plan,
            save_plan,
            extract_thumbnail
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
