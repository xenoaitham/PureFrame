use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

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

struct AppState {
    jobs: Mutex<HashMap<String, Child>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct StartJobOptions {
    output: Option<String>,
    no_audio: bool,
    no_clip: bool,
    /// One of: "plan", "apply", "process". Validated server-side.
    mode: String,
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

    let input = validated_path(&input_path, ALLOWED_VIDEO_EXTS, true)?;

    let mut args = vec![mode, input.to_string_lossy().into_owned()];

    if let Some(out) = &options.output {
        // Output may not exist yet; only the parent must.
        let out_path = if mode_is_plan(&args[0]) {
            validated_path(out, &["json"], false)?
        } else {
            validated_path(out, ALLOWED_VIDEO_EXTS, false)?
        };
        args.push("--output".to_string());
        args.push(out_path.to_string_lossy().into_owned());
    }

    if options.no_audio {
        args.push("--no-audio".to_string());
    }
    if options.no_clip {
        args.push("--no-clip".to_string());
    }

    let child = Command::new("pureframe")
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start pureframe: {}", e))?;

    let job_id = uuid::Uuid::new_v4().to_string();
    let mut jobs = state.jobs.lock().map_err(|e| e.to_string())?;
    jobs.insert(job_id.clone(), child);

    Ok(job_id)
}

fn mode_is_plan(mode: &str) -> bool {
    mode == "plan"
}

#[tauri::command]
fn cancel_job(id: String, state: State<'_, AppState>) -> Result<(), String> {
    let mut jobs = state.jobs.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = jobs.remove(&id) {
        // Best-effort: send SIGKILL/TerminateProcess, then reap so the
        // child does not become a zombie. We deliberately ignore errors
        // from `kill` (the process may have already exited) but we DO want
        // to wait so the OS reclaims the PID.
        let _ = child.kill();
        let _ = child.wait();
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
            load_plan,
            save_plan,
            extract_thumbnail
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
