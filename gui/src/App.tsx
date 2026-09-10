import { useState, useEffect, useCallback, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { Toaster, toast } from "sonner";
import {
  Upload,
  Settings,
  List,
  Check,
  X,
  Edit,
  Eye,
  Shield,
  HardDrive,
  Filter,
  XCircle,
  Loader2,
  FileJson,
  Film,
  AlertTriangle,
} from "lucide-react";

type Page = "onboarding" | "queue" | "plan-editor" | "settings";
type JobStatus = "RUNNING" | "DONE" | "FAILED" | "CANCELLED";
type JobMode = "plan" | "apply" | "process";
type HardwareProfile = "AUTO" | "HIGH" | "MEDIUM" | "LOW" | "CPU";
type ContentType = "live-action" | "animation" | "anime" | "low-light";

interface Job {
  id: string;
  path: string;
  status: JobStatus;
  mode: JobMode;
  /** Percent parsed from the CLI's progress output, if any. */
  progress: number | null;
  /** Most recent log line, shown under the file name while running. */
  lastLine: string | null;
  /** Where the plan / rendered video landed (reported by the backend). */
  output: string | null;
  exitCode: number | null;
}

interface JobStatusResponse {
  state: "running" | "done" | "failed";
  exit_code: number | null;
  mode: JobMode;
  output: string | null;
  log_tail: string[];
}

interface ShotVerdict {
  shot_index: number;
  action: string;
  category: string;
  confidence: number;
  reasoning: string;
}

interface Shot {
  index: number;
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
}

interface CensorPlan {
  pureframe_version: string;
  plan_version: number;
  input_metadata: {
    path?: string;
    source_path?: string;
    duration_seconds: number;
    [k: string]: unknown;
  };
  config_snapshot: Record<string, unknown>;
  shots: Shot[];
  verdicts: ShotVerdict[];
  total_censored_frames: number;
  total_blur_frames: number;
  generated_at: string;
}

interface AppSettings {
  hardware: HardwareProfile;
  threshold: number;
  contentType: ContentType;
}

const DEFAULT_SETTINGS: AppSettings = {
  hardware: "AUTO",
  threshold: 0.55,
  contentType: "live-action",
};
const VIDEO_EXTS = ["mkv", "mp4", "mov", "avi", "webm", "m4v", "ts", "wmv"];
const SETTINGS_KEY = "pureframe_settings";
const JOBS_KEY = "pureframe_jobs";
const POLL_INTERVAL_MS = 1200;
// Timeline scrubber: debounce for the ffmpeg frame fetch while dragging.
const SCRUB_DEBOUNCE_MS = 250;

function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    /* ignore */
  }
  return DEFAULT_SETTINGS;
}

function loadJobs(): Job[] {
  try {
    const raw = localStorage.getItem(JOBS_KEY);
    if (raw) {
      // Older sessions persisted jobs without the new status fields.
      return (JSON.parse(raw) as Partial<Job>[]).map(
        (j) =>
          ({
            progress: null,
            lastLine: null,
            output: null,
            exitCode: null,
            ...j,
          }) as Job,
      );
    }
  } catch {
    /* ignore */
  }
  return [];
}

function extOf(p: string): string {
  const i = p.lastIndexOf(".");
  return i >= 0 ? p.slice(i + 1).toLowerCase() : "";
}

function baseName(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}

function planSourcePath(plan: CensorPlan, planPath: string): string {
  const meta = plan.input_metadata || {};
  if (typeof meta.source_path === "string" && meta.source_path) return meta.source_path;
  if (typeof meta.path === "string" && meta.path) return meta.path;
  // Fallback: strip `.censorplan.json` suffix if present.
  return planPath.replace(/\.censorplan\.json$/i, "");
}

/** fps as a number, tolerating "30000/1001"-style fractions; 30 as fallback. */
function planFps(plan: CensorPlan): number {
  const raw = (plan.input_metadata as Record<string, unknown>).fps;
  if (typeof raw === "number" && raw > 0) return raw;
  if (typeof raw === "string") {
    const m = raw.match(/^(\d+)\/(\d+)$/);
    if (m) {
      const den = Number(m[2]);
      if (den > 0) return Number(m[1]) / den;
    }
    const n = Number(raw);
    if (n > 0) return n;
  }
  return 30;
}

function formatTimecode(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function StatusPill({ status }: { status: JobStatus }) {
  const styles: Record<JobStatus, { cls: string; label: string; icon?: React.ReactNode }> = {
    RUNNING: {
      cls: "bg-amber-500/10 text-amber-300 border-amber-500/30",
      label: "Running",
      icon: <Loader2 size={12} className="animate-spin" />,
    },
    DONE: {
      cls: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
      label: "Done",
      icon: <Check size={12} />,
    },
    FAILED: {
      cls: "bg-red-500/10 text-red-300 border-red-500/30",
      label: "Failed",
      icon: <AlertTriangle size={12} />,
    },
    CANCELLED: {
      cls: "bg-slate-500/10 text-slate-400 border-slate-500/30",
      label: "Cancelled",
    },
  };
  const s = styles[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${s.cls}`}
    >
      {s.icon}
      {s.label}
    </span>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("onboarding");
  const [jobs, setJobs] = useState<Job[]>(() => loadJobs());
  const [currentPlan, setCurrentPlan] = useState<CensorPlan | null>(null);
  const [currentPlanPath, setCurrentPlanPath] = useState<string>("");
  const [selectedShot, setSelectedShot] = useState<ShotVerdict | null>(null);
  const [thumbnailBase64, setThumbnailBase64] = useState<string>("");
  const [scrubTime, setScrubTime] = useState<number | null>(null);
  const [scrubThumb, setScrubThumb] = useState<string>("");
  const [scrubLoading, setScrubLoading] = useState(false);
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());

  // Monotonic token so a slow thumbnail fetch can't overwrite a newer one.
  const scrubRequestId = useRef(0);
  const scrubTimer = useRef<number | null>(null);

  // Latest jobs for the poller without re-arming the interval on each update.
  const jobsRef = useRef<Job[]>([]);
  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  useEffect(() => {
    const done = localStorage.getItem("onboarding_done");
    if (done) setPage("queue");
  }, []);

  useEffect(() => {
    localStorage.setItem(JOBS_KEY, JSON.stringify(jobs));
  }, [jobs]);

  useEffect(() => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }, [settings]);

  const completeOnboarding = () => {
    localStorage.setItem("onboarding_done", "true");
    setPage("queue");
  };

  const startJob = useCallback(
    async (path: string, mode: JobMode) => {
      try {
        const jobId = await invoke<string>("start_job", {
          inputPath: path,
          options: {
            output: null,
            no_audio: false,
            no_clip: false,
            mode,
            profile: settings.hardware,
            content_type: settings.contentType,
            threshold: settings.threshold,
          },
        });
        setJobs((prev) => [
          ...prev,
          {
            id: jobId,
            path,
            status: "RUNNING",
            mode,
            progress: null,
            lastLine: null,
            output: null,
            exitCode: null,
          },
        ]);
        toast.success(`Started ${mode} job for ${baseName(path)}`);
      } catch (e) {
        toast.error(`Failed to start job: ${String(e)}`);
      }
    },
    [settings],
  );

  const loadPlan = useCallback(async (path: string) => {
    try {
      const planJson = await invoke<string>("load_plan", { path });
      setCurrentPlan(JSON.parse(planJson) as CensorPlan);
      setCurrentPlanPath(path);
      setPage("plan-editor");
    } catch (e) {
      toast.error(`Failed to load plan: ${String(e)}`);
    }
  }, []);

  const cancelJob = useCallback(async (id: string) => {
    try {
      await invoke("cancel_job", { id });
      setJobs((prev) =>
        prev.map((j) => (j.id === id ? { ...j, status: "CANCELLED" } : j)),
      );
      toast.success("Job cancelled");
    } catch (e) {
      toast.error(`Failed to cancel job: ${String(e)}`);
    }
  }, []);

  // Poll the backend for every RUNNING job: status, exit code, output path,
  // and a progress line parsed from the CLI's log output.
  useEffect(() => {
    const t = setInterval(async () => {
      const running = jobsRef.current.filter((j) => j.status === "RUNNING");
      if (running.length === 0) return;

      const results = await Promise.all(
        running.map(async (j) => {
          try {
            const s = await invoke<JobStatusResponse>("job_status", { id: j.id });
            return { id: j.id, s };
          } catch {
            return null;
          }
        }),
      );

      const transitions: Job[] = [];
      setJobs((prev) =>
        prev.map((j) => {
          const r = results.find((x) => x && x.id === j.id);
          if (!r) return j;
          const { s } = r;

          const progressLine = [...s.log_tail].reverse().find((l) => /\d{1,3}%/.test(l));
          const pct = progressLine?.match(/(\d{1,3})%/);
          const next: Job = {
            ...j,
            progress: pct ? parseInt(pct[1], 10) : j.progress,
            lastLine: s.log_tail[s.log_tail.length - 1] ?? j.lastLine,
            output: s.output ?? j.output,
          };

          if (s.state !== "running") {
            next.status = s.state === "done" ? "DONE" : "FAILED";
            next.exitCode = s.exit_code;
            transitions.push(next);
          }
          return next;
        }),
      );

      for (const t2 of transitions) {
        if (t2.status === "DONE") {
          toast.success(`${baseName(t2.path)} finished`);
        } else {
          toast.error(
            `${baseName(t2.path)} failed (exit ${t2.exitCode ?? "?"}): ${t2.lastLine ?? "no output"}`,
          );
        }
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(t);
  }, []);

  // Route a path to the right handler (plan loader vs. start job).
  const dispatchPath = useCallback(
    (p: string) => {
      const ext = extOf(p);
      if (ext === "json") {
        void loadPlan(p);
      } else if (VIDEO_EXTS.includes(ext)) {
        void startJob(p, "process");
      } else {
        toast.error(`Unsupported file type: .${ext}`);
      }
    },
    [loadPlan, startJob],
  );

  // Native drag-and-drop from the OS — only the webview event exposes real paths.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    void getCurrentWebview()
      .onDragDropEvent((evt) => {
        if (evt.payload.type === "drop") {
          for (const p of evt.payload.paths) dispatchPath(p);
        }
      })
      .then((fn) => {
        unlisten = fn;
      });
    return () => {
      unlisten?.();
    };
  }, [dispatchPath]);

  const pickFile = async () => {
    try {
      const result = await openDialog({
        multiple: false,
        filters: [
          { name: "Video or Plan", extensions: [...VIDEO_EXTS, "json"] },
          { name: "Video", extensions: VIDEO_EXTS },
          { name: "Censor Plan", extensions: ["json"] },
        ],
      });
      if (typeof result === "string") dispatchPath(result);
    } catch (e) {
      toast.error(`Dialog error: ${String(e)}`);
    }
  };

  const savePlan = async () => {
    if (!currentPlan) return;
    try {
      await invoke("save_plan", {
        path: currentPlanPath,
        planJson: JSON.stringify(currentPlan, null, 2),
      });
      toast.success("Plan saved");
    } catch (e) {
      toast.error(`Save failed: ${String(e)}`);
    }
  };

  const openShot = async (verdict: ShotVerdict) => {
    setSelectedShot(verdict);
    if (!currentPlan) return;
    const shot = currentPlan.shots.find((s) => s.index === verdict.shot_index);
    if (!shot) return;
    try {
      const b64 = await invoke<string>("extract_thumbnail", {
        videoPath: planSourcePath(currentPlan, currentPlanPath),
        frameIdx: Math.floor((shot.start_frame + shot.end_frame) / 2),
      });
      setThumbnailBase64(b64);
    } catch (e) {
      console.error("No thumbnail:", e);
      setThumbnailBase64("");
    }
    // The scrubber follows selection so the bar always reflects what the
    // preview shows (the shot thumbnail fetch above already ran).
    setScrubTime((shot.start_time + shot.end_time) / 2);
  };

  /** Seek the timeline scrubber to *time* seconds and fetch its frame.

   * The position label updates immediately; the ffmpeg frame fetch is
   * debounced so a drag doesn't spawn a process per pixel. A monotonic
   * request id lets an older (slower) fetch be discarded when a newer one
   * has already landed.
   */
  const seekTo = (time: number) => {
    if (!currentPlan) return;
    const duration = currentPlan.input_metadata.duration_seconds || 1;
    const clamped = Math.min(Math.max(time, 0), duration);
    setScrubTime(clamped);
    if (scrubTimer.current !== null) window.clearTimeout(scrubTimer.current);
    scrubTimer.current = window.setTimeout(() => {
      const requestId = ++scrubRequestId.current;
      setScrubLoading(true);
      invoke<string>("extract_thumbnail", {
        videoPath: planSourcePath(currentPlan, currentPlanPath),
        frameIdx: Math.floor(clamped * planFps(currentPlan)),
      })
        .then((b64) => {
          if (scrubRequestId.current === requestId) setScrubThumb(b64);
        })
        .catch((e) => {
          console.error("No scrub thumbnail:", e);
          if (scrubRequestId.current === requestId) setScrubThumb("");
        })
        .finally(() => {
          if (scrubRequestId.current === requestId) setScrubLoading(false);
        });
    }, SCRUB_DEBOUNCE_MS);
  };

  const updateVerdictAction = (action: string) => {
    if (!selectedShot || !currentPlan) return;
    // Immutable update — replace verdicts array rather than mutating in place.
    const verdicts = currentPlan.verdicts.map((v) =>
      v.shot_index === selectedShot.shot_index ? { ...v, action } : v,
    );
    setCurrentPlan({ ...currentPlan, verdicts });
    setSelectedShot({ ...selectedShot, action });
  };

  const renderOnboarding = () => (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-slate-950">
      <div className="w-20 h-20 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center mb-8">
        <Shield className="w-10 h-10 text-sky-400" />
      </div>
      <h1 className="text-4xl font-bold mb-4 text-slate-100">Welcome to PureFrame</h1>
      <p className="text-lg text-slate-400 max-w-2xl mb-10 leading-relaxed">
        PureFrame blurs explicit visuals in your own movie files — locally, offline,
        without cutting a single second. By using this software, you confirm that you
        are modifying your own legal copies of media and are responsible for the
        output.
      </p>
      <button
        onClick={completeOnboarding}
        className="px-8 py-3.5 bg-sky-600 text-white rounded-xl text-base font-semibold hover:bg-sky-500 transition shadow-lg shadow-sky-950/50"
      >
        I Agree, Let's Go
      </button>
    </div>
  );

  const renderJobRow = (job: Job) => (
    <div
      key={job.id}
      className="p-4 bg-slate-900 border border-slate-800 rounded-xl flex items-center gap-4"
    >
      <div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center shrink-0">
        {job.mode === "plan" ? (
          <FileJson size={17} className="text-sky-400" />
        ) : (
          <Film size={17} className="text-sky-400" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3">
          <h4 className="font-semibold text-slate-100 truncate">{baseName(job.path)}</h4>
          <StatusPill status={job.status} />
          <span className="text-xs uppercase tracking-wide text-slate-500">
            {job.mode}
          </span>
        </div>
        {job.status === "RUNNING" && (
          <div className="mt-2">
            <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
              <div
                className="h-full bg-sky-500 rounded-full transition-all duration-500"
                style={{ width: `${job.progress ?? 4}%` }}
              />
            </div>
            {job.lastLine && (
              <p className="mt-1.5 text-xs text-slate-500 font-mono truncate">
                {job.lastLine}
              </p>
            )}
          </div>
        )}
        {job.status === "DONE" && job.output && (
          <p className="mt-1 text-xs text-slate-500 truncate">
            {job.mode === "plan" ? "Plan: " : "Output: "}
            {job.output}
          </p>
        )}
        {job.status === "FAILED" && (
          <p className="mt-1 text-xs text-red-400/80 truncate">
            {job.lastLine ?? `Exited with code ${job.exitCode ?? "?"}`}
          </p>
        )}
      </div>
      <div className="flex gap-2 shrink-0">
        {job.status === "RUNNING" && (
          <button
            onClick={() => void cancelJob(job.id)}
            className="p-2 text-red-400 hover:bg-red-500/10 rounded-lg transition"
            aria-label="Cancel job"
          >
            <XCircle size={18} />
          </button>
        )}
        {job.status === "DONE" && job.mode === "plan" && job.output && (
          <button
            onClick={() => void loadPlan(job.output!)}
            className="px-3 py-1.5 text-sm bg-sky-600/90 hover:bg-sky-500 text-white rounded-lg transition"
          >
            Review plan
          </button>
        )}
      </div>
    </div>
  );

  const renderQueue = () => (
    <div className="p-6 h-full flex flex-col bg-slate-950">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2.5">
          <List className="text-sky-400" /> Job Queue
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setPage("settings")}
            className="p-2 border border-slate-700 rounded-lg text-slate-300 hover:bg-slate-800 transition"
            aria-label="Settings"
          >
            <Settings />
          </button>
          <button
            onClick={pickFile}
            className="px-4 py-2 bg-sky-600 text-white rounded-lg hover:bg-sky-500 flex items-center gap-2 transition"
          >
            <Upload size={18} /> Add File
          </button>
        </div>
      </div>

      <div className="flex-1 border-2 border-dashed border-slate-800 rounded-2xl flex flex-col items-center justify-center text-slate-500 bg-slate-900/30">
        {jobs.length === 0 ? (
          <>
            <Upload className="w-14 h-14 mb-4 text-slate-600" />
            <p className="text-lg text-slate-400">Drag and drop video files here</p>
            <p className="text-sm">or use “Add File” to browse</p>
          </>
        ) : (
          <div className="w-full h-full p-4 flex flex-col gap-3 overflow-y-auto">
            {jobs.map(renderJobRow)}
          </div>
        )}
      </div>
    </div>
  );

  const renderPlanEditor = () => {
    if (!currentPlan) return null;
    const duration = currentPlan.input_metadata.duration_seconds || 1;

    return (
      <div className="p-6 h-full flex flex-col bg-slate-950">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2.5">
            <Edit className="text-sky-400" /> Plan Editor
          </h2>
          <div className="flex gap-2">
            <button
              onClick={() => setPage("queue")}
              className="px-4 py-2 border border-slate-700 rounded-lg text-slate-300 hover:bg-slate-800 transition"
            >
              Back
            </button>
            <button
              onClick={savePlan}
              className="px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-500 transition"
            >
              Save Plan
            </button>
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl mb-6">
          <h3 className="font-semibold mb-3 text-slate-200">Timeline</h3>
          <div className="relative h-12 bg-slate-800 rounded-lg overflow-hidden">
            {currentPlan.verdicts.map((v) => {
              const shot = currentPlan.shots.find((s) => s.index === v.shot_index);
              if (!shot) return null;

              const left = (shot.start_time / duration) * 100;
              const width = ((shot.end_time - shot.start_time) / duration) * 100;

              let color = "bg-sky-500";
              // SEXUAL_CONTEXT_NO_NUDITY contains "NUDITY" — the sexual
              // check must come first or that category renders red.
              if (v.category.includes("SEXUAL")) color = "bg-orange-500";
              else if (v.category.includes("NUDITY")) color = "bg-red-500";
              else if (v.category.includes("KISS")) color = "bg-yellow-500";
              if (v.action === "NONE") color = "bg-slate-600";

              return (
                <div
                  key={v.shot_index}
                  onClick={() => void openShot(v)}
                  className={`absolute h-full ${color} opacity-80 cursor-pointer hover:opacity-100 transition`}
                  style={{ left: `${left}%`, width: `${Math.max(0.5, width)}%` }}
                  title={`${v.category} (${(v.confidence * 100).toFixed(0)}%)`}
                />
              );
            })}
          </div>

          <div className="mt-3 flex items-center gap-3">
            <input
              aria-label="Scrub timeline"
              type="range"
              min={0}
              max={duration}
              step={0.05}
              value={scrubTime ?? 0}
              onChange={(e) => seekTo(Number(e.target.value))}
              className="flex-1 accent-sky-500"
            />
            <span
              data-testid="scrub-timecode"
              className="text-xs text-slate-400 font-mono tabular-nums whitespace-nowrap"
            >
              {scrubTime !== null
                ? `${formatTimecode(scrubTime)} / ${formatTimecode(duration)}`
                : formatTimecode(duration)}
            </span>
          </div>

          {scrubTime !== null && (
            <div className="mt-3 bg-slate-950/60 border border-slate-800 rounded-lg p-2 flex items-center justify-center min-h-[120px]">
              {scrubThumb ? (
                <img
                  src={scrubThumb}
                  alt="Scrub preview"
                  className="max-h-48 object-contain"
                />
              ) : scrubLoading ? (
                <Loader2 className="animate-spin text-slate-500" />
              ) : (
                <span className="text-slate-600 text-sm">
                  No preview at {formatTimecode(scrubTime)}
                </span>
              )}
            </div>
          )}
        </div>

        {selectedShot ? (
          <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-6 flex flex-col">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-xl font-bold text-slate-100">
                  Shot #{selectedShot.shot_index}
                </h3>
                <p className="text-slate-400">
                  {selectedShot.category} ({(selectedShot.confidence * 100).toFixed(1)}%)
                </p>
                <p className="text-sm text-slate-500 italic mt-1">
                  {selectedShot.reasoning}
                </p>
              </div>
              <button
                onClick={() => setSelectedShot(null)}
                className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 transition"
                aria-label="Close shot"
              >
                <X />
              </button>
            </div>

            <div className="flex-1 flex gap-6 min-h-0">
              <div className="flex-1 bg-slate-950/60 border border-slate-800 rounded-lg flex items-center justify-center overflow-hidden">
                {thumbnailBase64 ? (
                  <img
                    src={thumbnailBase64}
                    alt="Thumbnail"
                    className="max-h-full object-contain"
                  />
                ) : (
                  <Eye className="w-12 h-12 text-slate-700" />
                )}
              </div>

              <div className="w-64 flex flex-col gap-3 shrink-0">
                <h4 className="font-semibold text-slate-200">Action</h4>

                <button
                  onClick={() => updateVerdictAction("BLUR")}
                  className={`p-3 rounded-lg border text-left flex items-center gap-2 transition ${
                    selectedShot.action === "BLUR"
                      ? "bg-sky-500/10 border-sky-500 text-sky-300"
                      : "border-slate-700 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  <Filter size={16} /> Localized Blur
                </button>

                <button
                  onClick={() => updateVerdictAction("FULL_FRAME_BLUR")}
                  className={`p-3 rounded-lg border text-left flex items-center gap-2 transition ${
                    selectedShot.action === "FULL_FRAME_BLUR"
                      ? "bg-sky-500/10 border-sky-500 text-sky-300"
                      : "border-slate-700 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  <Filter size={16} /> Force Full-Frame Blur
                </button>

                <button
                  onClick={() => updateVerdictAction("NONE")}
                  className={`p-3 rounded-lg border text-left flex items-center gap-2 transition ${
                    selectedShot.action === "NONE"
                      ? "bg-emerald-500/10 border-emerald-500 text-emerald-300"
                      : "border-slate-700 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  <Check size={16} /> Whitelist (Ignore)
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 border border-slate-800 rounded-xl flex items-center justify-center text-slate-600 bg-slate-900/30">
            Click a colored segment on the timeline to edit.
          </div>
        )}
      </div>
    );
  };

  const renderSettings = () => (
    <div className="p-6 h-full flex flex-col bg-slate-950">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2.5">
          <Settings className="text-sky-400" /> Settings
        </h2>
        <button
          onClick={() => setPage("queue")}
          className="px-4 py-2 border border-slate-700 rounded-lg text-slate-300 hover:bg-slate-800 transition"
        >
          Back
        </button>
      </div>

      <div className="space-y-4 max-w-2xl">
        <div className="p-4 border border-slate-800 rounded-xl bg-slate-900">
          <h3 className="font-semibold mb-3 flex items-center gap-2 text-slate-200">
            <HardDrive size={18} className="text-sky-400" /> Hardware Profile
          </h3>
          <select
            className="w-full p-2.5 border border-slate-700 rounded-lg bg-slate-950 text-slate-200 focus:outline-none focus:border-sky-500"
            value={settings.hardware}
            onChange={(e) =>
              setSettings((s) => ({ ...s, hardware: e.target.value as HardwareProfile }))
            }
          >
            <option value="AUTO">Auto-detect</option>
            <option value="HIGH">High (CUDA, 8GB+ VRAM)</option>
            <option value="MEDIUM">Medium (CUDA, 4-8GB VRAM)</option>
            <option value="LOW">Low (CUDA, &lt; 4GB VRAM)</option>
            <option value="CPU">CPU only</option>
          </select>
          <p className="text-xs text-slate-500 mt-2">
            Applied to every job you add from now on.
          </p>
        </div>

        <div className="p-4 border border-slate-800 rounded-xl bg-slate-900">
          <h3 className="font-semibold mb-3 text-slate-200">Content Type</h3>
          <select
            className="w-full p-2.5 border border-slate-700 rounded-lg bg-slate-950 text-slate-200 focus:outline-none focus:border-sky-500"
            value={settings.contentType}
            onChange={(e) =>
              setSettings((s) => ({ ...s, contentType: e.target.value as ContentType }))
            }
          >
            <option value="live-action">Live-action movies &amp; TV (default)</option>
            <option value="animation">Animation (fewer false positives)</option>
            <option value="anime">Anime (tuned thresholds)</option>
            <option value="low-light">Low-light content (more sensitive)</option>
          </select>
        </div>

        <div className="p-4 border border-slate-800 rounded-xl bg-slate-900">
          <h3 className="font-semibold mb-3 text-slate-200">Detection Threshold</h3>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={settings.threshold}
            onChange={(e) =>
              setSettings((s) => ({ ...s, threshold: parseFloat(e.target.value) }))
            }
            className="w-full accent-sky-500"
          />
          <div className="flex justify-between text-sm text-slate-500 mt-1">
            <span>More Aggressive (0.0)</span>
            <span className="text-sky-400 font-medium">
              {settings.threshold.toFixed(2)}
            </span>
            <span>More Permissive (1.0)</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="h-screen w-screen bg-slate-950 text-slate-100 font-sans flex flex-col">
      <Toaster position="top-right" theme="dark" richColors />
      {page === "onboarding" && renderOnboarding()}
      {page === "queue" && renderQueue()}
      {page === "plan-editor" && renderPlanEditor()}
      {page === "settings" && renderSettings()}
    </div>
  );
}
