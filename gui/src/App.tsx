import { useState, useEffect, useCallback } from "react";
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
} from "lucide-react";

type Page = "onboarding" | "queue" | "plan-editor" | "settings";
type JobStatus = "RUNNING" | "DONE" | "ERROR" | "CANCELLED";
type JobMode = "plan" | "apply" | "process";
type HardwareProfile = "AUTO" | "HIGH" | "MEDIUM" | "LOW" | "CPU";

interface Job {
  id: string;
  path: string;
  status: JobStatus;
  mode: JobMode;
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
}

const DEFAULT_SETTINGS: AppSettings = { hardware: "AUTO", threshold: 0.4 };
const VIDEO_EXTS = ["mkv", "mp4", "mov", "avi", "webm", "m4v", "ts", "wmv"];
const SETTINGS_KEY = "pureframe_settings";
const JOBS_KEY = "pureframe_jobs";

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
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return [];
}

function extOf(p: string): string {
  const i = p.lastIndexOf(".");
  return i >= 0 ? p.slice(i + 1).toLowerCase() : "";
}

function planSourcePath(plan: CensorPlan, planPath: string): string {
  const meta = plan.input_metadata || {};
  if (typeof meta.source_path === "string" && meta.source_path) return meta.source_path;
  if (typeof meta.path === "string" && meta.path) return meta.path;
  // Fallback: strip `.censorplan.json` suffix if present.
  return planPath.replace(/\.censorplan\.json$/i, "");
}

export default function App() {
  const [page, setPage] = useState<Page>("onboarding");
  const [jobs, setJobs] = useState<Job[]>(() => loadJobs());
  const [currentPlan, setCurrentPlan] = useState<CensorPlan | null>(null);
  const [currentPlanPath, setCurrentPlanPath] = useState<string>("");
  const [selectedShot, setSelectedShot] = useState<ShotVerdict | null>(null);
  const [thumbnailBase64, setThumbnailBase64] = useState<string>("");
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());

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
          options: { output: null, no_audio: false, no_clip: false, mode },
        });
        setJobs((prev) => [...prev, { id: jobId, path, status: "RUNNING", mode }]);
        toast.success(`Started ${mode} job for ${path.split(/[\\/]/).pop()}`);
      } catch (e) {
        toast.error(`Failed to start job: ${String(e)}`);
      }
    },
    [],
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

  // Native drag-and-drop from the OS - only the webview event exposes real paths.
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
  };

  const updateVerdictAction = (action: string) => {
    if (!selectedShot || !currentPlan) return;
    // Immutable update - replace verdicts array rather than mutating in place.
    const verdicts = currentPlan.verdicts.map((v) =>
      v.shot_index === selectedShot.shot_index ? { ...v, action } : v,
    );
    setCurrentPlan({ ...currentPlan, verdicts });
    setSelectedShot({ ...selectedShot, action });
  };

  const renderOnboarding = () => (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-gray-50">
      <Shield className="w-24 h-24 text-blue-600 mb-6" />
      <h1 className="text-4xl font-bold mb-4 text-gray-800">Welcome to PureFrame</h1>
      <p className="text-xl text-gray-600 max-w-2xl mb-8">
        PureFrame is an automated video censorship tool. By using this software, you confirm
        that you are modifying your own legal copies of media and are responsible for the
        output.
      </p>
      <button
        onClick={completeOnboarding}
        className="px-8 py-4 bg-blue-600 text-white rounded-lg text-lg font-semibold hover:bg-blue-700 transition"
      >
        I Agree, Let's Go
      </button>
    </div>
  );

  const renderQueue = () => (
    <div className="p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <List /> Job Queue
        </h2>
        <div className="flex gap-2">
          <button
            onClick={() => setPage("settings")}
            className="p-2 border rounded hover:bg-gray-100"
            aria-label="Settings"
          >
            <Settings />
          </button>
          <button
            onClick={pickFile}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center gap-2"
          >
            <Upload size={18} /> Add File
          </button>
        </div>
      </div>

      <div className="flex-1 border-2 border-dashed border-gray-300 rounded-xl flex flex-col items-center justify-center text-gray-500 bg-gray-50/50">
        {jobs.length === 0 ? (
          <>
            <Upload className="w-16 h-16 mb-4 text-gray-400" />
            <p className="text-lg">Drag and drop video files here</p>
            <p className="text-sm">or use “Add File” to browse</p>
          </>
        ) : (
          <div className="w-full h-full p-4 flex flex-col gap-4 overflow-y-auto">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="p-4 bg-white border rounded shadow-sm flex items-center justify-between"
              >
                <div>
                  <h4 className="font-semibold">{job.path.split(/[\\/]/).pop()}</h4>
                  <p className="text-sm text-gray-500">
                    Mode: {job.mode} • Status: {job.status}
                  </p>
                </div>
                <div className="flex gap-2">
                  {job.status === "RUNNING" && (
                    <button
                      onClick={() => cancelJob(job.id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded"
                      aria-label="Cancel job"
                    >
                      <XCircle size={20} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const renderPlanEditor = () => {
    if (!currentPlan) return null;
    const duration = currentPlan.input_metadata.duration_seconds || 1;

    return (
      <div className="p-6 h-full flex flex-col">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Edit /> Plan Editor
          </h2>
          <div className="flex gap-2">
            <button
              onClick={() => setPage("queue")}
              className="px-4 py-2 border rounded hover:bg-gray-100"
            >
              Back
            </button>
            <button
              onClick={savePlan}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Save Plan
            </button>
          </div>
        </div>

        <div className="bg-gray-100 p-4 rounded-lg mb-6">
          <h3 className="font-semibold mb-2">Timeline</h3>
          <div className="relative h-12 bg-gray-300 rounded overflow-hidden">
            {currentPlan.verdicts.map((v) => {
              const shot = currentPlan.shots.find((s) => s.index === v.shot_index);
              if (!shot) return null;

              const left = (shot.start_time / duration) * 100;
              const width = ((shot.end_time - shot.start_time) / duration) * 100;

              let color = "bg-blue-500";
              if (v.category.includes("NUDITY")) color = "bg-red-500";
              else if (v.category.includes("SEXUAL")) color = "bg-orange-500";
              else if (v.category.includes("KISS")) color = "bg-yellow-500";
              if (v.action === "NONE") color = "bg-gray-400";

              return (
                <div
                  key={v.shot_index}
                  onClick={() => void openShot(v)}
                  className={`absolute h-full ${color} cursor-pointer hover:opacity-80 transition`}
                  style={{ left: `${left}%`, width: `${Math.max(0.5, width)}%` }}
                  title={`${v.category} (${(v.confidence * 100).toFixed(0)}%)`}
                />
              );
            })}
          </div>
        </div>

        {selectedShot ? (
          <div className="flex-1 bg-white border rounded-lg p-6 shadow-sm flex flex-col">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-xl font-bold">Shot #{selectedShot.shot_index}</h3>
                <p className="text-gray-600">
                  {selectedShot.category} ({(selectedShot.confidence * 100).toFixed(1)}%)
                </p>
                <p className="text-sm text-gray-500 italic mt-1">{selectedShot.reasoning}</p>
              </div>
              <button
                onClick={() => setSelectedShot(null)}
                className="p-2 hover:bg-gray-100 rounded"
                aria-label="Close shot"
              >
                <X />
              </button>
            </div>

            <div className="flex-1 flex gap-6">
              <div className="flex-1 bg-gray-100 rounded flex items-center justify-center overflow-hidden">
                {thumbnailBase64 ? (
                  <img
                    src={thumbnailBase64}
                    alt="Thumbnail"
                    className="max-h-full object-contain"
                  />
                ) : (
                  <Eye className="w-12 h-12 text-gray-400" />
                )}
              </div>

              <div className="w-64 flex flex-col gap-4">
                <h4 className="font-semibold">Action</h4>

                <button
                  onClick={() => updateVerdictAction("BLUR")}
                  className={`p-3 rounded border text-left flex items-center gap-2 ${
                    selectedShot.action === "BLUR"
                      ? "bg-blue-50 border-blue-500 text-blue-700"
                      : "hover:bg-gray-50"
                  }`}
                >
                  <Filter size={16} /> Localized Blur
                </button>

                <button
                  onClick={() => updateVerdictAction("FULL_FRAME_BLUR")}
                  className={`p-3 rounded border text-left flex items-center gap-2 ${
                    selectedShot.action === "FULL_FRAME_BLUR"
                      ? "bg-blue-50 border-blue-500 text-blue-700"
                      : "hover:bg-gray-50"
                  }`}
                >
                  <Filter size={16} /> Force Full-Frame Blur
                </button>

                <button
                  onClick={() => updateVerdictAction("NONE")}
                  className={`p-3 rounded border text-left flex items-center gap-2 ${
                    selectedShot.action === "NONE"
                      ? "bg-green-50 border-green-500 text-green-700"
                      : "hover:bg-gray-50"
                  }`}
                >
                  <Check size={16} /> Whitelist (Ignore)
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 border rounded-lg flex items-center justify-center text-gray-400 bg-gray-50">
            Click a colored segment on the timeline to edit.
          </div>
        )}
      </div>
    );
  };

  const renderSettings = () => (
    <div className="p-6 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Settings /> Settings
        </h2>
        <button
          onClick={() => setPage("queue")}
          className="px-4 py-2 border rounded hover:bg-gray-100"
        >
          Back
        </button>
      </div>

      <div className="space-y-6 max-w-2xl">
        <div className="p-4 border rounded bg-white">
          <h3 className="font-semibold mb-2 flex items-center gap-2">
            <HardDrive size={18} /> Hardware Profile
          </h3>
          <select
            className="w-full p-2 border rounded"
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
        </div>

        <div className="p-4 border rounded bg-white">
          <h3 className="font-semibold mb-2">Default Sensitivity Threshold</h3>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={settings.threshold}
            onChange={(e) =>
              setSettings((s) => ({ ...s, threshold: parseFloat(e.target.value) }))
            }
            className="w-full"
          />
          <div className="flex justify-between text-sm text-gray-500 mt-1">
            <span>More Aggressive (0.0)</span>
            <span>Current: {settings.threshold.toFixed(2)}</span>
            <span>More Permissive (1.0)</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="h-screen w-screen bg-white text-gray-900 font-sans flex flex-col">
      <Toaster position="top-right" richColors />
      {page === "onboarding" && renderOnboarding()}
      {page === "queue" && renderQueue()}
      {page === "plan-editor" && renderPlanEditor()}
      {page === "settings" && renderSettings()}
    </div>
  );
}
