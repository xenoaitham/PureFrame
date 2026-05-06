import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Upload, Settings, List, Check, X, Edit, Eye, Shield, HardDrive, Filter, XCircle } from "lucide-react";

type Page = "onboarding" | "queue" | "plan-editor" | "settings";

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
  input_metadata: any;
  config_snapshot: any;
  shots: Shot[];
  verdicts: ShotVerdict[];
  total_censored_frames: number;
  total_blur_frames: number;
  generated_at: string;
}

export default function App() {
  const [page, setPage] = useState<Page>("onboarding");
  const [jobs, setJobs] = useState<any[]>([]);
  const [currentPlan, setCurrentPlan] = useState<CensorPlan | null>(null);
  const [currentPlanPath, setCurrentPlanPath] = useState<string>("");
  const [selectedShot, setSelectedShot] = useState<ShotVerdict | null>(null);
  const [thumbnailBase64, setThumbnailBase64] = useState<string>("");

  useEffect(() => {
    // Check if onboarding is needed
    const done = localStorage.getItem("onboarding_done");
    if (done) setPage("queue");
  }, []);

  const completeOnboarding = () => {
    localStorage.setItem("onboarding_done", "true");
    setPage("queue");
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    // Tauri drag and drop is usually handled via fileDropEnabled in tauri.conf.json
    // But we can also use HTML5 drag and drop if running in browser
  };

  const startJob = async (path: string, mode: "plan" | "apply" | "process") => {
    try {
      const jobId = await invoke("start_job", {
        inputPath: path,
        options: { output: null, no_audio: false, no_clip: false, mode },
      });
      setJobs([...jobs, { id: jobId, path, status: "RUNNING", mode }]);
    } catch (e) {
      alert("Error starting job: " + e);
    }
  };

  const loadPlan = async (path: string) => {
    try {
      const planJson: string = await invoke("load_plan", { path });
      setCurrentPlan(JSON.parse(planJson));
      setCurrentPlanPath(path);
      setPage("plan-editor");
    } catch (e) {
      alert("Error loading plan: " + e);
    }
  };

  const savePlan = async () => {
    if (!currentPlan) return;
    try {
      await invoke("save_plan", {
        path: currentPlanPath,
        planJson: JSON.stringify(currentPlan, null, 2),
      });
      alert("Plan saved successfully!");
    } catch (e) {
      alert("Error saving plan: " + e);
    }
  };

  const openShot = async (verdict: ShotVerdict) => {
    setSelectedShot(verdict);
    if (currentPlan) {
      const shot = currentPlan.shots.find((s) => s.index === verdict.shot_index);
      if (shot) {
        try {
          const b64: string = await invoke("extract_thumbnail", {
            videoPath: currentPlanPath.replace(".censorplan.json", ""), // Naive matching
            frameIdx: Math.floor((shot.start_frame + shot.end_frame) / 2),
          });
          setThumbnailBase64(b64);
        } catch (e) {
          console.error("No thumbnail:", e);
          setThumbnailBase64("");
        }
      }
    }
  };

  const updateVerdictAction = (action: string) => {
    if (!selectedShot || !currentPlan) return;
    const newPlan = { ...currentPlan };
    const v = newPlan.verdicts.find((x) => x.shot_index === selectedShot.shot_index);
    if (v) v.action = action;
    setCurrentPlan(newPlan);
    setSelectedShot({ ...selectedShot, action });
  };

  const renderOnboarding = () => (
    <div className="flex flex-col items-center justify-center h-full p-8 text-center bg-gray-50">
      <Shield className="w-24 h-24 text-blue-600 mb-6" />
      <h1 className="text-4xl font-bold mb-4 text-gray-800">Welcome to PureFrame</h1>
      <p className="text-xl text-gray-600 max-w-2xl mb-8">
        PureFrame is an automated video censorship tool. By using this software, you confirm that you are 
        modifying your own legal copies of media and are responsible for the output.
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
    <div 
      className="p-6 h-full flex flex-col"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold flex items-center gap-2"><List /> Job Queue</h2>
        <div className="flex gap-2">
          <button onClick={() => setPage("settings")} className="p-2 border rounded hover:bg-gray-100"><Settings /></button>
          <button onClick={() => {
            const p = prompt("Enter path to video or .censorplan.json:");
            if (p) {
              if (p.endsWith(".json")) loadPlan(p);
              else startJob(p, "process");
            }
          }} className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 flex items-center gap-2">
            <Upload size={18} /> Add File
          </button>
        </div>
      </div>

      <div className="flex-1 border-2 border-dashed border-gray-300 rounded-xl flex flex-col items-center justify-center text-gray-500 bg-gray-50/50">
        {jobs.length === 0 ? (
          <>
            <Upload className="w-16 h-16 mb-4 text-gray-400" />
            <p className="text-lg">Drag and drop video files here</p>
          </>
        ) : (
          <div className="w-full h-full p-4 flex flex-col gap-4 overflow-y-auto">
            {jobs.map((job) => (
              <div key={job.id} className="p-4 bg-white border rounded shadow-sm flex items-center justify-between">
                <div>
                  <h4 className="font-semibold">{job.path.split('/').pop()}</h4>
                  <p className="text-sm text-gray-500">Mode: {job.mode} • Status: {job.status}</p>
                </div>
                <div className="flex gap-2">
                  <button className="p-2 text-red-600 hover:bg-red-50 rounded"><XCircle size={20} /></button>
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
    const duration = currentPlan.input_metadata.duration_seconds;
    
    return (
      <div className="p-6 h-full flex flex-col">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Edit /> Plan Editor
          </h2>
          <div className="flex gap-2">
            <button onClick={() => setPage("queue")} className="px-4 py-2 border rounded hover:bg-gray-100">Back</button>
            <button onClick={savePlan} className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">Save Plan</button>
          </div>
        </div>

        {/* Timeline */}
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
              
              if (v.action === "NONE") color = "bg-gray-400"; // Whitelisted
              
              return (
                <div 
                  key={v.shot_index}
                  onClick={() => openShot(v)}
                  className={`absolute h-full ${color} cursor-pointer hover:opacity-80 transition`}
                  style={{ left: `${left}%`, width: `${Math.max(0.5, width)}%` }}
                  title={`${v.category} (${(v.confidence * 100).toFixed(0)}%)`}
                />
              );
            })}
          </div>
        </div>

        {/* Selected Shot Modal/Panel */}
        {selectedShot ? (
          <div className="flex-1 bg-white border rounded-lg p-6 shadow-sm flex flex-col">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-xl font-bold">Shot #{selectedShot.shot_index}</h3>
                <p className="text-gray-600">{selectedShot.category} ({(selectedShot.confidence * 100).toFixed(1)}%)</p>
                <p className="text-sm text-gray-500 italic mt-1">{selectedShot.reasoning}</p>
              </div>
              <button onClick={() => setSelectedShot(null)} className="p-2 hover:bg-gray-100 rounded"><X /></button>
            </div>
            
            <div className="flex-1 flex gap-6">
              <div className="flex-1 bg-gray-100 rounded flex items-center justify-center overflow-hidden">
                {thumbnailBase64 ? (
                  <img src={thumbnailBase64} alt="Thumbnail" className="max-h-full object-contain" />
                ) : (
                  <Eye className="w-12 h-12 text-gray-400" />
                )}
              </div>
              
              <div className="w-64 flex flex-col gap-4">
                <h4 className="font-semibold">Action</h4>
                
                <button 
                  onClick={() => updateVerdictAction("BLACK_BOX")}
                  className={`p-3 rounded border text-left flex items-center gap-2 ${selectedShot.action === 'BLACK_BOX' ? 'bg-blue-50 border-blue-500 text-blue-700' : 'hover:bg-gray-50'}`}
                >
                  <div className="w-4 h-4 bg-black rounded-sm" /> Keep (Black Box)
                </button>
                
                <button 
                  onClick={() => updateVerdictAction("FULL_FRAME_BLUR")}
                  className={`p-3 rounded border text-left flex items-center gap-2 ${selectedShot.action === 'FULL_FRAME_BLUR' ? 'bg-blue-50 border-blue-500 text-blue-700' : 'hover:bg-gray-50'}`}
                >
                  <Filter size={16} /> Force-blur
                </button>
                
                <button 
                  onClick={() => updateVerdictAction("NONE")}
                  className={`p-3 rounded border text-left flex items-center gap-2 ${selectedShot.action === 'NONE' ? 'bg-green-50 border-green-500 text-green-700' : 'hover:bg-gray-50'}`}
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
        <h2 className="text-2xl font-bold flex items-center gap-2"><Settings /> Settings</h2>
        <button onClick={() => setPage("queue")} className="px-4 py-2 border rounded hover:bg-gray-100">Back</button>
      </div>
      
      <div className="space-y-6 max-w-2xl">
        <div className="p-4 border rounded bg-white">
          <h3 className="font-semibold mb-2 flex items-center gap-2"><HardDrive size={18} /> Hardware Profile</h3>
          <select className="w-full p-2 border rounded">
            <option>auto</option>
            <option>cpu-only</option>
            <option>cuda-fast</option>
            <option>mac-silicon</option>
          </select>
        </div>
        
        <div className="p-4 border rounded bg-white">
          <h3 className="font-semibold mb-2">Default Sensitivity Threshold</h3>
          <input type="range" min="0" max="100" defaultValue="40" className="w-full" />
          <div className="flex justify-between text-sm text-gray-500 mt-1">
            <span>More Aggressive</span>
            <span>Balanced (0.4)</span>
            <span>More Permissive</span>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="h-screen w-screen bg-white text-gray-900 font-sans flex flex-col">
      {page === "onboarding" && renderOnboarding()}
      {page === "queue" && renderQueue()}
      {page === "plan-editor" && renderPlanEditor()}
      {page === "settings" && renderSettings()}
    </div>
  );
}
