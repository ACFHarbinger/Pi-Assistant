import { useEffect, useState } from "react";
import { useTrainingStore, TrainingRun } from "../stores/trainingStore";

export function TrainingDashboard({ onClose }: { onClose: () => void }) {
  const {
    runs,
    fetchRuns,
    startTraining,
    stopTraining,
    deployModel,
    isLoading,
    error,
    initProgressSocket,
  } = useTrainingStore();
  const [showNewRun, setShowNewRun] = useState(false);
  const [deployRunId, setDeployRunId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  // Initialize progress socket
  useEffect(() => {
    initProgressSocket();
  }, []);

  // Auto-refresh runs
  useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6">
      <div className="glass rounded-2xl w-full max-w-5xl h-[80vh] flex flex-col overflow-hidden border border-white/20 shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-purple-500/20 flex items-center justify-center">
              <span className="text-xl">🎓</span>
            </div>
            <div>
              <h2 className="text-xl font-bold">Training Dashboard</h2>
              <p className="text-xs text-gray-400">Train, Evaluate, Deploy</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden flex">
          {/* Sidebar / List */}
          <div className="w-1/3 border-r border-white/10 flex flex-col bg-black/20">
            <div className="p-4 border-b border-white/5 flex justify-between items-center">
              <h3 className="font-semibold text-sm text-gray-300">
                Run History
              </h3>
              <button
                onClick={() => setShowNewRun(true)}
                className="px-3 py-1.5 bg-primary-600 rounded-lg text-xs font-medium hover:bg-primary-500 transition-colors"
              >
                + New Run
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
              {runs.length === 0 && !isLoading && (
                <div className="text-center text-gray-500 py-8 text-xs">
                  No runs yet
                </div>
              )}
              {runs.map((run) => (
                <RunCard
                  key={run.run_id}
                  run={run}
                  isSelected={selectedRunId === run.run_id}
                  onSelect={() => setSelectedRunId(run.run_id)}
                  onDeploy={() => setDeployRunId(run.run_id)}
                  onStop={() => stopTraining(run.run_id)}
                />
              ))}
            </div>
          </div>

          {/* Details / New Run Form */}
          <div className="flex-1 p-6 overflow-y-auto relative">
            {error && (
              <div className="mb-4 p-3 bg-red-500/20 border border-red-500/30 rounded-lg text-red-200 text-sm">
                Error: {error}
              </div>
            )}

            {showNewRun ? (
              <NewRunForm
                onCancel={() => setShowNewRun(false)}
                onSubmit={async (cfg) => {
                  await startTraining(cfg);
                  setShowNewRun(false);
                }}
              />
            ) : deployRunId ? (
              <DeployForm
                runId={deployRunId}
                onCancel={() => setDeployRunId(null)}
                onSubmit={async (name, device) => {
                  await deployModel(deployRunId, name, device);
                  setDeployRunId(null);
                }}
              />
            ) : selectedRunId ? (
              <RunDetails
                run={runs.find((r) => r.run_id === selectedRunId)!}
                onClose={() => setSelectedRunId(null)}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-gray-500">
                <p className="text-4xl mb-4">🧠</p>
                <p>Select a run to view details or start a new one.</p>
                <p className="text-xs mt-2 text-gray-600">
                  Training runs execute in the background sidecar.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function RunCard({
  run,
  isSelected,
  onSelect,
  onDeploy,
  onStop,
}: {
  run: TrainingRun;
  isSelected: boolean;
  onSelect: () => void;
  onDeploy: () => void;
  onStop: () => void;
}) {
  const statusColor =
    {
      pending: "bg-gray-500/20 text-gray-400",
      running: "bg-blue-500/20 text-blue-400 animate-pulse",
      completed: "bg-green-500/20 text-green-400",
      failed: "bg-red-500/20 text-red-400",
      cancelled: "bg-yellow-500/20 text-yellow-400",
    }[run.status] || "bg-gray-500";

  return (
    <div
      className={`p-3 rounded-xl bg-white/5 border transition-all cursor-pointer group ${
        isSelected
          ? "border-primary-500 bg-primary-500/10"
          : "border-white/5 hover:border-white/10"
      }`}
      onClick={onSelect}
    >
      <div className="flex justify-between items-start mb-2">
        <span className="font-mono text-[10px] text-gray-400">
          #{run.run_id}
        </span>
        <span
          className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded-full ${statusColor}`}
        >
          {run.status}
        </span>
      </div>
      <div className="text-sm font-medium mb-1 truncate">
        {run.task_type || "Training Task"}
      </div>

      {run.metrics && (
        <div className="grid grid-cols-2 gap-2 mb-2 text-[10px] text-gray-400 font-mono">
          {Object.entries(run.metrics)
            .slice(0, 4)
            .map(([k, v]) => (
              <div key={k}>
                {k}: {typeof v === "number" ? v.toFixed(4) : v}
              </div>
            ))}
        </div>
      )}

      <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity justify-end">
        {run.status === "running" && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onStop();
            }}
            className="text-[10px] px-2 py-1 bg-red-500/20 hover:bg-red-500/40 text-red-300 rounded"
          >
            Stop
          </button>
        )}
        {run.status === "completed" && !run.deployed && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDeploy();
            }}
            className="text-[10px] px-2 py-1 bg-green-500/20 hover:bg-green-500/40 text-green-300 rounded"
          >
            Deploy
          </button>
        )}
        {run.deployed && (
          <span className="text-[10px] px-2 py-1 bg-purple-500/20 text-purple-300 rounded">
            Deployed as {run.tool_name}
          </span>
        )}
      </div>
    </div>
  );
}

function NewRunForm({
  onCancel,
  onSubmit,
}: {
  onCancel: () => void;
  onSubmit: (cfg: any) => Promise<void>;
}) {
  const [config, setConfig] = useState(
    JSON.stringify(
      {
        backbone: "transformer",
        head: "classification",
        data: {
          num_samples: 1000,
          batch_size: 32,
        },
        training: {
          max_epochs: 5,
        },
      },
      null,
      2,
    ),
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const parsed = JSON.parse(config);
      await onSubmit(parsed);
    } catch (e) {
      alert("Invalid JSON config");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col h-full">
      <h3 className="text-lg font-bold mb-4">Start New Training Run</h3>
      <textarea
        value={config}
        onChange={(e) => setConfig(e.target.value)}
        className="flex-1 bg-gray-900 font-mono text-xs p-4 rounded-xl border border-white/10 focus:outline-none focus:border-primary-500 mb-4"
        spellCheck={false}
      />
      <div className="flex justify-end gap-3">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 rounded-lg hover:bg-white/10"
        >
          Cancel
        </button>
        <button
          type="submit"
          className="px-4 py-2 rounded-lg bg-primary-600 hover:bg-primary-500 font-bold"
        >
          Start Training
        </button>
      </div>
    </form>
  );
}

function DeployForm({
  runId,
  onCancel,
  onSubmit,
}: {
  runId: string;
  onCancel: () => void;
  onSubmit: (name: string, device?: string) => Promise<void>;
}) {
  const [toolName, setToolName] = useState("");
  const [device, setDevice] = useState("");

  return (
    <div className="p-6 bg-white/5 rounded-2xl max-w-md mx-auto mt-20">
      <h3 className="text-lg font-bold mb-4">Deploy Model</h3>
      <p className="text-sm text-gray-400 mb-6">
        Deploying run #{runId} as a callable tool.
      </p>

      <label className="block text-xs uppercase text-gray-500 font-bold mb-1">
        Tool Name
      </label>
      <input
        type="text"
        value={toolName}
        onChange={(e) => setToolName(e.target.value)}
        placeholder="e.g. classify_sentiment"
        className="w-full bg-gray-900 rounded-lg px-3 py-2 mb-4 focus:outline-none focus:ring-1 focus:ring-primary-500"
      />

      <label className="block text-xs uppercase text-gray-500 font-bold mb-1">
        Device (Optional)
      </label>
      <input
        type="text"
        value={device}
        onChange={(e) => setDevice(e.target.value)}
        placeholder="cpu, cuda:0"
        className="w-full bg-gray-900 rounded-lg px-3 py-2 mb-6 focus:outline-none focus:ring-1 focus:ring-primary-500"
      />

      <div className="flex justify-end gap-3">
        <button
          type="button"
          onClick={onCancel}
          className="px-4 py-2 rounded-lg hover:bg-white/10"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={() => onSubmit(toolName, device || undefined)}
          disabled={!toolName}
          className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 font-bold disabled:opacity-50"
        >
          Deploy
        </button>
      </div>
    </div>
  );
}

function RunDetails({
  run,
  onClose,
}: {
  run: TrainingRun;
  onClose: () => void;
}) {
  const history = run.history || [];
  const metrics = Object.keys(run.metrics || {}).filter(
    (m) => m !== "epoch" && m !== "batch" && m !== "run_id" && m !== "event",
  );

  return (
    <div className="flex flex-col h-full animate-in fade-in slide-in-from-right-4 duration-300">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-xl font-bold">Run Details: {run.run_id}</h3>
          <span
            className={`text-[10px] uppercase font-bold px-2 py-1 rounded-full ${
              run.status === "running"
                ? "bg-blue-500/20 text-blue-400 animate-pulse"
                : "bg-gray-500/20 text-gray-400"
            }`}
          >
            {run.status}
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-white/10 rounded-lg transition-colors"
        >
          ✕
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6 pb-20">
        {metrics.map((metric) => (
          <div
            key={metric}
            className="bg-white/5 p-4 rounded-2xl border border-white/10 backdrop-blur-md"
          >
            <h4 className="text-xs font-bold uppercase text-gray-400 mb-4">
              {metric}
            </h4>
            <div className="h-48 w-full relative">
              <LineChart
                data={history.map((h) => h.metrics[metric] || 0)}
                color={metric.includes("loss") ? "#ef4444" : "#3b82f6"}
              />
            </div>
            {history.length > 0 &&
              history[history.length - 1].metrics[metric] !== undefined && (
                <div className="mt-2 flex justify-between text-[10px] font-mono text-gray-500">
                  <span>Start</span>
                  <span>
                    Current:{" "}
                    {history[history.length - 1].metrics[metric].toFixed(4)}
                  </span>
                </div>
              )}
          </div>
        ))}

        {metrics.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 text-gray-500 opacity-50">
            <div className="text-4xl mb-4 animate-bounce">📊</div>
            <p>Waiting for metrics from sidecar...</p>
          </div>
        )}
      </div>
    </div>
  );
}

function LineChart({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2)
    return (
      <div className="h-full flex items-center justify-center text-[10px] text-gray-600 italic">
        Insufficient data points for curve...
      </div>
    );

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const padding = range * 0.1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * 100;
      const y = 100 - ((v - (min - padding)) / (range + 2 * padding)) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      className="w-full h-full overflow-visible drop-shadow-[0_0_8px_rgba(0,0,0,0.5)]"
    >
      <defs>
        <linearGradient
          id={`grad-${color.replace("#", "")}`}
          x1="0"
          y1="0"
          x2="0"
          y2="1"
        >
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
        className="transition-all duration-500 ease-in-out"
      />
      <polygon
        fill={`url(#grad-${color.replace("#", "")})`}
        points={`0,100 ${points} 100,100`}
        className="transition-all duration-500 ease-in-out"
      />
    </svg>
  );
}
