import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useAgentStore } from "../stores/agentStore";

interface ExecutionEntry {
  id: string;
  tool_name: string;
  parameters: string;
  success: boolean;
  output: string | null;
  error: string | null;
  duration_ms: number | null;
  created_at: string;
}

const TOOL_COLORS: Record<
  string,
  { bg: string; border: string; text: string; dot: string }
> = {
  shell: {
    bg: "bg-blue-500/10",
    border: "border-blue-500/30",
    text: "text-blue-400",
    dot: "bg-blue-500",
  },
  browser: {
    bg: "bg-green-500/10",
    border: "border-green-500/30",
    text: "text-green-400",
    dot: "bg-green-500",
  },
  code: {
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/30",
    text: "text-yellow-400",
    dot: "bg-yellow-500",
  },
  train: {
    bg: "bg-purple-500/10",
    border: "border-purple-500/30",
    text: "text-purple-400",
    dot: "bg-purple-500",
  },
  canvas: {
    bg: "bg-pink-500/10",
    border: "border-pink-500/30",
    text: "text-pink-400",
    dot: "bg-pink-500",
  },
  database: {
    bg: "bg-cyan-500/10",
    border: "border-cyan-500/30",
    text: "text-cyan-400",
    dot: "bg-cyan-500",
  },
  api: {
    bg: "bg-orange-500/10",
    border: "border-orange-500/30",
    text: "text-orange-400",
    dot: "bg-orange-500",
  },
  diagram: {
    bg: "bg-indigo-500/10",
    border: "border-indigo-500/30",
    text: "text-indigo-400",
    dot: "bg-indigo-500",
  },
  cron: {
    bg: "bg-teal-500/10",
    border: "border-teal-500/30",
    text: "text-teal-400",
    dot: "bg-teal-500",
  },
};

const DEFAULT_COLOR = {
  bg: "bg-gray-500/10",
  border: "border-gray-500/30",
  text: "text-gray-400",
  dot: "bg-gray-500",
};

function getToolColor(name: string) {
  return TOOL_COLORS[name] || DEFAULT_COLOR;
}

function formatDuration(ms: number | null): string {
  if (ms === null) return "";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function tryFormatJson(str: string): string {
  try {
    return JSON.stringify(JSON.parse(str), null, 2);
  } catch {
    return str;
  }
}

export function ExecutionTimeline({ onClose }: { onClose: () => void }) {
  const { agents, activeAgentId } = useAgentStore();
  const state = activeAgentId ? agents[activeAgentId] : undefined;
  const taskId = state?.data?.task_id;

  const [entries, setEntries] = useState<ExecutionEntry[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filterTool, setFilterTool] = useState<string | null>(null);
  const [filterStatus, setFilterStatus] = useState<
    "all" | "success" | "failed"
  >("all");

  useEffect(() => {
    if (!taskId) return;

    const fetchTimeline = async () => {
      try {
        const data = await invoke<ExecutionEntry[]>("get_execution_timeline", {
          taskId,
        });
        setEntries(data);
      } catch (e) {
        console.error("Failed to fetch timeline:", e);
      }
    };

    fetchTimeline();
    const interval = setInterval(fetchTimeline, 3000);
    return () => clearInterval(interval);
  }, [taskId]);

  const filtered = entries.filter((e) => {
    if (filterTool && e.tool_name !== filterTool) return false;
    if (filterStatus === "success" && !e.success) return false;
    if (filterStatus === "failed" && e.success) return false;
    return true;
  });

  const uniqueTools = Array.from(new Set(entries.map((e) => e.tool_name)));
  const successCount = entries.filter((e) => e.success).length;
  const failCount = entries.length - successCount;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-6">
      <div className="glass rounded-2xl w-full max-w-4xl h-[80vh] flex flex-col overflow-hidden border border-white/20 shadow-2xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center text-xl">
              🕐
            </div>
            <div>
              <h2 className="text-xl font-bold">Execution Timeline</h2>
              <p className="text-xs text-gray-400">
                {entries.length} executions &middot; {successCount} ok &middot;{" "}
                {failCount} failed
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Tool filter */}
            <select
              value={filterTool || ""}
              onChange={(e) => setFilterTool(e.target.value || null)}
              className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-sm text-gray-300"
            >
              <option value="">All Tools</option>
              {uniqueTools.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>

            {/* Status filter */}
            <div className="flex rounded-lg border border-white/10 overflow-hidden">
              {(["all", "success", "failed"] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setFilterStatus(s)}
                  className={`px-3 py-1 text-xs transition-colors ${
                    filterStatus === s
                      ? "bg-white/10 text-white"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  {s === "all" ? "All" : s === "success" ? "OK" : "Fail"}
                </button>
              ))}
            </div>

            {/* Close */}
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-white transition-colors"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Timeline Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {!taskId ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <p className="text-4xl mb-4">📋</p>
              <p>No active task. Start an agent task to see the timeline.</p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <p className="text-4xl mb-4">⏳</p>
              <p>
                {entries.length === 0
                  ? "No tool executions yet."
                  : "No executions match the current filters."}
              </p>
            </div>
          ) : (
            <div className="relative">
              {/* Vertical timeline line */}
              <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-white/10" />

              {filtered.map((entry) => {
                const color = getToolColor(entry.tool_name);
                const isExpanded = expandedId === entry.id;

                return (
                  <div key={entry.id} className="relative pl-10 pb-6">
                    {/* Timeline dot */}
                    <div
                      className={`absolute left-2.5 top-4 w-3 h-3 rounded-full ${color.dot} ${
                        !entry.success ? "ring-2 ring-red-500" : ""
                      }`}
                    />

                    {/* Card */}
                    <div
                      className={`rounded-xl p-4 border cursor-pointer transition-colors ${color.bg} ${color.border} hover:bg-white/5`}
                      onClick={() =>
                        setExpandedId(isExpanded ? null : entry.id)
                      }
                    >
                      {/* Summary row */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span
                            className={`font-semibold text-sm ${color.text}`}
                          >
                            {entry.tool_name}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded-full text-xs ${
                              entry.success
                                ? "bg-green-500/20 text-green-400"
                                : "bg-red-500/20 text-red-400"
                            }`}
                          >
                            {entry.success ? "OK" : "FAIL"}
                          </span>
                          {entry.duration_ms != null && (
                            <span className="text-xs text-gray-500 font-mono">
                              {formatDuration(entry.duration_ms)}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-500">
                            {new Date(
                              entry.created_at + "Z",
                            ).toLocaleTimeString()}
                          </span>
                          <span className="text-xs text-gray-600">
                            {isExpanded ? "▲" : "▼"}
                          </span>
                        </div>
                      </div>

                      {/* Expanded details */}
                      {isExpanded && (
                        <div className="mt-3 space-y-2">
                          <div>
                            <span className="text-xs text-gray-400 block mb-1">
                              Parameters
                            </span>
                            <pre className="text-xs bg-gray-900/50 rounded-lg p-2 font-mono overflow-x-auto max-h-32 overflow-y-auto">
                              {tryFormatJson(entry.parameters)}
                            </pre>
                          </div>
                          {entry.output && (
                            <div>
                              <span className="text-xs text-gray-400 block mb-1">
                                Output
                              </span>
                              <pre className="text-xs bg-gray-900/50 rounded-lg p-2 font-mono overflow-x-auto max-h-48 overflow-y-auto whitespace-pre-wrap">
                                {entry.output.length > 2000
                                  ? entry.output.slice(0, 2000) + "..."
                                  : entry.output}
                              </pre>
                            </div>
                          )}
                          {entry.error && (
                            <div>
                              <span className="text-xs text-red-400 block mb-1">
                                Error
                              </span>
                              <pre className="text-xs bg-red-900/20 rounded-lg p-2 font-mono text-red-300">
                                {entry.error}
                              </pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
