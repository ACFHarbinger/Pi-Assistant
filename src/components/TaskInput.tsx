import { useState } from "react";
import { useAgentStore } from "../stores/agentStore";

export function TaskInput() {
  const { state, startAgent, isLoading, error, clearError } = useAgentStore();
  const [task, setTask] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim()) return;
    startAgent(task.trim());
    setTask("");
  };

  const isDisabled =
    state.status === "Running" || state.status === "Paused" || isLoading;

  return (
    <div className="glass rounded-2xl p-6">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <span className="text-primary-400">🎯</span>
        New Task
      </h2>

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-3 rounded-lg bg-red-500/20 border border-red-500/30 flex items-center justify-between">
          <p className="text-red-300 text-sm">{error}</p>
          <button
            onClick={clearError}
            className="text-red-400 hover:text-red-300 text-sm"
          >
            ✕
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <textarea
          value={task}
          onChange={(e) => setTask(e.target.value)}
          placeholder="Describe what you want the agent to do..."
          rows={4}
          disabled={isDisabled}
          className="w-full bg-gray-800 rounded-xl px-4 py-3 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50 resize-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isDisabled || !task.trim()}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-primary-600 to-accent-600 text-white font-medium hover:from-primary-500 hover:to-accent-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <span className="animate-spin">⏳</span>
              Starting...
            </>
          ) : (
            <>
              <span>🚀</span>
              Start Agent
            </>
          )}
        </button>
      </form>

      {/* Quick Actions */}
      <div className="mt-4 pt-4 border-t border-white/10">
        <p className="text-xs text-gray-500 mb-2">Quick tasks:</p>
        <div className="flex flex-wrap gap-2">
          {[
            "Write a hello world script",
            "List files in current directory",
            "Check system info",
          ].map((quickTask) => (
            <button
              key={quickTask}
              onClick={() => setTask(quickTask)}
              disabled={isDisabled}
              className="text-xs px-3 py-1.5 rounded-lg bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700 disabled:opacity-50 transition-colors"
            >
              {quickTask}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
