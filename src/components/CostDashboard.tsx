import { useAgentStore } from "../stores/agentStore";

export const CostDashboard: React.FC = () => {
  const state = useAgentStore((s) => s.state);
  const stats = state.data?.cost_stats;

  if (!stats) return null;

  // Approximate cost calc (we could fetch rates from backend eventually)
  // Using simplistic rates for visualization
  const estimatedCost =
    (stats.prompt_tokens * 3.0) / 1000000 +
    (stats.completion_tokens * 15.0) / 1000000;

  return (
    <div className="bg-neutral-900 border border-neutral-800 rounded-lg p-3 text-xs mb-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-neutral-400 font-semibold uppercase tracking-wider">
          Session Usage
        </span>
        <span className="text-emerald-400 font-mono font-bold">
          ${estimatedCost.toFixed(4)}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-neutral-300">
        <div className="flex justify-between">
          <span className="text-neutral-500">Prompt</span>
          <span>{stats.prompt_tokens.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-neutral-500">Output</span>
          <span>{stats.completion_tokens.toLocaleString()}</span>
        </div>
        <div className="flex justify-between col-span-2 border-t border-neutral-800 mt-1 pt-1">
          <span className="text-neutral-500">Total Tokens</span>
          <span className="text-white">
            {stats.total_tokens.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
};
