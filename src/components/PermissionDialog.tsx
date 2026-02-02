import { useAgentStore } from "../stores/agentStore";
import { invoke } from "@tauri-apps/api/core";

interface PermissionRequest {
  id: string;
  tool_name: string;
  command: string;
  tier: string;
  description: string;
}

export function PermissionDialog() {
  const { agents, activeAgentId } = useAgentStore();
  const state = (activeAgentId ? agents[activeAgentId] : undefined) || {
    status: "Idle",
    data: {},
  };

  // Check if we're in paused state with a permission request
  if (state.status !== "Paused" || !state.data?.awaiting_permission) {
    return null;
  }

  const request = state.data.awaiting_permission as PermissionRequest;

  const handleApprove = async (remember: boolean) => {
    try {
      await invoke("approve_permission", {
        requestId: request.id,
        approved: true,
        remember,
      });
    } catch (error) {
      console.error("Failed to approve permission:", error);
    }
  };

  const handleDeny = async () => {
    try {
      await invoke("approve_permission", {
        requestId: request.id,
        approved: false,
        remember: false,
      });
    } catch (error) {
      console.error("Failed to deny permission:", error);
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case "high":
        return "text-red-400 bg-red-500/20";
      case "medium":
        return "text-yellow-400 bg-yellow-500/20";
      default:
        return "text-green-400 bg-green-500/20";
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="glass rounded-2xl p-6 max-w-lg w-full mx-4 border border-white/20 shadow-2xl">
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-yellow-500/20 flex items-center justify-center">
            <span className="text-xl">🔐</span>
          </div>
          <div>
            <h2 className="text-lg font-bold">Permission Required</h2>
            <p className="text-sm text-gray-400">The agent wants to execute:</p>
          </div>
        </div>

        {/* Tool Info */}
        <div className="bg-gray-800/50 rounded-xl p-4 mb-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-primary-400 font-semibold">
              {request.tool_name}
            </span>
            <span
              className={`px-2 py-0.5 rounded-full text-xs ${getTierColor(request.tier)}`}
            >
              {request.tier}
            </span>
          </div>
          <code className="block text-sm text-gray-300 bg-gray-900/50 rounded-lg p-3 font-mono overflow-x-auto">
            {request.command}
          </code>
          <p className="text-sm text-gray-400 mt-2">{request.description}</p>
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            <button
              onClick={() => handleApprove(false)}
              className="flex-1 px-4 py-2 rounded-xl bg-primary-600 hover:bg-primary-500 font-medium transition-colors"
            >
              Allow Once
            </button>
            <button
              onClick={() => handleApprove(true)}
              className="flex-1 px-4 py-2 rounded-xl bg-green-600 hover:bg-green-500 font-medium transition-colors"
            >
              Always Allow
            </button>
          </div>
          <button
            onClick={handleDeny}
            className="w-full px-4 py-2 rounded-xl bg-gray-700 hover:bg-gray-600 font-medium transition-colors"
          >
            Deny
          </button>
        </div>
      </div>
    </div>
  );
}
