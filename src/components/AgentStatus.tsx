import { useAgentStore } from "../stores/agentStore";

export function AgentStatus() {
    const { state, stopAgent, pauseAgent, resumeAgent } = useAgentStore();

    const statusConfig = {
        Idle: { color: "bg-gray-500", label: "Idle", pulse: false },
        Running: { color: "bg-green-500", label: "Running", pulse: true },
        Paused: { color: "bg-yellow-500", label: "Paused", pulse: false },
        Stopped: { color: "bg-red-500", label: "Stopped", pulse: false },
        AssistantMessage: { color: "bg-indigo-500", label: "Agent Talking", pulse: true },
    };

    const config = statusConfig[state.status] || statusConfig.Idle;

    return (
        <div className="flex items-center gap-4">
            {/* Status Indicator */}
            <div className="flex items-center gap-2">
                <div
                    className={`w-3 h-3 rounded-full ${config.color} ${config.pulse ? "status-pulse" : ""}`}
                />
                <span className="text-sm font-medium text-gray-300">{config.label}</span>
            </div>

            {/* Control Buttons */}
            <div className="flex items-center gap-2">
                {state.status === "Running" && (
                    <>
                        <button
                            onClick={pauseAgent}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 transition-colors"
                        >
                            Pause
                        </button>
                        <button
                            onClick={stopAgent}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                        >
                            Stop
                        </button>
                    </>
                )}
                {state.status === "Paused" && (
                    <>
                        <button
                            onClick={resumeAgent}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-colors"
                        >
                            Resume
                        </button>
                        <button
                            onClick={stopAgent}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                        >
                            Stop
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}
