import { useAgentStore } from "./stores/agentStore";
import { AgentStatus } from "./components/AgentStatus";
import { ChatInterface } from "./components/ChatInterface.tsx";
import { TaskInput } from "./components/TaskInput.tsx";
import { PermissionDialog } from "./components/PermissionDialog.tsx";

import Settings from "./components/Settings";
import { useState } from "react";

function App() {
    const { state } = useAgentStore();
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
            {/* Header */}
            <header className="glass sticky top-0 z-50 border-b border-white/10">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
                            <span className="text-xl">🤖</span>
                        </div>
                        <div>
                            <h1 className="text-xl font-bold gradient-text">Pi-Assistant</h1>
                            <p className="text-xs text-gray-400">Universal Agent Harness</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <AgentStatus />
                        <button
                            onClick={() => setIsSettingsOpen(true)}
                            className="p-2 text-gray-400 hover:text-white transition-colors"
                            title="Settings"
                        >
                            ⚙️
                        </button>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="max-w-7xl mx-auto px-6 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Task Input & Status */}
                    <div className="lg:col-span-1 space-y-6">
                        <TaskInput />

                        {/* Status Card */}
                        <div className="glass rounded-2xl p-6">
                            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                                <span className="text-primary-400">📊</span>
                                Agent Status
                            </h2>
                            <div className="space-y-3 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-gray-400">State</span>
                                    <span className="font-medium">{getStateLabel(state)}</span>
                                </div>
                                {state.data && "iteration" in state.data && (
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Iteration</span>
                                        <span className="font-mono">{(state as any).data.iteration}</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Chat Interface */}
                    <div className="lg:col-span-2">
                        <ChatInterface />
                    </div>
                </div>
            </main>

            {/* Permission Dialog */}
            <PermissionDialog />

            {/* Settings Dialog */}
            <Settings isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
        </div>
    );
}

function getStateLabel(state: any): string {
    if (state.status === "Idle") return "Idle";
    if (state.status === "Running") return "Running";
    if (state.status === "Paused") return state.data?.question ? "Waiting for Input" : "Paused";
    if (state.status === "Stopped") return state.data?.reason || "Stopped";
    return "Unknown";
}

export default App;
