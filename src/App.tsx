import { useAgentStore } from "./stores/agentStore";
import { AgentStatus } from "./components/AgentStatus";
import { ChatInterface } from "./components/ChatInterface.tsx";
import { TaskInput } from "./components/TaskInput.tsx";
import { AgentSelector } from "./components/AgentSelector.tsx";
import { PermissionDialog } from "./components/PermissionDialog.tsx";
import { HatchingExperience } from "./components/HatchingExperience.tsx";
import { VoicePanel } from "./components/VoicePanel";
import { Canvas } from "./components/Canvas";
import { TaskTree } from "./components/TaskTree";
import { TrainingDashboard } from "./components/TrainingDashboard";
import { ExecutionTimeline } from "./components/ExecutionTimeline";

import Settings from "./components/Settings";
import { useState, useEffect } from "react";
import init, { init_panic_hook } from "./wasm/pi-core/pi_core";

function App() {
  const { agents, activeAgentId, setupListeners } = useAgentStore();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isCanvasOpen, setIsCanvasOpen] = useState(false);
  const [isTrainingOpen, setIsTrainingOpen] = useState(false);
  const [isTimelineOpen, setIsTimelineOpen] = useState(false);
  const [isHatched, setIsHatched] = useState<boolean | null>(null);

  // Derive active state
  const state =
    activeAgentId && agents[activeAgentId]
      ? agents[activeAgentId]
      : { status: "Idle" as const };

  // Initialize listeners and Wasm
  useEffect(() => {
    const initWasm = async () => {
      try {
        await init();
        init_panic_hook();
        console.log("Core Wasm initialized");
      } catch (e) {
        console.error("Failed to initialize Core Wasm:", e);
      }
    };
    initWasm();

    let unlisten: (() => void) | undefined;
    setupListeners().then((fn) => {
      unlisten = fn;
    });
    return () => {
      if (unlisten) unlisten();
    };
  }, []);

  // Check if hatching has been completed
  useEffect(() => {
    const hatched = localStorage.getItem("pi-hatched") === "true";
    console.log("App: Hatching state check:", hatched);
    setIsHatched(hatched);
  }, []);

  console.log("App: Current state:", {
    status: state.status,
    isHatched,
    activeAgentId,
  });

  // Show hatching experience if not yet hatched
  if (isHatched === null) {
    console.log("App: State is null, showing loading spinner");
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-primary-500 animate-spin text-4xl">⏳</div>
      </div>
    );
  }

  if (!isHatched) {
    console.log("App: Not hatched, showing HatchingExperience");
    return <HatchingExperience onComplete={() => setIsHatched(true)} />;
  }

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
            <AgentStatus state={state} />
            <button
              onClick={() => setIsTimelineOpen(!isTimelineOpen)}
              className={`p-2 transition-colors ${isTimelineOpen ? "text-primary-400" : "text-gray-400 hover:text-white"}`}
              title="Execution Timeline"
            >
              🕐
            </button>
            <button
              onClick={() => setIsCanvasOpen(!isCanvasOpen)}
              className={`p-2 transition-colors ${isCanvasOpen ? "text-primary-400" : "text-gray-400 hover:text-white"}`}
              title="Live Canvas"
            >
              🎨
            </button>
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Settings"
            >
              ⚙️
            </button>
          </div>
        </div>
        <AgentSelector />
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
                    <span className="font-mono">
                      {(state as any).data.iteration}
                    </span>
                  </div>
                )}
                {state.data?.task_tree && state.data.task_tree.length > 0 && (
                  <div className="pt-4 border-t border-white/5">
                    <h3 className="text-sm font-medium mb-3 text-slate-300">
                      Strategy & Subtasks
                    </h3>
                    <TaskTree
                      subtasks={state.data.task_tree}
                      activeSubtaskId={state.data.active_subtask_id}
                    />
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

      {/* Voice Control Panel */}
      <VoicePanel />

      {/* Live Canvas */}
      <Canvas isOpen={isCanvasOpen} onClose={() => setIsCanvasOpen(false)} />

      {/* Training Dashboard */}
      {isTrainingOpen && (
        <TrainingDashboard onClose={() => setIsTrainingOpen(false)} />
      )}

      {/* Settings Dialog */}
      <Settings
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />
    </div>
  );
}

function getStateLabel(state: any): string {
  if (state.status === "Idle") return "Idle";
  if (state.status === "Running") return "Running";
  if (state.status === "Paused")
    return state.data?.question ? "Waiting for Input" : "Paused";
  if (state.status === "Stopped") return state.data?.reason || "Stopped";
  if (state.status === "AssistantMessage") return "Responding...";
  return "Unknown";
}

export default App;
