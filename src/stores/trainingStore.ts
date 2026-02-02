import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

export interface TrainingRun {
  run_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  started_at?: string;
  completed_at?: string;
  metrics?: Record<string, number>;
  error?: string;
  model_path?: string;
  tool_name?: string;
  deployed: boolean;
  deploy_device?: string;
  task_type?: string;
  config?: any; // Training config
}

interface TrainingState {
  runs: TrainingRun[];
  isLoading: boolean;
  error: string | null;

  fetchRuns: () => Promise<void>;
  startTraining: (config: any) => Promise<string>;
  stopTraining: (runId: string) => Promise<void>;
  deployModel: (
    runId: string,
    toolName: string,
    device?: string,
  ) => Promise<void>;
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  runs: [],
  isLoading: false,
  error: null,

  fetchRuns: async () => {
    set({ isLoading: true, error: null });
    try {
      const runs = await invoke<TrainingRun[]>("sidecar_request", {
        method: "training.list",
        params: {},
      });
      set({ runs, isLoading: false });
    } catch (e: any) {
      console.error("Failed to fetch training runs:", e);
      set({ error: e.toString(), isLoading: false });
    }
  },

  startTraining: async (config: any) => {
    set({ isLoading: true, error: null });
    try {
      const res = await invoke<{ run_id: string }>("sidecar_request", {
        method: "training.start",
        params: config,
      });
      await get().fetchRuns(); // Refresh list
      return res.run_id;
    } catch (e: any) {
      console.error("Failed to start training:", e);
      set({ error: e.toString(), isLoading: false });
      throw e;
    }
  },

  stopTraining: async (runId: string) => {
    try {
      await invoke("sidecar_request", {
        method: "training.stop",
        params: { run_id: runId },
      });
      await get().fetchRuns();
    } catch (e: any) {
      console.error("Failed to stop training:", e);
      set({ error: e.toString() });
    }
  },

  deployModel: async (runId: string, toolName: string, device?: string) => {
    set({ isLoading: true });
    try {
      await invoke("sidecar_request", {
        method: "training.deploy",
        params: { run_id: runId, tool_name: toolName, device },
      });
      await get().fetchRuns();
    } catch (e: any) {
      console.error("Failed to deploy model:", e);
      set({ error: e.toString(), isLoading: false });
      throw e;
    }
  },
}));
