import { create } from "zustand";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

/**
 * Snapshot of system resources.
 */
export interface ResourceSnapshot {
  timestamp: number;
  cpu_usage_percent: number;
  memory: { total: number; used: number; free: number; percent: number };
  swap: { total: number; used: number; percent: number };
  gpu?: Record<string, { total_mb: number; used_mb: number; free_mb: number }>;
}

interface ResourceState {
  current: ResourceSnapshot | null;
  history: ResourceSnapshot[];
  maxHistory: number;
  unlisten: UnlistenFn | null;

  initResourceSocket: () => Promise<void>;
}

export const useResourceStore = create<ResourceState>((set, get) => ({
  current: null,
  history: [],
  maxHistory: 30,
  unlisten: null,

  initResourceSocket: async () => {
    if (get().unlisten) return; // Already initialized

    const unlisten = await listen<any>("resource-update", (event) => {
      const snapshot: ResourceSnapshot = {
        ...event.payload,
        timestamp: Date.now(),
      };

      set((state) => {
        const history = [...state.history, snapshot].slice(-state.maxHistory);
        return { current: snapshot, history };
      });
    });

    set({ unlisten });
  },
}));
