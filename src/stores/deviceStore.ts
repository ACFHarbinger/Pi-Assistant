import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

export interface GpuInfo {
  index: number;
  name: string;
  vendor: string;
  vram_total_mb: number;
  vram_used_mb: number;
  vram_free_mb: number;
  compute_capability: string | null;
}

export interface DeviceCapability {
  device_id: string;
  can_train: boolean;
  can_infer: boolean;
  memory_total_mb: number;
  memory_free_mb: number;
  vendor: string;
}

export interface DeviceInfo {
  cpu: {
    architecture: string;
    cores_physical: number;
    cores_logical: number;
    brand: string;
  };
  ram_total_mb: number;
  ram_available_mb: number;
  gpus: GpuInfo[];
  platform: string;
  capabilities: DeviceCapability[];
}

interface DeviceStore {
  info: DeviceInfo | null;
  loading: boolean;
  error: string | null;

  fetchDeviceInfo: () => Promise<void>;
  refreshMemory: () => Promise<void>;
  migrateModel: (modelId: string, targetDevice: string) => Promise<void>;
  deployModel: (
    runId: string,
    toolName: string,
    device?: string,
  ) => Promise<void>;
}

export const useDeviceStore = create<DeviceStore>((set) => ({
  info: null,
  loading: false,
  error: null,

  fetchDeviceInfo: async () => {
    set({ loading: true, error: null });
    try {
      const info = await invoke<DeviceInfo>("get_device_info");
      set({ info, loading: false });
    } catch (e) {
      set({ error: String(e), loading: false });
    }
  },

  refreshMemory: async () => {
    try {
      await invoke("refresh_device_memory");
      const info = await invoke<DeviceInfo>("get_device_info");
      set({ info });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  migrateModel: async (modelId: string, targetDevice: string) => {
    set({ error: null });
    try {
      await invoke("migrate_model", {
        modelId,
        targetDevice,
      });
      // Refresh device info to reflect updated memory usage
      const info = await invoke<DeviceInfo>("get_device_info");
      set({ info });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  deployModel: async (runId: string, toolName: string, device?: string) => {
    set({ error: null });
    try {
      await invoke("sidecar_request", {
        method: "training.deploy",
        params: { run_id: runId, tool_name: toolName, device },
      });
    } catch (e) {
      set({ error: String(e) });
    }
  },
}));
