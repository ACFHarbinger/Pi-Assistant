import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

interface PermissionRequest {
  id: string;
  tool_name: string;
  command: string;
  tier: string;
  description: string;
}

interface AgentState {
  status: "Idle" | "Running" | "Paused" | "Stopped" | "AssistantMessage";
  data?: {
    task_id?: string;
    iteration?: number;
    question?: string;
    reason?: string;
    awaiting_permission?: PermissionRequest;
    content?: string;
  };
}

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
}

interface AgentStore {
  state: AgentState;
  messages: Message[];
  isLoading: boolean;
  error: string | null;

  // Model Selection
  availableModels: { id: string; provider: string }[];
  selectedModel: string | null;
  selectedProvider: string | null;

  // Actions
  startAgent: (task: string) => Promise<void>;
  stopAgent: () => Promise<void>;
  pauseAgent: () => Promise<void>;
  resumeAgent: () => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  refreshState: () => Promise<void>;
  addMessage: (role: Message["role"], content: string) => void;
  clearError: () => void;
  setupListeners: () => Promise<UnlistenFn>;
  fetchModels: () => Promise<void>;
  loadModel: (modelId: string, backend?: string | null) => Promise<void>;
  unloadModel: (modelId: string) => Promise<void>;
  fetchHistory: () => Promise<void>;

  // Model Selection Actions
  setModels: (models: { id: string; provider: string }[]) => void;
  setSelectedModel: (modelId: string | null) => Promise<void>;
  setSelectedProvider: (provider: string | null) => void;
}

export const useAgentStore = create<AgentStore>((set, get) => {
  // Initial state and basic properties
  const initialState = {
    state: { status: "Idle" as const },
    messages: [],
    isLoading: false,
    error: null,
    availableModels: [],
    selectedModel: null,
    selectedProvider: null,
  };

  return {
    ...initialState,

    startAgent: async (task: string) => {
      set({ isLoading: true, error: null });
      const { selectedProvider, selectedModel } = get();
      try {
        await invoke("start_agent", {
          task,
          maxIterations: null,
          provider: selectedProvider || null,
          modelId: selectedModel || null,
        });
        get().addMessage("system", `Starting task: ${task}`);
        await get().refreshState();
      } catch (e) {
        set({ error: String(e) });
      } finally {
        set({ isLoading: false });
      }
    },

    stopAgent: async () => {
      set({ isLoading: true, error: null });
      try {
        await invoke("stop_agent");
        get().addMessage("system", "Agent stopped");
        await get().refreshState();
      } catch (e) {
        set({ error: String(e) });
      } finally {
        set({ isLoading: false });
      }
    },

    pauseAgent: async () => {
      try {
        await invoke("pause_agent");
        await get().refreshState();
      } catch (e) {
        set({ error: String(e) });
      }
    },

    resumeAgent: async () => {
      try {
        await invoke("resume_agent");
        await get().refreshState();
      } catch (e) {
        set({ error: String(e) });
      }
    },

    sendMessage: async (content: string) => {
      // Optimistic update
      get().addMessage("user", content);

      const { selectedProvider, selectedModel } = get();
      try {
        await invoke("send_message", {
          message: content,
          provider: selectedProvider || null,
          modelId: selectedModel || null,
        });
      } catch (e) {
        set({ error: String(e) });
      }
    },

    fetchModels: async () => {
      try {
        const cloud = await invoke<{
          models: { id: string; provider: string }[];
        }>("get_models_config");

        const local = await invoke<any[]>("list_local_models");
        const loadedLocalModels = local
          .filter((m) => m.loaded)
          .map((m) => ({ id: m.model_id, provider: "local" }));

        const allModels = [...cloud.models, ...loadedLocalModels];
        set({ availableModels: allModels });

        const current = await invoke<string | null>("get_current_model");
        if (current) {
          set({ selectedModel: current });
          const model = allModels.find((m) => m.id === current);
          if (model) set({ selectedProvider: model.provider });
        } else if (allModels.length > 0) {
          set({ selectedProvider: allModels[0].provider });
        }
      } catch (e) {
        console.error("Store: Failed to fetch models:", e);
      }
    },

    loadModel: async (modelId: string, backend: string | null = null) => {
      try {
        await invoke("load_model", { modelId, backend });
        await get().fetchModels();
      } catch (e) {
        console.error("Store: Failed to load model:", e);
        throw e;
      }
    },

    unloadModel: async (modelId: string) => {
      try {
        await invoke("unload_model", { modelId });
        await get().fetchModels();
      } catch (e) {
        console.error("Store: Failed to unload model:", e);
        throw e;
      }
    },

    fetchHistory: async () => {
      try {
        const history = await invoke<any[]>("get_history");
        const messages = history.map((m) => ({
          id: m.id,
          role: m.role as "user" | "assistant" | "system",
          content: m.content,
          timestamp: new Date(m.timestamp),
        }));
        set({ messages });
      } catch (e) {
        console.error("Store: Failed to fetch history:", e);
      }
    },

    setModels: (models) => set({ availableModels: models }),

    setSelectedModel: async (modelId) => {
      try {
        if (modelId) {
          await invoke("save_current_model", { modelId });
          // The original code had `await invoke("load_model", { modelId });` here.
          // The provided edit removes it. Assuming the edit is intentional.
        }
        set({ selectedModel: modelId });
      } catch (e) {
        set({ error: String(e) });
        throw e;
      }
    },

    setSelectedProvider: (provider) => {
      set({ selectedProvider: provider });
      // Automatically select first model of this provider
      const { availableModels } = get();
      const firstModel = availableModels.find((m) => m.provider === provider);
      if (firstModel) {
        get().setSelectedModel(firstModel.id);
      }
    },

    refreshState: async () => {
      try {
        const state = await invoke<AgentState>("get_agent_state");
        set({ state });
      } catch (e) {
        console.error("Failed to refresh state:", e);
      }
    },

    addMessage: (role, content) => {
      // Note: Backend also stores the message. This is for UI responsiveness.
      // We don't want to double-add if we just fetched history.
      const message: Message = {
        id: crypto.randomUUID(),
        role,
        content,
        timestamp: new Date(),
      };
      set((s) => ({ messages: [...s.messages, message] }));
    },

    clearError: () => set({ error: null }),

    setupListeners: async () => {
      // Load initial history
      await get().fetchHistory();

      const unlisten = await listen<AgentState>(
        "agent-state-changed",
        (event) => {
          const newState = event.payload;
          console.log("Agent state changed:", newState);

          if (newState.status === "AssistantMessage") {
            const content = newState.data?.content;
            if (content) {
              get().addMessage("assistant", content);
            }
            return;
          }

          set({ state: newState });
        },
      );
      return unlisten;
    },
  };
});
