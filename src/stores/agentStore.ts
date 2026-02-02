import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

interface Subtask {
  id: string;
  parent_id: string | null;
  title: string;
  description: string | null;
  status: "pending" | "running" | "completed" | "failed" | "blocked";
  result: string | null;
  created_at: string;
  updated_at: string;
}

interface PermissionRequest {
  id: string;
  tool_name: string;
  command: string;
  tier: string;
  description: string;
}

export interface AgentState {
  status: "Idle" | "Running" | "Paused" | "Stopped" | "AssistantMessage";
  data?: {
    agent_id?: string;
    task_id?: string;
    iteration?: number;
    task_tree?: Subtask[];
    active_subtask_id?: string | null;
    question?: string;
    reason?: string;
    awaiting_permission?: PermissionRequest;
    content?: string;
    is_streaming?: boolean;
    consecutive_errors?: number;
    cost_stats?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
    reflection?: string;
  };
}

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
}

interface AgentStore {
  agents: Record<string, AgentState>;
  activeAgentId: string | null;
  messages: Message[];
  isLoading: boolean;
  error: string | null;

  // Model Selection
  availableModels: { id: string; provider: string }[];
  selectedModel: string | null;
  selectedProvider: string | null;

  // Actions
  startAgent: (task: string) => Promise<void>;
  stopAgent: (agentId?: string) => Promise<void>;
  pauseAgent: (agentId?: string) => Promise<void>;
  resumeAgent: (agentId?: string) => Promise<void>;
  sendMessage: (content: string, agentId?: string) => Promise<void>;
  sendAnswer: (content: string, agentId?: string) => Promise<void>;
  setActiveAgent: (agentId: string) => void;
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

  // Client-Side AI
  clientAI: {
    isLoaded: boolean;
    model: any;
  };
  setClientAIModel: (model: any) => void;

  // Cost Config
  costConfig: {
    max_tokens_per_session: number | null;
    max_cost_per_session: number | null;
  };
  setCostConfig: (
    config: Partial<{
      max_tokens_per_session: number | null;
      max_cost_per_session: number | null;
    }>,
  ) => void;
}

export const useAgentStore = create<AgentStore>((set, get) => {
  // Initial state and basic properties
  const initialState = {
    agents: {},
    activeAgentId: null,
    messages: [],
    isLoading: false,
    error: null,
    availableModels: [],
    selectedModel: null,
    selectedProvider: null,
    clientAI: {
      isLoaded: false,
      model: null,
    },
    costConfig: {
      max_tokens_per_session: null,
      max_cost_per_session: null,
    },
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
          costConfig: get().costConfig,
        });
        get().addMessage("system", `Starting task: ${task}`);
        await get().refreshState();
      } catch (e) {
        set({ error: String(e) });
      } finally {
        set({ isLoading: false });
      }
    },

    setActiveAgent: (agentId: string) => set({ activeAgentId: agentId }),

    stopAgent: async (agentId?: string) => {
      set({ isLoading: true, error: null });
      const targetId = agentId || get().activeAgentId;
      try {
        await invoke("stop_agent", { agentId: targetId });
        get().addMessage("system", "Agent stopped");
      } catch (e) {
        set({ error: String(e) });
      } finally {
        set({ isLoading: false });
      }
    },

    pauseAgent: async (agentId?: string) => {
      const targetId = agentId || get().activeAgentId;
      try {
        await invoke("pause_agent", { agentId: targetId });
      } catch (e) {
        set({ error: String(e) });
      }
    },

    resumeAgent: async (agentId?: string) => {
      const targetId = agentId || get().activeAgentId;
      try {
        await invoke("resume_agent", { agentId: targetId });
      } catch (e) {
        set({ error: String(e) });
      }
    },

    sendMessage: async (content: string, agentId?: string) => {
      // Optimistic update
      get().addMessage("user", content);

      const { selectedProvider, selectedModel } = get();
      const targetId = agentId || get().activeAgentId;
      try {
        await invoke("send_message", {
          message: content,
          provider: selectedProvider || null,
          model_id: selectedModel || null,
          agentId: targetId,
        });
      } catch (e) {
        set({ error: String(e) });
      }
    },

    sendAnswer: async (content: string, agentId?: string) => {
      // Optimistic update
      get().addMessage("user", content);

      const targetId = agentId || get().activeAgentId;
      try {
        await invoke("answer_question", {
          answer: content,
          agentId: targetId,
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
        // Backend `get_agent_state` might be legacy single-agent.
        // For now, we rely on events.
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
            const isStreaming = newState.data?.is_streaming;
            // const agentId = newState.data?.agent_id;

            if (content) {
              const { messages } = get();
              const lastMessage = messages[messages.length - 1];

              if (isStreaming && lastMessage?.role === "assistant") {
                // Update existing streaming message
                const updatedMessages = [...messages];
                updatedMessages[updatedMessages.length - 1] = {
                  ...lastMessage,
                  content: content,
                };
                set({ messages: updatedMessages });
              } else {
                // New message or final transition
                get().addMessage("assistant", content);
              }
            }
            return;
          }

          // Handle other states
          const agentId = newState.data?.agent_id;
          if (agentId) {
            set((s) => {
              const newAgents = { ...s.agents, [agentId]: newState };
              // Auto-select if no active agent
              const activeId = s.activeAgentId || agentId;
              return { agents: newAgents, activeAgentId: activeId };
            });
          }
        },
      );
      return unlisten;
    },

    // Client-Side AI
    clientAI: {
      isLoaded: false,
      model: null as any, // Type will be handled in hook or refined later to avoid build issues if import fails
    },
    setClientAIModel: (model: any) =>
      set((state) => ({
        clientAI: { ...state.clientAI, isLoaded: true, model },
      })),

    setCostConfig: (config) =>
      set((state) => ({
        costConfig: { ...state.costConfig, ...config },
      })),
  };
});
