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

    // Actions
    startAgent: (task: string) => Promise<void>;
    stopAgent: () => Promise<void>;
    pauseAgent: () => Promise<void>;
    resumeAgent: () => Promise<void>;
    sendMessage: (content: string, provider?: string, modelId?: string) => Promise<void>;
    refreshState: () => Promise<void>;
    addMessage: (role: Message["role"], content: string) => void;
    clearError: () => void;
    setupListeners: () => Promise<UnlistenFn>;
}

export const useAgentStore = create<AgentStore>((set, get) => {
    // Initial state and basic properties
    const initialState = {
        state: { status: "Idle" as const },
        messages: [],
        isLoading: false,
        error: null,
    };

    return {
        ...initialState,

        startAgent: async (task: string, provider?: string, modelId?: string) => {
            set({ isLoading: true, error: null });
            try {
                await invoke("start_agent", {
                    task,
                    maxIterations: null,
                    provider: provider || null,
                    modelId: modelId || null,
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

        sendMessage: async (content: string, provider?: string, modelId?: string) => {
            get().addMessage("user", content);
            try {
                await invoke("send_message", {
                    message: content,
                    provider: provider || null,
                    modelId: modelId || null,
                });
            } catch (e) {
                set({ error: String(e) });
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
            const unlisten = await listen<AgentState>("agent-state-changed", (event) => {
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
            });
            return unlisten;
        },
    };
});
