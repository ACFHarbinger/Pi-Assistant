import { create } from "zustand";
import { invoke } from "@tauri-apps/api/core";

interface AgentState {
    status: "Idle" | "Running" | "Paused" | "Stopped";
    data?: {
        task_id?: string;
        iteration?: number;
        question?: string;
        reason?: string;
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
    sendMessage: (content: string) => Promise<void>;
    refreshState: () => Promise<void>;
    addMessage: (role: Message["role"], content: string) => void;
    clearError: () => void;
}

export const useAgentStore = create<AgentStore>((set, get) => ({
    state: { status: "Idle" },
    messages: [],
    isLoading: false,
    error: null,

    startAgent: async (task: string) => {
        set({ isLoading: true, error: null });
        try {
            await invoke("start_agent", { task, maxIterations: null });
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
        get().addMessage("user", content);
        try {
            await invoke("send_message", { message: content });
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
}));
