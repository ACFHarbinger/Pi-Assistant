import { describe, it, expect, vi, beforeEach } from "vitest";
import { useAgentStore } from "../../stores/agentStore";

// Mock Tauri invoke
const mockInvoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
  invoke: (...args: any[]) => mockInvoke(...args),
}));

// Mock crypto.randomUUID
Object.defineProperty(global, "crypto", {
  value: {
    randomUUID: () => "test-uuid",
  },
});

describe("useAgentStore", () => {
  beforeEach(() => {
    useAgentStore.setState({
      state: { status: "Idle" },
      messages: [],
      isLoading: false,
      error: null,
    });
    mockInvoke.mockReset();
  });

  it("has initial state", () => {
    const state = useAgentStore.getState();
    expect(state.state.status).toBe("Idle");
    expect(state.messages).toEqual([]);
  });

  it("startAgent calls invoke and updates state", async () => {
    mockInvoke.mockResolvedValue({});
    // Mock refreshState response
    mockInvoke.mockImplementation(async (cmd) => {
      if (cmd === "get_agent_state") return { status: "Running" };
      return {};
    });

    await useAgentStore.getState().startAgent("test task");

    expect(mockInvoke).toHaveBeenCalledWith("start_agent", {
      task: "test task",
      maxIterations: null,
      provider: null,
      modelId: null,
    });
    expect(useAgentStore.getState().messages).toHaveLength(1);
    expect(useAgentStore.getState().messages[0].content).toContain(
      "Starting task: test task",
    );
  });

  it("sendMessage adds message and invokes backend", async () => {
    await useAgentStore.getState().sendMessage("hello");

    expect(useAgentStore.getState().messages).toHaveLength(1);
    expect(useAgentStore.getState().messages[0].role).toBe("user");
    expect(useAgentStore.getState().messages[0].content).toBe("hello");
    expect(mockInvoke).toHaveBeenCalledWith("send_message", {
      message: "hello",
      provider: null,
      modelId: null,
    });
  });
});
