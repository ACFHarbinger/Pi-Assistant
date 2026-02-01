import { useState, useEffect, useRef } from "react";
import { useAgentStore } from "../stores/agentStore";

export function ChatInterface() {
  const {
    messages,
    state,
    sendMessage,
    availableModels,
    selectedModel,
    selectedProvider,
    fetchModels,
    setSelectedModel,
    setSelectedProvider,
    sendAnswer,
  } = useAgentStore();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load available models and current selection
  useEffect(() => {
    fetchModels();
  }, []);

  const handleModelChange = async (id: string) => {
    try {
      await setSelectedModel(id);
    } catch (e) {
      console.error("Failed to switch model:", e);
      alert("Failed to switch model: " + e);
    }
  };

  const handleProviderChange = (provider: string) => {
    setSelectedProvider(provider);
  };

  const filteredModels = availableModels.filter(
    (m) => m.provider === selectedProvider,
  );
  const uniqueProviders = Array.from(
    new Set(availableModels.map((m) => m.provider)),
  );

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    if (isWaitingForInput) {
      sendAnswer(input.trim());
    } else {
      sendMessage(input.trim());
    }

    setInput("");
  };

  const isWaitingForInput = state.status === "Paused" && state.data?.question;

  return (
    <div className="glass rounded-2xl h-[600px] flex flex-col">
      {/* Header */}
      <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <span className="text-primary-400">💬</span>
          Chat
        </h2>

        <div className="flex items-center gap-4">
          {/* Provider Selector */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-gray-500 uppercase tracking-tighter font-black">
              Provider:
            </span>
            <select
              value={selectedProvider || ""}
              onChange={(e) => handleProviderChange(e.target.value)}
              className="bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg px-2 py-1 text-[11px] text-gray-300 focus:outline-none focus:ring-1 focus:ring-primary-500/50 appearance-none cursor-pointer capitalize"
            >
              {uniqueProviders.map((p) => (
                <option key={p} value={p} className="bg-gray-900 capitalize">
                  {p}
                </option>
              ))}
            </select>
          </div>

          {/* Model Selector */}
          <div className="flex items-center gap-2 border-l border-white/10 pl-4">
            <span className="text-[10px] text-gray-500 uppercase tracking-tighter font-black">
              Model:
            </span>
            <select
              value={selectedModel || ""}
              onChange={(e) => handleModelChange(e.target.value)}
              className="bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg px-2 py-1 text-[11px] text-gray-300 focus:outline-none focus:ring-1 focus:ring-primary-500/50 appearance-none cursor-pointer"
            >
              {filteredModels.map((m) => (
                <option key={m.id} value={m.id} className="bg-gray-900">
                  {m.id}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <p className="text-4xl mb-2">🤖</p>
              <p>No messages yet. Start a task to begin!</p>
            </div>
          </div>
        ) : (
          messages.map((msg, index) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  msg.role === "user"
                    ? "bg-primary-600 text-white"
                    : msg.role === "system"
                      ? "bg-gray-800 text-gray-300 text-sm italic"
                      : "bg-gray-800 text-gray-100"
                }`}
              >
                <div className="whitespace-pre-wrap">
                  {msg.content === "" &&
                  msg.role === "assistant" &&
                  index === messages.length - 1 &&
                  state.status === "AssistantMessage" &&
                  state.data?.is_streaming ? (
                    <div className="flex items-center gap-2 text-primary-400 font-medium text-xs">
                      <span className="thinking-spinner" />
                      Thinking...
                    </div>
                  ) : (
                    <>
                      {msg.content}
                      {msg.role === "assistant" &&
                        index === messages.length - 1 &&
                        state.status === "AssistantMessage" &&
                        state.data?.is_streaming && (
                          <span className="inline-flex gap-1 ml-2 align-middle">
                            <span className="typing-dot animate-pulse" />
                            <span className="typing-dot animate-pulse" />
                            <span className="typing-dot animate-pulse" />
                          </span>
                        )}
                    </>
                  )}
                </div>
                <p className="text-xs opacity-50 mt-1">
                  {msg.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Question Banner */}
      {isWaitingForInput && (
        <div className="mx-6 mb-4 p-4 rounded-xl bg-yellow-500/20 border border-yellow-500/30">
          <p className="text-yellow-300 text-sm font-medium">
            🤔 Agent is asking: {state.data?.question}
          </p>
        </div>
      )}

      {/* Reflection Banner */}
      {state.data?.reflection && (
        <div className="mx-6 mb-4 p-3 rounded-xl bg-blue-500/10 border border-blue-500/20">
          <div className="flex items-start gap-2">
            <span className="text-blue-400 text-xs mt-0.5">🧠</span>
            <div>
              <p className="text-blue-300 text-xs font-bold uppercase tracking-wider mb-1">
                Reflection
              </p>
              <p className="text-blue-200 text-xs italic">
                "{state.data.reflection}"
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-white/10">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              isWaitingForInput ? "Type your answer..." : "Send a message..."
            }
            className="flex-1 bg-gray-800 rounded-xl px-4 py-3 text-sm placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          />
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-6 py-3 rounded-xl bg-primary-600 text-white font-medium text-sm hover:bg-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
